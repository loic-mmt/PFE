"""Download bounded LSE corporate-action pages and append validated revisions.

Explicit provider schema map JSON::

    {"split": {"symbol": "symbol", "action_id": "provider_id", "effective_date": "effective_date",
               "ratio": "provider_ratio"},
     "dividend": {"symbol": "symbol", "action_id": "provider_id",
                  "effective_date": "effective_date",
                  "amount": "provider_cash", "currency": "provider_currency"}}

Offline snapshot JSON contains split/dividend arrays in that mapped schema.
No provider numeric field, split direction or currency is guessed. Full-limit
pages remain incomplete. Use a new snapshot ID for a new provider vintage.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd

from pfe.data.corporate_actions import (
    ACTION_COLUMNS,
    collect_lse_actions,
    normalize_corporate_actions,
)
from pfe.data.lse_transport import LSEDataTransport
from pfe.data.qualification import (
    DiscoveryError,
    DiscoveryLimitError,
    DiscoveryResponse,
    sanitize_metadata,
)
from pfe.data.sources import default_source_registry
from pfe.data.storage import RawStore, RawStoreError, _atomic_write, _locked


def _json(path, cap):
    with path.open("rb") as stream:
        body = stream.read(cap + 1)
    if len(body) > cap:
        raise DiscoveryLimitError("Action input exceeds max_bytes")
    try:
        return json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("Action input requires valid JSON") from None


class _OfflineTransport:
    def __init__(self, snapshot):
        if not isinstance(snapshot, dict) or set(snapshot) != {"split", "dividend"}:
            raise ValueError("Offline snapshot requires split and dividend arrays")
        self.snapshot = snapshot

    def get(self, endpoint, params, *, max_bytes):
        kind = "split" if endpoint == "/ref/stock_splits" else "dividend"
        rows = self.snapshot[kind]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Offline action pages require arrays of objects")
        body = json.dumps(rows, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(body) > max_bytes:
            raise DiscoveryLimitError("Offline action response exceeds max_bytes")
        if len(rows) > params["limit"]:
            raise DiscoveryLimitError("Offline action response exceeds row limit")
        return DiscoveryResponse(rows, byte_count=len(body))


def _same(left, right):
    left_missing, right_missing = pd.isna(left), pd.isna(right)
    if left_missing or right_missing:
        return bool(left_missing and right_missing)
    return bool(left == right)


def _append_history(path, incoming):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic read/merge/replace protects concurrent snapshots targeting one file.
    with _locked(path.parent):
        history = pd.read_parquet(path) if path.exists() else incoming.iloc[:0]
        merged = pd.concat([history, incoming], ignore_index=True)
        keys = ["source_id", "action_id", "revision_id"]
        semantic = [name for name in ACTION_COLUMNS if name != "available_at"]
        semantic += ["status", "amount_unit", "dividend_basis"]
        for _, group in merged.groupby(keys, dropna=False, sort=False):
            first = group.iloc[0]
            if len(group) > 1 and any(
                not _same(first[name], value) for name in semantic for value in group[name]
            ):
                raise ValueError(
                    "Existing action revision conflicts with the mapped incoming vintage"
                )
        # An identical provider revision received twice keeps its first availability.
        merged = merged.sort_values("available_at", kind="stable").drop_duplicates(
            keys, keep="first"
        )
        merged = normalize_corporate_actions(merged)
        buffer = io.BytesIO()
        merged.to_parquet(buffer, index=False)
        _atomic_write(path, buffer.getvalue(), replace=True)
    return len(merged)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--run", action="store_true", help="Explicitly fetch two bounded LSE reference pages"
    )
    modes.add_argument(
        "--snapshot", type=Path, help="Offline JSON: split/dividend provider row arrays"
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--entity-id", required=True)
    parser.add_argument("--instrument-id", required=True)
    parser.add_argument(
        "--start",
        required=True,
        help="Provider effective-date window start YYYY-MM-DD; boundary semantics unqualified",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Provider date-window end YYYY-MM-DD; SDK boundary semantics unqualified",
    )
    parser.add_argument("--snapshot-id", help="Required online; replay reuses immutable raw pages")
    parser.add_argument(
        "--field-map",
        required=True,
        type=Path,
        help="JSON split/dividend canonical-to-provider field maps",
    )
    parser.add_argument("--ratio-convention", required=True, choices=("new_per_old", "old_per_new"))
    parser.add_argument(
        "--dividend-basis",
        choices=("pre_split_share", "post_split_share", "unknown"),
        default="unknown",
    )
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-bytes", type=int, default=1_048_576)
    parser.add_argument("--store-dir", type=Path, default=Path("data/raw/lse/actions"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/market/corporate_actions.parquet")
    )
    args = parser.parse_args(argv)
    try:
        if type(args.max_bytes) is not int or args.max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        field_maps = _json(args.field_map, args.max_bytes)
        source_metadata = default_source_registry().get("lse").to_dict()
        if args.snapshot:
            snapshot = sanitize_metadata(
                _json(args.snapshot, args.max_bytes), secrets=(os.environ.get("LSE_API_KEY", ""),)
            )
            identity = json.dumps(
                snapshot, sort_keys=True, ensure_ascii=False, allow_nan=False
            ).encode()
            digest = hashlib.sha256(identity).hexdigest()
            snapshot_id = (args.snapshot_id or "offline") + ":" + digest
            source_metadata["offline_snapshot_sha256"] = digest
            transport = _OfflineTransport(snapshot)
        else:
            if not args.snapshot_id:
                parser.error("Online collection requires an explicit --snapshot-id")
            key = os.environ.get("LSE_API_KEY", "").strip()
            snapshot_id = args.snapshot_id
            transport = LSEDataTransport(key) if key else None
        collection_id, actions, summary = collect_lse_actions(
            transport,
            RawStore(args.store_dir, max_page_bytes=args.max_bytes),
            symbol=args.symbol,
            entity_id=args.entity_id,
            instrument_id=args.instrument_id,
            start=args.start,
            end=args.end,
            snapshot_id=snapshot_id,
            field_maps=field_maps,
            limit=args.limit,
            max_bytes=args.max_bytes,
            ratio_convention=args.ratio_convention,
            dividend_basis=args.dividend_basis,
            source_metadata=source_metadata,
        )
        total = _append_history(args.output, actions)
        print(
            f"Saved corporate actions: collection_id={collection_id}, rows={len(actions)}, "
            f"history_rows={total}, window_complete={summary['window_complete']}; "
            f"output={args.output}"
        )
        return collection_id
    except (DiscoveryError, RawStoreError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
