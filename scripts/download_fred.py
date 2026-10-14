"""Collect bounded FRED/ALFRED v1 pages, with explicit date-only vintage policy.

Offline JSON: {"responses": [{"endpoint": "/series", "received_at": "...Z",
"payload": {...native FRED JSON...}}, ...]}. Paginated entries use an offset;
release/dates entries additionally use release_id. All API-native count/offset/
limit fields are checked. No key or network is needed offline or for cached replay.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from pfe.data.contracts import OBSERVATION_COLUMNS, normalize_observations
from pfe.data.fred import FREDError, collect_fred_series
from pfe.data.http import BoundedHTTPTransport, HTTPResponse
from pfe.data.qualification import sanitize_metadata
from pfe.data.storage import RawStore, RawStoreError, _atomic_write, _locked


def _snapshot(path, cap):
    with path.open("rb") as stream:
        body = stream.read(cap + 1)
    if len(body) > cap:
        raise FREDError("Offline FRED snapshot exceeds max_bytes")
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise FREDError("Offline FRED snapshot must be valid JSON") from None
    return sanitize_metadata(data, secrets=(os.environ.get("FRED_API_KEY", ""),))


class _OfflineTransport:
    def __init__(self, snapshot):
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("responses"), list):
            raise FREDError("Offline snapshot requires a responses array")
        self.responses = snapshot["responses"]

    def get(self, url, *, params, max_bytes):
        endpoint = urlsplit(url).path.removeprefix("/fred")
        matching = [
            entry
            for entry in self.responses
            if isinstance(entry, dict)
            and entry.get("endpoint") == endpoint
            and entry.get("offset", 0) == params.get("offset", 0)
            and (endpoint != "/release/dates" or entry.get("release_id") == params["release_id"])
        ]
        if len(matching) != 1:
            raise FREDError("Offline FRED response page missing or ambiguous")
        entry = matching[0]
        try:
            body = json.dumps(entry["payload"], allow_nan=False).encode()
        except (KeyError, ValueError, TypeError, RecursionError):
            raise FREDError("Invalid offline FRED response") from None
        if len(body) > max_bytes:
            raise FREDError("Offline FRED page exceeds total byte cap")
        return HTTPResponse(body, entry.get("received_at"), "application/json")


def _same(left, right):
    absent_left, absent_right = pd.isna(left), pd.isna(right)
    if absent_left or absent_right:
        return bool(absent_left and absent_right)
    return bool(left == right)


def _append_history(path, incoming):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path.parent):
        prior = (
            normalize_observations(pd.read_parquet(path)) if path.exists() else incoming.iloc[:0]
        )
        merged = pd.concat([prior, incoming], ignore_index=True)
        keys = ["source_id", "entity_id", "metric", "observation_date", "revision_id"]
        semantic = [field for field in OBSERVATION_COLUMNS if field != "available_at"]
        semantic += [
            "availability_basis",
            "availability_convention",
            "vintage_timezone",
            "frequency",
            "seasonal_adjustment",
        ]
        for _, group in merged.groupby(keys, dropna=False, sort=False):
            if len(group) > 1 and any(
                not _same(group.iloc[0][field], value)
                for field in semantic
                for value in group[field]
            ):
                raise FREDError("Conflicting immutable FRED revision or availability convention")
        merged = merged.sort_values("available_at", kind="stable").drop_duplicates(
            keys, keep="first"
        )
        merged = normalize_observations(merged)
        output = io.BytesIO()
        merged.to_parquet(output, index=False)
        _atomic_write(path, output.getvalue(), replace=True)
    return len(merged)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path, help="Offline API response JSON")
    mode.add_argument("--run", action="store_true", help="Explicit bounded FRED HTTP GETs")
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--start", required=True, help="Economic observation date start YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Economic observation date end YYYY-MM-DD")
    parser.add_argument("--realtime-start", required=True)
    parser.add_argument("--realtime-end", required=True)
    parser.add_argument("--snapshot-id", help="Required online; new ID for a new capture")
    parser.add_argument(
        "--availability", choices=("receipt", "vintage_next_day"), default="receipt"
    )
    parser.add_argument("--vintage-timezone", default="America/New_York")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--max-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--store-dir", type=Path, default=Path("data/raw/fred"))
    parser.add_argument("--output", type=Path, default=Path("data/macro/us_macro_vintages.parquet"))
    args = parser.parse_args(argv)
    try:
        if type(args.max_bytes) is not int or args.max_bytes < 1:
            raise FREDError("max_bytes must be positive")
        key = None
        if args.snapshot:
            snapshot = _snapshot(args.snapshot, args.max_bytes)
            digest = hashlib.sha256(
                json.dumps(snapshot, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            snapshot_id = (args.snapshot_id or "offline") + ":" + digest
            transport = _OfflineTransport(snapshot)
        else:
            if not args.snapshot_id:
                parser.error("Online FRED collection requires --snapshot-id")
            snapshot_id = args.snapshot_id
            key = os.environ.get("FRED_API_KEY", "").strip() or None
            transport = (
                BoundedHTTPTransport(("api.stlouisfed.org",), min_interval=1) if key else None
            )
        collection_id, observations, summary = collect_fred_series(
            transport,
            RawStore(args.store_dir, max_page_bytes=args.max_bytes),
            series_id=args.series_id,
            start=args.start,
            end=args.end,
            snapshot_id=snapshot_id,
            realtime_start=args.realtime_start,
            realtime_end=args.realtime_end,
            api_key=key,
            availability=args.availability,
            vintage_timezone=args.vintage_timezone,
            page_size=args.page_size,
            max_rows=args.max_rows,
            max_bytes=args.max_bytes,
            max_requests=args.max_requests,
        )
        total = _append_history(args.output, observations)
        print(
            f"Saved FRED: collection_id={collection_id}, observations={len(observations)}, "
            f"history_rows={total}, pagination_complete={summary['pagination_complete']}; "
            f"output={args.output}"
        )
        return collection_id
    except (ValueError, OSError, RawStoreError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
