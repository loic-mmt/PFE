"""Persist a bounded metadata audit; offline by explicit snapshot, online by --run.

Offline example (snapshot needs audited_at, catalog, reference and authority flags)::

    python scripts/qualify_lse.py --snapshot discovery.json --audit-id sample-1

Online metadata only, never price/whole-dataset exports::

    python scripts/qualify_lse.py --run --audit-id lse-audit-20261011 --symbol QQQ

Repeating an audit ID and request reuses immutable pages; use a new ID for a fresh audit.
"""

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from pfe.data.qualification import (
    BoundedLSETransport,
    DiscoveryError,
    DiscoveryLimitError,
    DiscoveryLimits,
    SymbolRequest,
    collect_lse_discovery,
    qualify_lse_snapshot,
    sanitize_metadata,
)
from pfe.data.sources import default_source_registry
from pfe.data.storage import RawStore, RawStoreError


def _snapshot(path, max_bytes):
    with path.open("rb") as file:
        raw = file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise DiscoveryLimitError("LSE snapshot exceeds max_bytes")
    try:
        snapshot = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise DiscoveryError("Invalid LSE snapshot JSON") from None
    if not isinstance(snapshot, dict):
        raise DiscoveryError("LSE snapshot must be a JSON object")
    snapshot = sanitize_metadata(snapshot, secrets=(os.environ.get("LSE_API_KEY", ""),))
    # Credentials and irrelevant whitespace never change an audit's identity.
    identity = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return snapshot, hashlib.sha256(identity).hexdigest()


def _symbol(value):
    parts = value.split("@")
    if len(parts) > 2:
        raise argparse.ArgumentTypeError("Use exact SYMBOL or SYMBOL@DATASET")
    try:
        return SymbolRequest(parts[0], parts[1] if len(parts) == 2 else None)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _read(store, collection_id, page_id):
    return json.loads(store.read_page(collection_id, page_id))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--snapshot", type=Path, help="Offline saved discovery JSON; no key or network"
    )
    mode.add_argument(
        "--run", action="store_true", help="Explicitly execute metadata-only HTTP GETs"
    )
    parser.add_argument("--audit-id", help="Immutable audit identity; required online")
    parser.add_argument(
        "--symbol",
        action="append",
        type=_symbol,
        default=[],
        help="Exact case-sensitive SYMBOL or SYMBOL@DATASET; repeatable",
    )
    parser.add_argument("--store-dir", type=Path, default=Path("data/raw/lse/discovery"))
    parser.add_argument("--max-rows", type=int, default=25_000)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=1_048_576,
        help="Total discovery response/input and per-stored-page byte cap",
    )
    parser.add_argument("--max-requests", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        limits = DiscoveryLimits(args.max_rows, args.max_bytes, args.max_requests)
        offline, digest = (
            _snapshot(args.snapshot, limits.max_bytes) if args.snapshot else (None, None)
        )
        audit_id = args.audit_id or (offline or {}).get("audit_id")
        if offline is not None and not audit_id:
            audit_id = f"offline-{digest[:24]}"
        if not isinstance(audit_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", audit_id
        ):
            parser.error("Require a safe --audit-id (explicitly required with --run)")
        if len(args.symbol) > limits.max_rows:
            raise DiscoveryLimitError("Requested symbols exceed max_rows")
        request = {
            "audit_id": audit_id,
            "mode": "offline" if offline is not None else "metadata",
            "limits": asdict(limits),
            "symbols": [asdict(symbol) for symbol in args.symbol],
        }
        if digest is not None:
            request["snapshot_sha256"] = digest
        store = RawStore(args.store_dir, max_page_bytes=limits.max_bytes)
        collection_id = store.begin(
            "lse",
            "discovery",
            request,
            source_metadata=default_source_registry().get("lse").to_dict(),
        )
        # Recover a committed raw/journal pair left behind by an interrupted
        # manifest write. Refetching would change receipt timestamps and conflict.
        store.recover(collection_id)
        manifest = store.manifest(collection_id)
        page_ids = {page["page_id"] for page in manifest["pages"]}
        if "qualification" in page_ids:
            report = _read(store, collection_id, "qualification")
        else:
            if "snapshot" in page_ids:
                snapshot = _read(store, collection_id, "snapshot")
            elif offline is not None:
                snapshot = {**offline, "audit_id": audit_id, "mode": "offline"}
            else:
                key = os.environ.get("LSE_API_KEY", "").strip()
                if not key:
                    parser.error("Set LSE_API_KEY to execute --run")
                transport = BoundedLSETransport(key)
                cached = {
                    name: _read(store, collection_id, name)
                    for name in ("meta", "catalog", "reference", "usage")
                    if name in page_ids
                }

                def save_endpoint(name, page):
                    count = len(page["response"]) if isinstance(page["response"], list) else None
                    store.write_page(collection_id, name, page, row_count=count)

                snapshot = collect_lse_discovery(
                    transport,
                    audit_id=audit_id,
                    audited_at=manifest["created_at"],
                    limits=limits,
                    cached_pages=cached,
                    on_page=save_endpoint,
                )
            report = qualify_lse_snapshot(snapshot, symbols=args.symbol, limits=limits)
            # Reports expand metadata; enforce the output cap before committing a snapshot.
            encoded = json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(encoded) > limits.max_bytes:
                raise DiscoveryLimitError("LSE qualification report exceeds max_bytes")
            if "snapshot" not in page_ids:
                store.write_page(collection_id, "snapshot", sanitize_metadata(snapshot))
            complete = (
                report["catalogue_coverage"]["complete"]
                and report["catalogue_coverage"]["authoritative"]
                and report["reference_coverage"]["complete"]
                and report["reference_coverage"]["authoritative"]
            )
            store.write_page(
                collection_id,
                "qualification",
                report,
                complete=complete,
                row_count=len(report["matrix"]),
                metadata={"completion_scope": "metadata_discovery", "incomplete": not complete},
            )
        print(
            f"Saved LSE qualification: collection_id={collection_id}, "
            f"catalogue_complete={report['catalogue_coverage']['complete']}, "
            f"instruments={len(report['matrix'])}, requirements={len(report['requirements'])}; "
            f"store={store.root}"
        )
        return collection_id
    except (DiscoveryError, RawStoreError, OSError, ValueError) as error:
        # Provider HTTP bodies never reach this handler; paths and messages contain no key.
        parser.error(str(error))


if __name__ == "__main__":
    main()
