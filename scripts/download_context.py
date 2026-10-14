"""Collect EIA, Cboe VIX or World Bank context with explicit timing and schemas.

--snapshot reads local source bytes; --run explicitly permits network access.
EIA snapshots may be {pages:[{payload,received_at},...]} or a raw API response
with --received-at. CSV/XLSX snapshots require --received-at. --config supplies
the source query or exact workbook/column/unit selection. No period is used as
publication. Incomplete EIA outputs receive a .partial-N filename.
"""

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

from pfe.data.cboe import collect_cboe_vix
from pfe.data.eia import ContextError, ContextLimits, EIAQuery, collect_eia
from pfe.data.qualification import sanitize_metadata
from pfe.data.storage import ImmutablePageError, RawStore, RawStoreError
from pfe.data.world_bank import collect_pink_sheet


def _read(path, limit):
    with path.open("rb") as file:
        body = file.read(limit + 1)
    if len(body) > limit:
        raise ContextError("Local context input exceeds max_bytes")
    return body


def _json(body):
    try:
        return json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise ContextError("Context configuration/snapshot must be valid JSON") from None


def _publish(path, body):
    if path.exists():
        with path.open("rb") as file:
            existing = file.read(len(body) + 1)
        if existing != body:
            raise ImmutablePageError("Context output already exists with different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(body)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open("rb") as file:
                if file.read(len(body) + 1) != body:
                    raise ImmutablePageError(
                        "Context output already exists with different content"
                    ) from None
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("eia", "cboe", "world_bank"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--received-at", help="Explicit aware receipt timestamp for local source bytes"
    )
    parser.add_argument(
        "--snapshot-id", help="Required online; immutable explicit source snapshot identity"
    )
    parser.add_argument("--store-dir", type=Path, default=Path("data/raw/context"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=25_000)
    parser.add_argument("--max-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=5000)
    args = parser.parse_args(argv)
    try:
        limits = ContextLimits(args.max_rows, args.max_bytes, args.max_requests, args.page_size)
        if args.output.suffix.lower() != ".parquet":
            raise ContextError("Context output must end in .parquet")
        paths = [args.config, args.output, args.output.with_suffix(".manifest.json")]
        if args.snapshot:
            paths.append(args.snapshot)
        if len(set(path.resolve() for path in paths)) != len(paths):
            raise ContextError("Configuration, input, output and manifest must be distinct paths")
        config = _json(_read(args.config, limits.max_bytes))
        if not isinstance(config, dict):
            raise ContextError("Context config must be an object")
        config = sanitize_metadata(config, secrets=(os.environ.get("EIA_API_KEY", ""),))
        offline = _read(args.snapshot, limits.max_bytes) if args.snapshot else None
        snapshot_id = args.snapshot_id
        if offline is not None and not snapshot_id:
            snapshot_id = "offline-" + hashlib.sha256(offline).hexdigest()[:24]
        if not snapshot_id:
            raise ContextError("Online context collection requires explicit --snapshot-id")
        transport = None
        if args.run:
            from pfe.data.http import BoundedHTTPTransport

            transport = BoundedHTTPTransport(
                allowed_hosts={"api.eia.gov", "cdn-api.cboe.com", "thedocs.worldbank.org"}
            )
        store = RawStore(args.store_dir, max_page_bytes=limits.max_bytes)
        if args.source == "eia":
            pages = None
            if offline is not None:
                payload = _json(offline)
                if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
                    pages = payload["pages"]
                else:
                    if not args.received_at:
                        raise ContextError("Raw EIA snapshot requires explicit --received-at")
                    pages = [{"payload": payload, "received_at": args.received_at}]
            result = collect_eia(
                transport,
                store,
                query=EIAQuery(**config),
                snapshot_id=snapshot_id,
                api_key=os.environ.get("EIA_API_KEY"),
                limits=limits,
                offline_pages=pages,
            )
        else:
            if offline is not None and not args.received_at:
                raise ContextError("Local CSV/XLSX requires explicit --received-at")
            saved = (offline, args.received_at) if offline is not None else None
            collector = collect_cboe_vix if args.source == "cboe" else collect_pink_sheet
            result = collector(
                transport, store, snapshot_id=snapshot_id, limits=limits, offline=saved, **config
            )
        output = args.output
        if result.complete:
            body = store.read_page(result.collection_id, "normalized")
        else:
            output = output.with_name(output.stem + f".partial-{len(result.observations)}.parquet")
            buffer = io.BytesIO()
            result.observations.to_parquet(buffer)
            body = buffer.getvalue()
            if len(body) > limits.max_bytes:
                raise ContextError("Partial context output exceeds max_bytes")
        _publish(output, body)
        _publish(
            output.with_suffix(".manifest.json"),
            json.dumps(result.metadata, sort_keys=True, indent=2, allow_nan=False).encode(),
        )
        print(
            f"Saved {len(result.observations):,} {args.source} observations: {output}; "
            f"complete={result.complete}, collection_id={result.collection_id}"
        )
        return result.collection_id
    except (ContextError, RawStoreError, OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
