"""Capture one small Lot 4 query, offline by snapshot or online with --run.

No bulk archives, crawling or model downloads. Config contains exact provider
identifiers, periods, dimensions, units and fiscal/calendar declarations.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

from pfe.data.eia import ContextError
from pfe.data.international import INTERNATIONAL_SOURCES, InternationalLimits, collect_international
from pfe.data.storage import RawStore, RawStoreError
from pfe.features.international import compute_international_features


def _read(path, maximum):
    with Path(path).open("rb") as stream:
        body = stream.read(maximum + 1)
    if len(body) > maximum:
        raise ContextError("International input exceeds byte budget")
    return body


def _publish(path, body):
    from tempfile import NamedTemporaryFile

    path = Path(path)
    if path.exists():
        if _read(path, len(body)) != body:
            raise ContextError("International output already exists with different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".international-", delete=False) as file:
        temporary = Path(file.name)
        file.write(body)
        file.flush()
        os.fsync(file.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read(path, len(body)) != body:
                raise ContextError(
                    "International output already exists with different content"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=tuple(INTERNATIONAL_SOURCES), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--received-at")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--store-dir", type=Path, default=Path("data/raw/international"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--features-config", type=Path, help="JSON {decisions, series}; optional local features"
    )
    parser.add_argument("--features-output", type=Path)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-response-bytes", type=int, default=524_288)
    parser.add_argument("--max-archive-bytes", type=int, default=4_194_304)
    args = parser.parse_args(argv)
    try:
        limits = InternationalLimits(args.max_rows, args.max_response_bytes, args.max_archive_bytes)
        if args.output.suffix != ".parquet":
            raise ContextError("International output must be Parquet")
        if bool(args.features_config) != bool(args.features_output):
            raise ContextError("Provide both features config and output")
        paths = [args.config, args.output, args.output.with_suffix(".manifest.json")]
        if args.snapshot:
            paths.append(args.snapshot)
        if args.features_config:
            paths.extend([args.features_config, args.features_output])
        if len({p.resolve() for p in paths}) != len(paths):
            raise ContextError("International input/config/output paths must be distinct")
        config = json.loads(_read(args.config, min(limits.max_response_bytes, 65_536)))
        offline = None
        if args.snapshot:
            if not args.received_at:
                raise ContextError("Local snapshot requires explicit aware --received-at")
            offline = (_read(args.snapshot, limits.max_response_bytes), args.received_at)
        snapshot_id = args.snapshot_id
        if snapshot_id is None and offline:
            snapshot_id = "offline-" + hashlib.sha256(offline[0]).hexdigest()[:24]
        if not snapshot_id:
            raise ContextError("Online collection requires explicit --snapshot-id")
        key_name = {"opendart": "OPENDART_API_KEY", "estat": "ESTAT_APP_ID"}.get(args.source)
        store = RawStore(args.store_dir, max_page_bytes=limits.max_archive_bytes)
        result = collect_international(
            args.source,
            config,
            store,
            snapshot_id=snapshot_id,
            limits=limits,
            offline=offline,
            credential=os.environ.get(key_name) if key_name else None,
        )
        output = (
            args.output
            if result.complete
            else args.output.with_name(args.output.stem + ".partial.parquet")
        )
        normalized = store.read_page(result.collection_id, "normalized")
        metadata = json.dumps(
            result.metadata, indent=2, sort_keys=True, ensure_ascii=False
        ).encode()
        extra = None
        if args.features_config:
            selection = json.loads(_read(args.features_config, 65_536))
            if (
                set(selection) != {"decisions", "series"}
                or len(selection["decisions"]) > 512
                or len(selection["series"]) > 32
            ):
                raise ContextError("Feature config requires at most 512 decisions and 32 series")
            if args.features_output.suffix != ".parquet":
                raise ContextError("International feature output must be Parquet")
            stream = io.BytesIO()
            compute_international_features(
                selection["decisions"], result.observations, series=selection["series"]
            ).to_parquet(stream)
            extra = stream.getvalue()
        if (
            len(normalized) + len(metadata) + (len(extra) if extra else 0)
            > limits.max_archive_bytes
        ):
            raise ContextError("Published outputs exceed byte budget")
        _publish(output, normalized)
        _publish(output.with_suffix(".manifest.json"), metadata)
        if extra is not None:
            _publish(args.features_output, extra)
        print(
            f"Saved {len(result.observations)} {args.source} rows; "
            f"complete={result.complete}; {output}"
        )
        return result.collection_id
    except (ContextError, RawStoreError, OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
