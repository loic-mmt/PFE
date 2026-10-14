"""Small optional Lot 5 captures: COT, licensed exports or prospective options."""

import argparse
import json
import os
from pathlib import Path

from pfe import (
    CaptureLimits,
    LicensedEstimateAdapter,
    RawStore,
    archive_option_sample,
    collect_cot,
    collect_option_chain,
)
from pfe.data.lse_transport import LSEDataTransport
from pfe.data.storage import _reject_secrets


def read_small(path, maximum):
    with Path(path).open("rb") as stream:
        body = stream.read(maximum + 1)
    if len(body) > maximum:
        raise ValueError("P3 input exceeds byte budget")
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("cot", "estimates", "options"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--received-at")
    parser.add_argument("--store-dir", type=Path, default=Path("data/raw/p3"))
    parser.add_argument("--max-bytes", type=int, default=524_288)
    parser.add_argument("--max-rows", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        limits = CaptureLimits(max_rows=args.max_rows, max_response_bytes=args.max_bytes)
        config = json.loads(read_small(args.config, min(limits.max_response_bytes, 65536)))
        _reject_secrets(config)
        offline = None
        if args.snapshot:
            if not args.received_at:
                raise ValueError("Offline P3 source requires explicit --received-at")
            offline = (read_small(args.snapshot, limits.max_response_bytes), args.received_at)
        store = RawStore(args.store_dir, max_page_bytes=limits.max_archive_bytes)
        if args.source == "cot":
            result = collect_cot(
                config, store, snapshot_id=args.snapshot_id, offline=offline, limits=limits
            )
        elif args.source == "estimates":
            if offline is None:
                raise ValueError(
                    "Estimates CLI imports licensed exports; "
                    "authenticated clients use Python adapter"
                )
            client = LicensedEstimateAdapter(
                config["license"],
                column_map=config.get("column_map"),
                value_maps=config.get("value_maps"),
            )
            result = client.capture(
                store,
                snapshot_id=args.snapshot_id,
                query=config["query"],
                offline=offline,
                limits=limits,
                format=config.get("format", "json"),
            )
        elif offline is not None:
            result = archive_option_sample(
                json.loads(offline[0]),
                store,
                snapshot_id=args.snapshot_id,
                received_at=offline[1],
                limits=limits,
            )
        else:
            # Lazy credential use; no SDK import or model installation.
            transport = LSEDataTransport(api_key=os.environ.get("LSE_API_KEY", ""))
            result = collect_option_chain(
                transport, store, snapshot_id=args.snapshot_id, limits=limits, **config
            )
        print(f"Saved {args.source}: {result.collection_id}")
        return result.collection_id
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
