"""Collect bounded SEC metadata/XBRL with immutable raw cache and canonical output."""

import argparse
import os
import tempfile
from pathlib import Path

from pfe.data.sec import SECClient


def _save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        frame.to_parquet(temporary, index=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None, *, transport=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cik", required=True)
    parser.add_argument("--user-agent", required=True, help="Application name and contact email")
    parser.add_argument("--audit-id", required=True, help="New ID for a refreshed snapshot")
    parser.add_argument("--store-dir", type=Path, default=Path("data/raw/sec"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fundamentals/sec"))
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-filings", type=int, default=1000)
    parser.add_argument("--max-bytes", type=int, default=4194304)
    parser.add_argument("--forms", nargs="+")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--recent-only", action="store_true")
    parser.add_argument(
        "--availability",
        choices=("receipt", "publication"),
        default="receipt",
        help=(
            "publication is a legacy acceptance+latency reconstruction, "
            "never proof of dissemination"
        ),
    )
    parser.add_argument("--latency", default="0s")
    args = parser.parse_args(argv)
    try:
        client = SECClient(
            args.user_agent, transport=transport, store=args.store_dir, max_bytes=args.max_bytes
        )
        data = client.download(
            args.cik,
            audit_id=args.audit_id,
            include_history=not args.recent_only,
            max_pages=args.max_pages,
            max_filings=args.max_filings,
            forms=args.forms,
            start=args.start,
            end=args.end,
            availability=args.availability,
            latency=args.latency,
        )
        for name, frame in (("filings", data.filings), ("company_facts", data.facts)):
            destination = args.output_dir / f"{name}_{data.collection_id}.parquet"
            if destination.exists():
                import pandas as pd

                saved = pd.read_parquet(destination)
                saved = saved.astype(frame.dtypes.to_dict())
                if saved.attrs.get("raw_collection_id") != data.collection_id or not saved.equals(
                    frame
                ):
                    raise ValueError("SEC canonical output differs from cached collection")
            else:
                _save(frame, destination)
        print(
            f"Saved SEC filings={len(data.filings)}, facts={len(data.facts)}; "
            f"complete={data.complete}; collection_id={data.collection_id}"
        )
        return data
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
