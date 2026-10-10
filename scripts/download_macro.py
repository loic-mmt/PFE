"""Download bounded LSE macro JSON samples, without bulk exports.

uv run --extra lse --env-file .env python scripts/download_macro.py \
  --series fdtr cpi_yoy --calendar --start 2025-01-01 --end 2025-01-08 --limit 20
"""

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from lse import LSE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="*", default=[])
    parser.add_argument("--calendar", action="store_true")
    parser.add_argument("--region", default="US")
    parser.add_argument("--start", required=True, help="Inclusive UTC date/time")
    parser.add_argument("--end", required=True, help="Exclusive UTC date/time")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows per query (1..5000)")
    parser.add_argument("--max-bytes", type=int, default=1048576, help="Maximum output size")
    parser.add_argument("--output", type=Path, default=Path("data/raw/lse/macro/sample.json"))
    args = parser.parse_args()
    if not args.series and not args.calendar:
        parser.error("Choose --series and/or --calendar")
    if not 1 <= args.limit <= 5000 or args.max_bytes < 1:
        parser.error("Require limit between 1 and 5000 and positive max-bytes")
    if len(args.series) != len(set(args.series)) or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._^-]{0,63}", name) for name in args.series
    ):
        parser.error("Series identifiers must be unique symbols")
    try:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError
    except ValueError:
        parser.error("--start must precede --end and both must be valid dates")
    if args.output.exists():
        parser.error("Output already exists; choose another --output")
    key = os.environ.get("LSE_API_KEY", "").strip()
    if not key:
        parser.error("Set LSE_API_KEY (or use uv run --env-file .env)")
    payload = dict(
        provider="LSE",
        snapshot_at=datetime.now(UTC).isoformat(),
        start=start.isoformat(),
        end=end.isoformat(),
        limit=args.limit,
        series={},
        calendar=[],
        region=args.region,
        limit_reached={},
    )
    with LSE(api_key=key) as client:
        for name in args.series:
            rows = client.economics(
                name,
                start=start.date().isoformat(),
                end=end.ceil("D").date().isoformat(),
                order="asc",
                limit=args.limit,
            )
            if not isinstance(rows, list) or len(rows) > args.limit:
                raise ValueError(
                    "Provider returned more rows than requested, or an invalid response"
                )
            payload["limit_reached"][name] = len(rows) == args.limit
            payload["series"][name] = [
                row for row in rows if start <= pd.to_datetime(row["date"], utc=True) < end
            ]
        if args.calendar:
            rows = client.economic_calendar(
                region=args.region,
                start=start.date().isoformat(),
                end=end.ceil("D").date().isoformat(),
                order="asc",
                limit=args.limit,
            )
            if not isinstance(rows, list) or len(rows) > args.limit:
                raise ValueError(
                    "Provider returned more rows than requested, or an invalid response"
                )
            payload["limit_reached"]["calendar"] = len(rows) == args.limit
            payload["calendar"] = [
                row for row in rows if start <= pd.to_datetime(row["datetime"], utc=True) < end
            ]
    # Snapshot time describes when this response was received, not publication time.
    payload["snapshot_at"] = datetime.now(UTC).isoformat()
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > args.max_bytes:
        raise ValueError("Output exceeds --max-bytes; lower --limit or request fewer series")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        file.write(encoded)
    counts = {name: len(rows) for name, rows in payload["series"].items()}
    print(
        f"Saved {len(encoded):,} bytes: {args.output}; series={counts}, "
        f"calendar={len(payload['calendar'])}"
    )


if __name__ == "__main__":
    main()
