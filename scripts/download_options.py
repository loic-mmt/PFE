"""Download small LSE options samples, without pagination or bulk exports."""

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
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--flow", action="store_true")
    parser.add_argument("--contract", help="OSI ticker: download this contract's 1-minute candles")
    parser.add_argument("--expiry", help="Filter chain/flow by YYYY-MM-DD expiration")
    parser.add_argument("--type", choices=("call", "put"))
    parser.add_argument("--strike-min", type=float)
    parser.add_argument("--strike-max", type=float)
    parser.add_argument(
        "--start", help="Inclusive UTC date/time for flow/candles, not chain history"
    )
    parser.add_argument("--end", help="Exclusive UTC date/time for flow/candles")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=1048576)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.ticker = args.ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.]{0,14}", args.ticker):
        parser.error("--ticker must be a ticker symbol")
    if not (args.chain or args.flow or args.contract):
        parser.error("Choose --chain, --flow and/or --contract")
    if not 1 <= args.limit <= 5000 or args.max_bytes < 1:
        parser.error("Require limit 1..5000 and positive max-bytes")
    if bool(args.start) != bool(args.end) or (args.start and not (args.flow or args.contract)):
        parser.error("--start/--end must be provided together for flow/candles")
    start, end = None, None
    try:
        if args.start:
            start, end = pd.to_datetime(args.start, utc=True), pd.to_datetime(args.end, utc=True)
            if pd.isna(start) or pd.isna(end) or start >= end:
                raise ValueError
        if args.expiry:
            stamp = pd.Timestamp(args.expiry)
            if pd.isna(stamp) or stamp.strftime("%Y-%m-%d") != args.expiry:
                raise ValueError
    except ValueError:
        parser.error("Require valid dates, start < end, and YYYY-MM-DD expiry")
    strike = None
    if args.strike_min is not None or args.strike_max is not None:
        if not args.chain or args.strike_min is None or args.strike_max is None:
            parser.error("Both strike bounds require --chain")
        if not 0 < args.strike_min <= args.strike_max < float("inf"):
            parser.error("Strike bounds must be finite positive and ordered")
        strike = (args.strike_min, args.strike_max)
    if args.contract:
        args.contract = args.contract.strip().upper()
        match = re.fullmatch(r"([A-Z0-9.]+)(\d{6})([CP])(\d{8})", args.contract)
        if not match or match[1] != args.ticker:
            parser.error("--contract must be an OSI ticker for --ticker")
    output = args.output or Path(f"data/raw/lse/options/{args.ticker}_sample.json")
    if output.exists():
        parser.error("Output exists; choose another --output")
    if not os.environ.get("LSE_API_KEY", "").strip():
        parser.error("Set LSE_API_KEY (or use uv run --env-file .env)")
    payload = dict(
        provider="LSE",
        underlying=args.ticker,
        limit=args.limit,
        chain=[],
        flow=[],
        candles=[],
        limit_reached={},
        filters=dict(
            expiry=args.expiry,
            type=args.type,
            strike=strike,
            start=start.isoformat() if start is not None else None,
            end=end.isoformat() if end is not None else None,
        ),
    )
    with LSE(timeout=30) as client:
        if args.chain:
            payload["chain"] = client.options(
                args.ticker, expiry=args.expiry, type=args.type, strike=strike, limit=args.limit
            )
        if args.flow:
            payload["flow"] = client.options_flow(
                args.ticker,
                expiry=args.expiry,
                type=args.type,
                order="desc",
                limit=args.limit,
                start=start.isoformat() if start is not None else None,
                end=end.isoformat() if end is not None else None,
            )
        if args.contract:
            payload["candles"] = client.option_candles(
                args.contract,
                order="desc",
                limit=args.limit,
                start=start.isoformat() if start is not None else None,
                end=end.isoformat() if end is not None else None,
            )
    for kind, time_column in (("chain", None), ("flow", "ts"), ("candles", "minute")):
        rows = payload[kind]
        if not isinstance(rows, list) or len(rows) > args.limit:
            raise ValueError("Provider returned an invalid or oversized row response")
        payload["limit_reached"][kind] = len(rows) == args.limit
        if time_column and start is not None:
            payload[kind] = [
                row for row in rows if start <= pd.to_datetime(row[time_column], utc=True) < end
            ]
    payload["snapshot_at"] = datetime.now(UTC).isoformat()
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > args.max_bytes:
        raise ValueError("Output exceeds --max-bytes; lower --limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        file.write(encoded)
    print(
        f"Saved {len(encoded):,} bytes: {output}; "
        f"rows={ {kind: len(payload[kind]) for kind in ('chain', 'flow', 'candles')} }"
    )


if __name__ == "__main__":
    main()
