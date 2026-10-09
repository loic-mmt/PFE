"""Download one-minute history for a ticker using London Strategic Edge."""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from lse import LSE

from pfe import normalize_bars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="QQQ", help="LSE ticker (default: QQQ)")
    parser.add_argument("--start", required=True, help="Inclusive start, ISO 8601 in UTC")
    parser.add_argument("--end", required=True, help="Exclusive end, ISO 8601 in UTC")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9._^-]{0,63}", ticker):
        parser.error("--ticker must be a symbol containing letters, digits, '.', '-', '_' or '^'")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    if pd.isna(start) or pd.isna(end) or start >= end:
        parser.error("--start must precede --end")
    key = os.environ.get("LSE_API_KEY", "").strip()
    if not key:
        parser.error("Set LSE_API_KEY (or use uv run --env-file .env)")

    window = f"{start.strftime('%Y%m%dT%H%M%SZ')}_{end.strftime('%Y%m%dT%H%M%SZ')}"
    raw_dir = args.data_dir / "raw" / "lse" / ticker / "1m" / window
    output = args.data_dir / "processed" / f"{ticker}_1m_{window}.parquet"
    if raw_dir.exists() or output.exists():
        parser.error("Window already exists; choose a new date range or --data-dir")

    with LSE(api_key=key) as client:
        # Bulk exports accept dates only. Fetch whole UTC days covering the
        # requested window, then apply the precise timestamp bounds locally.
        frame = client.history(
            ticker,
            timeframe="1m",
            start=start.date().isoformat(),
            end=end.ceil("D").date().isoformat(),
            dest=str(raw_dir),
        )
    # Vault Parquet uses `ts`; the REST candles API uses `timestamp`.
    timestamp_column = "timestamp" if "timestamp" in frame.columns else "ts"
    bars = normalize_bars(frame, timestamp_column=timestamp_column, source_timezone="UTC")
    bars = bars.loc[(bars.index >= start) & (bars.index < end)]
    if bars.empty:
        raise ValueError(f"No {ticker} bars in the requested window")
    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(output)
    print(f"Saved {len(bars):,} {ticker} 1m bars: {output}")


if __name__ == "__main__":
    main()
