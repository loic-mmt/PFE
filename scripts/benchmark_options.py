"""Offline options benchmark; selected columns, sparse snapshots and rolling prints."""

import argparse
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from pfe import compute_option_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.repeat < 1:
        parser.error("rows and repeat must be positive")
    bars = pd.date_range("2025-01-14 12:00", periods=args.rows, freq="min", tz="UTC")
    snapshots = pd.date_range(bars[0], periods=12, freq="h")
    rows = []
    for snapshot in snapshots:
        for side in ("call", "put"):
            for strike in (90, 95, 100, 105, 110):
                rows.append(
                    dict(
                        underlying="QQQ",
                        ticker=f"CONTRACT_{side}_{strike}",
                        contract_type=side,
                        strike=strike,
                        expiry="2025-01-31",
                        snapshot_at=snapshot,
                        available_at=snapshot + pd.Timedelta("1s"),
                        quote_at=snapshot,
                        underlying_price=100.0,
                        volume=100.0,
                        premium=10000.0,
                        open_interest=1000.0,
                        iv=0.2 + abs(strike - 100) / 1000,
                        delta=0.5 if side == "call" else -0.5,
                        gamma=0.01,
                        theta=-0.05,
                        vega=0.1,
                        rho=0.01,
                    )
                )
    chain = pd.DataFrame(rows)
    count = 10000
    trade_times = bars[0] + pd.to_timedelta(
        np.linspace(0, max(args.rows - 1, 1), count), unit="min"
    )
    flow = pd.DataFrame(
        dict(
            underlying=["QQQ"] * count,
            ticker=["CALL"] * count,
            contract_type=["call"] * count,
            strike=np.full(count, 100.0),
            expiry=["2025-04-11"] * count,
            trade_id=np.arange(count),
            trade_at=trade_times,
            available_at=trade_times + pd.Timedelta("1s"),
            volume=np.full(count, 10.0),
            premium=np.full(count, 1000.0),
            iv=np.full(count, 0.2),
            delta=np.full(count, 0.5),
        )
    )
    report = dict(
        platform=platform.platform(),
        rows=args.rows,
        chain_rows=len(chain),
        print_rows=count,
        repeat=args.repeat,
        results={},
    )
    cases = {
        "iv_mean": dict(names=("options.volatility.iv_mean",)),
        "volatility": dict(groups="volatility"),
        "flow_premium": dict(names=("options.flow.total_premium",)),
        "all": dict(),
    }
    for label, options in cases.items():
        durations = []
        for _ in range(args.repeat):
            start = perf_counter()
            out = compute_option_features(bars, chain=chain, flow=flow, **options)
            durations.append(perf_counter() - start)
        report["results"][label] = dict(
            median_seconds=statistics.median(durations),
            columns=len(out.columns),
            output_bytes=int(out.memory_usage(index=False).sum()),
        )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
