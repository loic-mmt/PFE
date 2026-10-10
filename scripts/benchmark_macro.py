"""Offline macro alignment benchmark; synthetic data, no files unless --output is set."""

import argparse
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from pfe import compute_macro_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.repeat < 1:
        parser.error("rows and repeat must be positive")
    bars = pd.date_range("2025-01-01", periods=args.rows, freq="min", tz="UTC")
    periods = pd.date_range("2024-01-01", periods=1000, freq="D", tz="UTC")
    observations = pd.DataFrame(
        dict(
            series=["rate"] * len(periods),
            observation_at=periods,
            available_at=periods + pd.Timedelta("12h"),
            value=3 + np.sin(np.arange(len(periods)) / 20),
        )
    )
    events = pd.DataFrame(
        dict(
            event_id=np.arange(len(periods)),
            event=["CPI"] * len(periods),
            scheduled_at=periods + pd.Timedelta("12h"),
            known_at=periods - pd.Timedelta("7D"),
            actual=np.sin(np.arange(len(periods)) / 7),
            forecast=np.zeros(len(periods)),
            actual_available_at=periods + pd.Timedelta("12h5s"),
            forecast_available_at=periods - pd.Timedelta("1D"),
        )
    )
    report = dict(
        platform=platform.platform(),
        rows=args.rows,
        macro_rows=len(periods),
        repeat=args.repeat,
        results={},
    )
    cases = {
        "level": dict(names=("macro.rate.level",)),
        "series": dict(groups="series"),
        "calendar": dict(groups="calendar"),
        "surprise": dict(groups="surprise"),
    }
    for label, options in cases.items():
        durations = []
        for _ in range(args.repeat):
            start = perf_counter()
            out = compute_macro_features(bars, observations=observations, events=events, **options)
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
