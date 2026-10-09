"""Reproducible synthetic speed/RSS benchmark; each case runs in a fresh process.

Examples:
    python scripts/benchmark_features.py --rows 50000 --case group-rsi
    python scripts/benchmark_features.py --rows 50000 --case selected-rsi
    python scripts/benchmark_features.py --rows 100000000 --case stream-rsi --batch-size 100000
RSS includes Python, dependencies, input/output and JIT. No market-data download.
"""

import argparse
import json
import platform
import resource
import time

import numpy as np
import pandas as pd

from pfe import compute_feature, compute_features, iter_features


def bars(start, count):
    position = np.arange(start, start + count, dtype=float)
    close = 100 + np.sin(position / 4) + 0.1 * np.sin(position / 117)
    opening = 100 + np.sin((position - 1) / 4) + 0.1 * np.sin((position - 1) / 117)
    return pd.DataFrame(
        dict(
            open=opening,
            high=np.maximum(opening, close) + 0.2,
            low=np.minimum(opening, close) - 0.2,
            close=close,
            volume=100 + position % 17,
        ),
        index=pd.date_range(
            pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=start),
            periods=count,
            freq="min",
            name="timestamp",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "group-rsi",
            "selected-rsi",
            "group-mean",
            "selected-mean",
            "selected-rv",
            "stream-rsi",
        ),
    )
    args = parser.parse_args()
    if args.rows < 1 or args.batch_size < 1:
        parser.error("Row counts must be positive")
    if args.case == "stream-rsi":
        # Compile/load JIT before timed processing; report its setup cost separately.
        setup = time.perf_counter()
        list(iter_features([bars(0, 100)], "technical.rsi_w14"))
        setup_seconds = time.perf_counter() - setup
        batches = (
            bars(i, min(args.batch_size, args.rows - i))
            for i in range(0, args.rows, args.batch_size)
        )
        started = time.perf_counter()
        count, checksum = 0, 0.0
        for frame in iter_features(batches, "technical.rsi_w14"):
            count += len(frame)
            checksum += np.nansum(frame.to_numpy())
        seconds = time.perf_counter() - started
        columns = 1
    else:
        data = bars(0, args.rows)
        operation = {
            "group-rsi": lambda b: compute_features(b, windows=(14,), groups=("technical",)),
            "selected-rsi": lambda b: compute_feature(b, "technical.rsi", window=14),
            "group-mean": lambda b: compute_features(b, windows=(14,), groups=("statistics",)),
            "selected-mean": lambda b: compute_feature(b, "statistics.close_mean", window=14),
            "selected-rv": lambda b: compute_feature(b, "path.realized_variance", frequency="7min"),
        }[args.case]
        setup = time.perf_counter()
        operation(data.iloc[:100])
        setup_seconds = time.perf_counter() - setup
        started = time.perf_counter()
        frame = operation(data)
        seconds = time.perf_counter() - started
        count = len(frame)
        columns = len(frame.columns) if isinstance(frame, pd.DataFrame) else 1
        checksum = np.nansum(frame.to_numpy())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024**2 if platform.system() == "Darwin" else 1024)
    print(
        json.dumps(
            dict(
                case=args.case,
                rows=args.rows,
                output_rows=count,
                columns=columns,
                seconds=round(seconds, 6),
                setup_seconds=round(setup_seconds, 6),
                rows_per_second=round(args.rows / seconds),
                peak_rss_mib=round(rss_mb, 1),
                checksum=float(checksum),
                machine=platform.machine(),
                python=platform.python_version(),
            )
        )
    )


if __name__ == "__main__":
    main()
