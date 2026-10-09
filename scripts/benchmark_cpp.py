"""Compare equivalent Python and C++ outputs in fresh, sequential processes.

uv run python scripts/benchmark_cpp.py --rows 100000 --repeat 3
Raw times, RSS and machine metadata are saved as JSON. Creation of input, module
imports, build and warmup are excluded from timing; validation and Python API
wrappers are included. No disk I/O or real-market download in measured sections.
"""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pfe import aggregate_selected_features, compute_feature, enrich_bars, feature_backend
from pfe.features.backend import native

CASES = (
    "tpo",
    "intrabar",
    "window_mad",
    "window_regression",
    "window_sample_entropy",
    "enriched_group",
)


def bars(count):
    rng = np.random.default_rng(1945)
    # Stationary bounded prices, varied OHLC ranges and volumes; fixed seed.
    position = np.arange(count)
    close = 100 + np.sin(position / 4) + rng.normal(0, 0.01, count)
    opening = np.r_[100.0, close[:-1]]
    return pd.DataFrame(
        dict(
            open=opening,
            high=np.maximum(opening, close) + 0.2,
            low=np.minimum(opening, close) - 0.2,
            close=close,
            volume=rng.integers(0, 1000, count).astype(float),
        ),
        index=pd.date_range("2026-01-01", periods=count, freq="min", tz="UTC", name="timestamp"),
    )


def calculate(data, case, window, frequency):
    if case == "tpo":
        return aggregate_selected_features(
            data,
            (
                "profile.tpo_poc",
                "profile.tpo_entropy",
                "profile.tpo_value_area_low",
                "profile.tpo_value_area_high",
                "profile.zone_revisits",
                "profile.segment4_overlap_previous",
            ),
            frequency,
            bins=10,
            segments=5,
        )
    if case == "intrabar":
        return aggregate_selected_features(
            data,
            (
                "path.max_drawdown",
                "path.max_drawup",
                "path.drawdown_underwater_seconds",
                "path.drawdown_recovery_seconds",
                "path.regression_slope",
                "path.return_autocorr",
                "path.volume_time_center",
                "path.segment4_realized_variance",
            ),
            frequency,
        )
    if case == "enriched_group":
        return enrich_bars(data, frequency)
    name = {
        "window_mad": "statistics.close_mad",
        "window_regression": "trend.regression_slope",
        "window_sample_entropy": "complexity.return_sample_entropy",
    }[case]
    return compute_feature(data, name, window=window)


def worker(args):
    data = bars(args.rows)
    with feature_backend(args.backend):
        setup = time.perf_counter()
        calculate(data.iloc[: max(120, args.window + 10)], args.case, args.window, args.frequency)
        setup_seconds = time.perf_counter() - setup
        samples = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            result = calculate(data, args.case, args.window, args.frequency)
            samples.append(time.perf_counter() - started)
        values = (
            result.select_dtypes(include="number").to_numpy()
            if isinstance(result, pd.DataFrame)
            else result.to_numpy()
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            dict(
                case=args.case,
                backend=args.backend,
                rows=args.rows,
                output_rows=len(result),
                numeric_columns=values.shape[1] if values.ndim == 2 else 1,
                samples_seconds=samples,
                median_seconds=median(samples),
                setup_seconds=setup_seconds,
                peak_rss_mib=rss / (1024**2 if platform.system() == "Darwin" else 1024),
                checksum=float(np.nansum(values)),
                nan_count=int(np.isnan(values).sum()),
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--frequency", default="30min")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/cpp.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("cpp", "python"), help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 120 or args.repeat < 1 or args.window < 2:
        parser.error("Require rows>=120, repeat>=1 and window>=2")
    if args.worker:
        worker(args)
        return
    if native is None:
        parser.error("Compile the extension first: uv sync --reinstall-package pfe")
    probe = bars(min(args.rows, 2000))
    report = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        system=platform.platform(),
        machine=platform.machine(),
        rows=args.rows,
        repeat=args.repeat,
        window=args.window,
        frequency=args.frequency,
        comparisons=[],
        method="Sequential fresh processes; median after warmup; API validation included; no I/O",
    )
    for case in args.cases:
        with feature_backend("python"):
            old = calculate(probe, case, args.window, args.frequency)
        with feature_backend("cpp"):
            new = calculate(probe, case, args.window, args.frequency)
        checker = assert_frame_equal if isinstance(old, pd.DataFrame) else assert_series_equal
        checker(old, new, atol=1e-10, rtol=1e-9)
        results = {}
        for backend in ("python", "cpp"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--backend",
                backend,
                "--case",
                case,
                "--rows",
                str(args.rows),
                "--repeat",
                str(args.repeat),
                "--window",
                str(args.window),
                "--frequency",
                args.frequency,
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            results[backend] = json.loads(result.stdout)
        assert results["python"]["nan_count"] == results["cpp"]["nan_count"]
        np.testing.assert_allclose(
            results["python"]["checksum"], results["cpp"]["checksum"], rtol=1e-9, atol=1e-8
        )
        speedup = results["python"]["median_seconds"] / results["cpp"]["median_seconds"]
        report["comparisons"].append(dict(case=case, speedup=speedup, **results))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{case}: Python {results['python']['median_seconds']:.4f}s; "
            f"C++ {results['cpp']['median_seconds']:.4f}s; x{speedup:.1f}",
            flush=True,
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
