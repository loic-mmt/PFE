"""Measure selected/all linear outputs and refit cadence on synthetic bars.

uv run python scripts/benchmark_linear.py --rows 100000 --repeat 3
Inputs, imports and warmup are outside timings; feature construction/validation
are included. Cadence changes model predictions, so cadence timings are not
comparisons of mathematically identical outputs.
"""

import argparse
import json
import platform
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from pfe import compute_linear_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/linear.json"))
    args = parser.parse_args()
    if args.rows < 1000 or args.repeat < 1:
        parser.error("Require rows>=1000 and repeat>=1")
    rng = np.random.default_rng(17)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0002, args.rows)))
    opening = np.r_[100.0, close[:-1]]
    bars = pd.DataFrame(
        dict(
            open=opening,
            high=np.maximum(opening, close) + 0.05,
            low=np.minimum(opening, close) - 0.05,
            close=close,
            volume=rng.integers(1000, 10000, args.rows).astype(float),
        ),
        index=pd.date_range("2026-01-01", periods=args.rows, freq="min", tz="UTC"),
    )
    options = dict(horizons=(5,), train_window=500, model="ridge", alpha=1.0)
    selected = ("linear.prediction_h5",)
    all_probe = compute_linear_features(bars.iloc[:1000], refit_every=20, **options)
    one_probe = compute_linear_features(bars.iloc[:1000], names=selected, refit_every=20, **options)
    assert_series_equal(all_probe.iloc[:, 0], one_probe.iloc[:, 0])
    report = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        system=platform.platform(),
        rows=args.rows,
        repeat=args.repeat,
        train_window=500,
        horizon=5,
        model="ridge",
        predictors=3,
        method="Median after warmup; input creation and I/O excluded; API included",
        results=[],
    )
    for label, names, cadence in (
        ("prediction_only", selected, 20),
        ("all_outputs", None, 20),
        ("prediction_refit_each_bar", selected, 1),
    ):
        compute_linear_features(bars.iloc[:1000], names=names, refit_every=cadence, **options)
        samples = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            result = compute_linear_features(bars, names=names, refit_every=cadence, **options)
            samples.append(time.perf_counter() - started)
        item = dict(
            case=label,
            refit_every=cadence,
            columns=len(result.columns),
            samples_seconds=samples,
            median_seconds=median(samples),
        )
        report["results"].append(item)
        print(f"{label}: {item['median_seconds']:.4f}s ({len(result.columns)} columns)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
