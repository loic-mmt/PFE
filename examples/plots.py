"""Export seven plots from synthetic data; no LSE key or network needed.

Run: uv run --extra plots python examples/plots.py --frequency 7min
For headless execution, set MPLBACKEND=Agg before launching Python.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pfe import (
    build_features,
    normalize_bars,
    plot_bars,
    plot_feature_correlation,
    plot_feature_distribution,
    plot_features,
    plot_intrabar,
    plot_price_time_profile,
    plot_profile_bars,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", default="7min")
    parser.add_argument("--output-dir", type=Path, default=Path("data/plots"))
    args = parser.parse_args()
    n = 210
    close = 100 + np.sin(np.arange(n) / 7) + np.arange(n) * 0.002
    opening = np.r_[100.0, close[:-1]]
    bars = normalize_bars(
        pd.DataFrame(
            {
                "open": opening,
                "high": np.maximum(opening, close) + 0.05,
                "low": np.minimum(opening, close) - 0.05,
                "close": close,
                "volume": 100 + np.arange(n) % 20 * 10,
            },
            index=pd.date_range("2026-01-05 14:30", periods=n, freq="min", tz="UTC"),
        )
    )
    features = build_features(
        bars,
        args.frequency,
        origin=bars.index[0],
        windows=(3, 5),
        groups=("returns", "volatility", "technical"),
    )
    columns = ["path.realized_variance", "path.directional_efficiency", "technical.rsi_w5"]
    figures = {
        "candles": plot_bars(
            bars,
            frequency=args.frequency,
            origin=bars.index[0],
            overlays=features,
            columns="technical.sma_w5",
            title="QQQ demo (synthetic)",
        ),
        "features": plot_features(features, columns),
        "intrabar": plot_intrabar(bars, start=bars.index[0], frequency=args.frequency),
        "profile": plot_price_time_profile(
            bars, start=bars.index[0], frequency=args.frequency, bins=12, segments=5
        ),
        "profile_bars": plot_profile_bars(
            bars.loc[bars.index < bars.index[0] + pd.Timedelta(args.frequency) * 4],
            frequency=args.frequency,
            origin=bars.index[0],
            bins=12,
            segments=5,
        ),
        "distributions": plot_feature_distribution(features, columns),
        "correlation": plot_feature_correlation(features, columns),
    }
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        path = args.output_dir / f"{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(path)


if __name__ == "__main__":
    main()
