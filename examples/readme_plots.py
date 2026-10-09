"""Regenerate the README gallery from deterministic synthetic OHLCV data.

Run: uv run --extra plots python examples/readme_plots.py
For headless execution, set MPLBACKEND=Agg before launching Python.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pfe import (
    compute_selected_features,
    normalize_bars,
    plot_bars,
    plot_features,
    plot_profile_bars,
)


def main():
    rng = np.random.default_rng(42)
    n = 420
    time = np.arange(n)
    noise = rng.normal(0, 0.12 + 0.10 * np.sin(time / 47) ** 2, n)
    close = 450 + np.cumsum(noise + 0.018 * np.sin(time / 28))
    opening = np.r_[450.0, close[:-1]]
    bars = normalize_bars(
        pd.DataFrame(
            {
                "open": opening,
                "high": np.maximum(opening, close) + rng.uniform(0.02, 0.12, n),
                "low": np.minimum(opening, close) - rng.uniform(0.02, 0.12, n),
                "close": close,
                "volume": rng.integers(10000, 45000, n) + (np.abs(noise) * 80000).astype(int),
            },
            index=pd.date_range("2026-01-05 14:30", periods=n, freq="min", tz="UTC"),
        )
    )
    columns = [
        "path.realized_variance",
        "path.directional_efficiency",
        "technical.rsi_w14",
    ]
    features = compute_selected_features(
        bars,
        (*columns, "technical.sma_w14"),
        frequency="7min",
        origin=bars.index[0],
    )
    figures = {
        "candles": plot_bars(
            bars,
            frequency="7min",
            origin=bars.index[0],
            overlays=features,
            columns="technical.sma_w14",
            title="PFE | 7-minute candles + SMA(14) | synthetic data",
            figsize=(12, 5),
        ),
        "profile_bars": plot_profile_bars(
            bars.iloc[:120],
            frequency="30min",
            origin=bars.index[0],
            bins=8,
            segments=6,
            cell_alpha=0.14,
            title="PFE | 30-minute TPO candles | synthetic data",
            figsize=(12, 5),
        ),
        "features": plot_features(
            features,
            columns,
            title="PFE | 7-minute features | synthetic data",
            figsize=(12, 6),
        ),
    }
    import matplotlib.pyplot as plt

    output = Path(__file__).resolve().parents[1] / "docs" / "assets"
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        path = output / f"{name}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        print(path)


if __name__ == "__main__":
    main()
