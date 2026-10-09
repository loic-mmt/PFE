"""Both feature families on synthetic one-minute data, with a seven-minute target."""

import numpy as np
import pandas as pd

from pfe import build_features, compute_features, feature_catalog, normalize_bars

n = 210
close = 100 + np.sin(np.arange(n) / 7) + np.arange(n) * 0.002
opening = np.r_[100.0, close[:-1]]
bars = normalize_bars(
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 14:30", periods=n, freq="min", tz="UTC"),
            "open": opening,
            "high": np.maximum(opening, close) + 0.1,
            "low": np.minimum(opening, close) - 0.1,
            "close": close,
            "volume": 100 + np.arange(n) % 31,
        }
    )
)

source_features = compute_features(bars, windows=(5, 20))
enriched = build_features(
    bars,
    frequency="7min",
    origin=bars.index[0],
    windows=(3, 5),
    groups=("returns", "technical", "patterns"),
    segments=5,
    bins=8,
)
print(f"Source: {source_features.shape}; enriched 7min: {enriched.shape}")
print(
    enriched[
        [
            "available_at",
            "close",
            "path.realized_volatility",
            "path.directional_efficiency",
            "technical.rsi_w3",
        ]
    ].tail(3)
)
print(feature_catalog(enriched).groupby("family").size())
