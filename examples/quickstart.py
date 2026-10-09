"""Run offline with synthetic candles and basic price features."""

import pandas as pd

from pfe import compute_features, normalize_bars

bars = normalize_bars(
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 14:30", periods=4, freq="min", tz="UTC"),
            "open": [100.0, 101.0, 100.5, 102.0],
            "high": [102.0, 102.0, 103.0, 103.0],
            "low": [99.0, 100.0, 100.0, 101.0],
            "close": [101.0, 100.5, 102.0, 102.5],
            "volume": [1000, 1200, 900, 1100],
        }
    )
)
print(bars)
print(compute_features(bars, windows=(2,), groups=("returns", "candles")))
