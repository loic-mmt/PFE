"""Walk-forward Ridge features on synthetic 1-minute bars; no API key needed."""

import numpy as np
import pandas as pd

from pfe import compute_linear_feature, compute_linear_features, normalize_bars


def main():
    rng = np.random.default_rng(17)
    n = 2000
    returns = rng.normal(0, 0.0005, n) + 0.0002 * np.sin(np.arange(n) / 19)
    close = 450 * np.exp(np.cumsum(returns))
    opening = np.r_[450.0, close[:-1]]
    bars = normalize_bars(
        pd.DataFrame(
            dict(
                open=opening,
                high=np.maximum(opening, close) + 0.05,
                low=np.minimum(opening, close) - 0.05,
                close=close,
                volume=rng.integers(1000, 10000, n),
            ),
            index=pd.date_range("2026-01-05 14:30", periods=n, freq="min", tz="UTC"),
        )
    )
    options = dict(
        predictors=("returns.log", "candles.range_relative", "volume.relative_w20"),
        train_window=250,
        refit_every=20,
        model="ridge",
        alpha=1.0,
    )
    prediction = compute_linear_feature(bars, "linear.prediction_h5", **options)
    features = compute_linear_features(
        bars, horizons=(5, 15), groups=("prediction", "fit", "errors"), **options
    )
    print(prediction.dropna().tail())
    print(features.dropna().tail())
    print("Index labels opens; outputs become available at index + 1 minute.")


if __name__ == "__main__":
    main()
