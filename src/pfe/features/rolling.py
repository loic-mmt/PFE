"""Whole-array C++ windows, with the previous formulas as a Python reference."""

import numpy as np
import pandas as pd

from pfe.features._utils import centered_moment, runs
from pfe.features.backend import native, using_cpp

COMPLEXITY_CPP = {
    "sample_entropy",
    "permutation_entropy",
    "sign_entropy",
    "zero_crossings",
    "petrosian_dimension",
    "tail_fraction",
    "expected_shortfall_05",
    "hurst",
    "trend_slope",
    *(f"wavelet_detail_{name}_l{level}" for name in ("energy", "first") for level in (1, 2, 3)),
}


def rolling_kernel(series, window, key, weights=None):
    """Evaluate only one callback, in one C++ call over all trailing windows."""
    if using_cpp():
        values = native.rolling(
            series.to_numpy(dtype=float, copy=False),
            window,
            [key],
            np.asarray(weights, dtype=float) if weights is not None else np.empty(0),
        )[0]
        return pd.Series(values, index=series.index)

    def function(x):
        if key == "skew":
            return centered_moment(x, 3)
        if key == "kurtosis":
            return centered_moment(x, 4)
        if key == "mad":
            return np.median(np.abs(x - np.median(x)))
        if key.startswith("regression_"):
            from pfe.features._kernels import regression_component

            return regression_component(x, key.removeprefix("regression_"))
        if key in {"high_age", "low_age"}:
            return len(x) - 1 - np.flatnonzero(x == (x.max() if key == "high_age" else x.min()))[-1]
        if key == "max_drawdown":
            return np.min(x / np.maximum.accumulate(x) - 1)
        if key == "turn_count":
            return np.count_nonzero(np.diff(x[x != 0]))
        if key in {"longest_up_run", "longest_down_run"}:
            direction = 1 if key == "longest_up_run" else -1
            return max((n for s, n in zip(*runs(x), strict=True) if s == direction), default=0)
        if key == "dot":
            return np.dot(x, weights)
        raise ValueError(f"Unknown reference rolling kernel: {key}")

    return series.rolling(window).apply(function, raw=True)


def slope_convergence(high, low, window, weights, high_slope, low_slope):
    """Preserve binary decisions when dot-product rounding could flip a tie."""
    result = (high_slope < low_slope).astype(float).where(high_slope.notna())
    if using_cpp():
        scale = np.maximum(high.abs(), low.abs()).rolling(window).max()
        tolerance = 8 * np.finfo(float).eps * window * np.abs(weights).sum() * scale
        ambiguous = (high_slope - low_slope).abs() <= tolerance
        highs, lows = high.to_numpy(), low.to_numpy()
        for end in np.flatnonzero(ambiguous.to_numpy()):
            start = end - window + 1
            result.iloc[end] = float(
                np.dot(highs[start : end + 1], weights) < np.dot(lows[start : end + 1], weights)
            )
    return result
