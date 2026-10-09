"""Composable causal transforms of any numeric feature frame."""

import numpy as np
import pandas as pd

from pfe.features._utils import divide, validate_windows
from pfe.features.rolling import rolling_kernel


def transform_features(
    features: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 2, 5),
    windows: tuple[int, ...] = (20,),
    operations: tuple[str, ...] = ("lag", "diff", "zscore", "rank"),
    interactions: tuple[tuple[str, str], ...] = (),
) -> pd.DataFrame:
    """Return transformed columns; inputs are not modified.

    Positive lags only. Normalization is trailing. Interactions are explicitly
    chosen column pairs to avoid an uncontrolled quadratic feature expansion.
    """
    windows = validate_windows(windows)
    if any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("lags must be positive integers")
    allowed = {"lag", "diff", "mean", "std", "zscore", "rank", "slope", "tanh"}
    if set(operations) - allowed:
        raise ValueError(f"Unknown operations: {set(operations) - allowed}")
    if not features.columns.is_unique or not features.index.is_monotonic_increasing:
        raise ValueError("Feature columns must be unique and index sorted")
    if len(set(lags)) != len(lags) or len(set(operations)) != len(operations):
        raise ValueError("lags and operations must be unique")
    out = {}
    for name in features.select_dtypes(include="number"):
        s = features[name]
        if "tanh" in operations:
            out[f"{name}__tanh"] = np.tanh(s)
        for lag in lags:
            if "lag" in operations:
                out[f"{name}__lag{lag}"] = s.shift(lag)
            if "diff" in operations:
                out[f"{name}__diff{lag}"] = s.diff(lag)
        for w in windows:
            roll = s.rolling(w)
            if "mean" in operations:
                out[f"{name}__mean_w{w}"] = roll.mean()
            if "std" in operations:
                out[f"{name}__std_w{w}"] = roll.std()
            if "zscore" in operations:
                out[f"{name}__zscore_w{w}"] = divide(s - roll.mean(), roll.std())
            if "rank" in operations:
                out[f"{name}__rank_w{w}"] = roll.rank(pct=True)
            if "slope" in operations:
                weights = np.arange(w, dtype=float) - (w - 1) / 2
                weights /= np.sum(weights**2)
                out[f"{name}__slope_w{w}"] = rolling_kernel(s, w, "dot", weights)
    for a, b in interactions:
        if a not in features or b not in features:
            raise ValueError(f"Unknown interaction columns: {(a, b)}")
        out[f"{a}__times__{b}"] = features[a] * features[b]
        out[f"{a}__ratio__{b}"] = divide(features[a], features[b])
    return pd.DataFrame(out, index=features.index).replace([np.inf, -np.inf], np.nan)
