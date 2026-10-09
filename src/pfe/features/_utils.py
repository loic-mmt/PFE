"""Shared numerical conventions: undefined ratios are NaN, never infinity."""

import numpy as np
import pandas as pd


def divide(numerator, denominator):
    if isinstance(denominator, (pd.Series, pd.DataFrame)):
        return numerator / denominator.replace(0, np.nan)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(np.shape(numerator), denominator.shape), np.nan),
        where=denominator != 0,
    )


def ratio(a: float, b: float) -> float:
    return float(a / b) if b != 0 and np.isfinite(b) else np.nan


def entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if len(values) == 0:
        return np.array([], dtype=values.dtype), np.array([], dtype=int)
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return values[edges[:-1]], np.diff(edges)


def regression(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    if n < 2:
        return dict(slope=np.nan, intercept=np.nan, r_squared=np.nan, residual_std=np.nan)
    x = np.arange(n, dtype=float)
    slope = float(np.dot(x - x.mean(), values - values.mean()) / np.sum((x - x.mean()) ** 2))
    intercept = float(values.mean() - slope * x.mean())
    residuals = values - (intercept + slope * x)
    total = float(np.sum((values - values.mean()) ** 2))
    return dict(
        slope=slope,
        intercept=intercept,
        r_squared=1 - ratio(float(np.sum(residuals**2)), total),
        residual_std=float(np.sqrt(np.mean(residuals**2))),
    )


def centered_moment(values: np.ndarray, order: int) -> float:
    """Unbiased sample skew/excess kurtosis computed inside the window only.

    pandas' rolling skew/kurt preconditioning can use the full input's mean,
    making rounding of past outputs depend on later data. Local centering avoids
    that dependency. Constant windows have undefined standardized moments.
    """
    n = len(values)
    centered = values - values.mean()
    m2 = float(np.mean(centered**2))
    if m2 == 0:
        return np.nan
    if order == 3 and n >= 3:
        return float(np.sqrt(n * (n - 1)) / (n - 2) * np.mean(centered**3) / m2**1.5)
    if order == 4 and n >= 4:
        return float(
            (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * np.mean(centered**4) / m2**2 - 3 * (n - 1))
        )
    return np.nan


def event_age(event: pd.Series) -> pd.Series:
    positions = pd.Series(np.arange(len(event), dtype=float), index=event.index)
    last = positions.where(event.fillna(False)).ffill()
    return positions - last


def wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    from pfe.features.backend import using_cpp

    if using_cpp():
        from pfe.features.rolling import rolling_kernel

        return rolling_kernel(series, window, "dot", weights / weights.sum())
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def validate_windows(windows) -> tuple[int, ...]:
    windows = tuple(windows)
    if not windows or any(isinstance(w, bool) or not isinstance(w, int) or w < 2 for w in windows):
        raise ValueError("windows must contain integers >= 2")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must be unique")
    return windows
