"""Compiled kernels for recurrences and native weighted moving averages."""

import numpy as np
import pandas as pd
import talib
from numba import njit


@njit(cache=True, nogil=True, fastmath=False)
def adaptive(values, rates, window):
    result = np.full(len(values), np.nan)
    if len(values) < window:
        return result
    state = np.mean(values[:window])
    result[window - 1] = state
    for i in range(window, len(values)):
        if np.isfinite(rates[i]):
            state += rates[i] * (values[i] - state)
            result[i] = state
    return result


def adaptive_mean(series, alpha, window):
    return pd.Series(adaptive(series.to_numpy(), alpha.to_numpy(), window), index=series.index)


def wma(series, window):
    if window == 1:
        return series.copy()
    return pd.Series(talib.WMA(series.to_numpy(dtype=float), timeperiod=window), index=series.index)


@njit(cache=True, nogil=True, fastmath=False)
def regression_component(values, key):
    """One regression descriptor; omit residual work when only slope is needed."""
    n = len(values)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=np.float64)
    mean_x, mean_y = x.mean(), values.mean()
    slope = np.dot(x - mean_x, values - mean_y) / np.sum((x - mean_x) ** 2)
    if key == "slope":
        return slope
    intercept = mean_y - slope * mean_x
    if key == "intercept":
        return intercept
    residuals = values - (intercept + slope * x)
    if key == "residual_std":
        return np.sqrt(np.mean(residuals**2))
    total = np.sum((values - mean_y) ** 2)
    return 1 - np.sum(residuals**2) / total if total != 0 else np.nan
