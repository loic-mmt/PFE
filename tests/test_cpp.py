"""Native kernels must match reference formulas and validate buffer boundaries."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pfe import (
    aggregate_selected_features,
    compute_feature,
    compute_features,
    enrich_bars,
    feature_backend,
    price_time_profile,
)
from pfe.features.backend import native, using_cpp
from pfe.features.rolling import COMPLEXITY_CPP, rolling_kernel
from pfe.features.selected import _complexity


@pytest.fixture
def native_bars():
    rng = np.random.default_rng(123)
    n = 180
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    opening = np.r_[100.0, close[:-1]]
    return pd.DataFrame(
        dict(
            open=opening,
            high=np.maximum(opening, close) + 0.25,
            low=np.minimum(opening, close) - 0.25,
            close=close,
            volume=rng.integers(0, 1000, n).astype(float),
        ),
        index=pd.date_range("2026-01-01 14:30", periods=n, freq="min", tz="UTC", name="timestamp"),
    )


@pytest.mark.parametrize("mode", ["regular", "holes", "flat", "zero_volume", "one_bar"])
@pytest.mark.parametrize("frequency", ["7min", "30min"])
def test_all_intrabar_features_match_python(native_bars, mode, frequency):
    bars = native_bars.copy()
    if mode == "holes":
        bars = bars.drop(bars.index[[8, 31, 59, 60, 88]])
    elif mode == "flat":
        bars.loc[:, ["open", "high", "low", "close"]] = 100.0
    elif mode == "zero_volume":
        bars.volume = 0.0
    elif mode == "one_bar":
        bars = bars.iloc[:1]
    with feature_backend("python"):
        expected = enrich_bars(bars, frequency, segments=5, bins=10, partial="keep")
    with feature_backend("cpp"):
        result = enrich_bars(bars, frequency, segments=5, bins=10, partial="keep")
    assert_frame_equal(expected, result, check_freq=False, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize(
    "family", ["statistics", "trend", "levels", "sequences", "technical", "structure", "complexity"]
)
def test_group_callbacks_match_python(native_bars, family):
    with feature_backend("python"):
        expected = compute_features(native_bars, groups=(family,), windows=(5, 20))
    with feature_backend("cpp"):
        result = compute_features(native_bars, groups=(family,), windows=(5, 20))
    assert_frame_equal(expected, result, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("bins,segments", [(1, 1), (3, 7), (15, 5)])
def test_profile_arrays_match_for_fine_time_grid(native_bars, bins, segments):
    bars = native_bars.iloc[:30].copy()
    bars.index = pd.date_range("2026-01-01", periods=30, freq="2s", tz="UTC")
    options = dict(start=bars.index[0], duration=pd.Timedelta("1min"), bins=bins, segments=segments)
    with feature_backend("python"):
        expected = price_time_profile(bars, **options)
    with feature_backend("cpp"):
        result = price_time_profile(bars, **options)
    for a, b in zip(expected, result, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "key",
    [
        "skew",
        "kurtosis",
        "mad",
        "regression_slope",
        "regression_intercept",
        "regression_r_squared",
        "regression_residual_std",
        "high_age",
        "low_age",
        "max_drawdown",
        "turn_count",
        "longest_up_run",
        "longest_down_run",
        "dot",
    ],
)
def test_rolling_kernel_windows_missing_and_read_only_strides(key):
    rng = np.random.default_rng(25)
    values = 100 + rng.normal(size=120)
    values[15] = np.nan
    values[72] = np.inf
    values.flags.writeable = False
    series = pd.Series(values[::2], copy=False)
    weights = np.arange(1, 6) / 15
    if key in {"turn_count", "longest_up_run", "longest_down_run"}:
        series = np.sign(series.diff())
    with feature_backend("python"):
        expected = rolling_kernel(series, 5, key, weights)
    with feature_backend("cpp"):
        result = rolling_kernel(series, 5, key, weights)
    assert_series_equal(expected, result, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("window", [2, 3, 4, 5, 8, 20])
def test_complexity_callbacks_edge_windows_and_constants(native_bars, window):
    # Avoid a constant log-price's roundoff in an undefined regression R².
    names = tuple(f"complexity.price_{key}_w{window}" for key in sorted(COMPLEXITY_CPP))
    with feature_backend("python"):
        expected = _complexity(native_bars, names)
    with feature_backend("cpp"):
        result = _complexity(native_bars, names)
    for name in names:
        np.testing.assert_allclose(expected[name], result[name], atol=1e-10, rtol=1e-9)


def test_targeted_cpp_never_traverses_python_groups(native_bars, monkeypatch):
    from pfe.features._plan import Planner
    from pfe.features.targeted import _path_plan

    _path_plan.cache_clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("Python window/bin callbacks executed")

    monkeypatch.setattr(Planner, "evaluate", forbidden)
    with feature_backend("cpp"):
        aggregate_selected_features(
            native_bars,
            (
                "profile.tpo_poc",
                "profile.tpo_entropy",
                "path.max_drawdown",
                "path.regression_slope",
            ),
            "7min",
        )


def test_backend_context_and_native_validation():
    assert native is not None
    initial = using_cpp()
    with feature_backend("python"):
        assert not using_cpp()
        with feature_backend("cpp"):
            assert using_cpp()
        assert not using_cpp()
    assert using_cpp() == initial
    with pytest.raises(ValueError):
        with feature_backend("invalid"):
            pass
    with pytest.raises(ValueError):
        native.rolling(np.ones((2, 3)), 5, ["mad"], np.empty(0))
    with pytest.raises(ValueError):
        native.rolling(np.ones(10), 5, ["dot"], np.ones(2))
    with pytest.raises(ValueError):
        native.rolling(np.ones(10), 5, ["unknown"], np.empty(0))
    with pytest.raises(ValueError):
        native.profile_arrays(np.ones(2), np.ones(1), np.ones(2), np.arange(2), 0, 10, 3, 5)
    with pytest.raises(ValueError):
        native.profile_arrays(np.ones(2), np.ones(2), np.ones(2), np.arange(2), 0, 10, 0, 5)


@pytest.mark.parametrize("value", [0.1, 100.1, 123.45, 0.0033, 500.00001])
@pytest.mark.parametrize("window", [5, 20, 60, 130])
def test_constant_window_rounding_is_preserved(value, window):
    series = pd.Series(np.full(window + 5, value))
    for key in ("skew", "kurtosis", "regression_r_squared"):
        with feature_backend("python"):
            expected = rolling_kernel(series, window, key)
        with feature_backend("cpp"):
            result = rolling_kernel(series, window, key)
        assert_series_equal(expected, result, atol=1e-10, rtol=1e-9)


def test_binary_slope_ties_preserve_numpy_decisions(native_bars):
    native_bars.loc[:, ["open", "close"]] = 100.1
    native_bars.high = 100.3
    native_bars.low = 99.9
    with feature_backend("python"):
        expected = compute_features(native_bars, groups=("structure",), windows=(5, 20, 60))
    with feature_backend("cpp"):
        result = compute_features(native_bars, groups=("structure",), windows=(5, 20, 60))
    assert_frame_equal(expected, result, atol=1e-10, rtol=1e-9)


def test_cpp_callback_does_not_use_pandas_apply(native_bars, monkeypatch):
    from pandas.core.window.rolling import Rolling

    def forbidden(*args, **kwargs):
        raise AssertionError("Per-window Python callback executed")

    monkeypatch.setattr(Rolling, "apply", forbidden)
    with feature_backend("cpp"):
        compute_feature(native_bars, "statistics.close_mad", window=20)
        compute_feature(native_bars, "trend.regression_slope", window=20)
        compute_feature(native_bars, "complexity.return_sample_entropy", window=20)
