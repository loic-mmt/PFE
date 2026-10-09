"""Hand-computed values, full-catalog causality, warm-up and gap behavior."""

import numpy as np
import pandas as pd
import pytest
import talib
from pandas.testing import assert_frame_equal, assert_series_equal

from pfe import (
    FEATURE_GROUPS,
    compute_features,
    feature_catalog,
    normalize_bars,
    transform_features,
)


@pytest.fixture
def fine_bars():
    n = 180
    c = 100 + np.sin(np.arange(n) / 4) + 0.005 * np.arange(n)
    o = np.r_[100.0, c[:-1]]
    return normalize_bars(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30", periods=n, freq="min", tz="UTC"),
                "open": o,
                "high": np.maximum(o, c) + 0.2,
                "low": np.minimum(o, c) - 0.2,
                "close": c,
                "volume": 100 + np.arange(n) % 17,
            }
        )
    )


@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_each_family_is_causal_and_does_not_mutate_input(fine_bars, group):
    before = fine_bars.copy(deep=True)
    prefix = compute_features(fine_bars.iloc[:90], windows=(5, 20), groups=(group,))
    altered = fine_bars.copy()
    altered.iloc[90:] *= 2
    full = compute_features(altered, windows=(5, 20), groups=(group,))
    assert_frame_equal(prefix, full.loc[prefix.index], atol=1e-10, rtol=1e-10)
    assert_frame_equal(fine_bars, before)
    assert prefix.columns.is_unique
    assert not np.isinf(full.to_numpy()).any()


def test_returns_and_statistics_hand_calculation(raw_bars):
    bars = normalize_bars(raw_bars)
    features = compute_features(bars, windows=(2,), groups=("returns", "statistics", "candles"))
    assert features["returns.log"].iloc[1] == pytest.approx(np.log(102 / 101))
    assert np.isnan(features["returns.log"].iloc[0])
    assert features["statistics.close_mean_w2"].iloc[1] == 101.5
    assert features["statistics.close_std_w2"].iloc[1] == pytest.approx(np.sqrt(0.5))
    assert features["candles.close_position"].iloc[0] == pytest.approx(2 / 3)


def test_technical_reference_and_all_candlestick_patterns(fine_bars):
    f = compute_features(fine_bars, windows=(5,), groups=("technical", "patterns"))
    expected = pd.Series(
        talib.RSI(fine_bars.close.to_numpy(), timeperiod=5),
        index=fine_bars.index,
        name="technical.rsi_w5",
    )
    assert_series_equal(f["technical.rsi_w5"], expected)
    assert f["technical.sma_w5"].iloc[4] == pytest.approx(fine_bars.close.iloc[:5].mean())
    pattern_names = talib.get_function_groups()["Pattern Recognition"]
    assert len(pattern_names) == 61
    assert all(f"patterns.{name.lower()}" in f for name in pattern_names)
    assert f["patterns.cdlhammer"].iloc[:10].isna().all()


def test_gap_resets_returns_and_rolling_state(fine_bars):
    bars = fine_bars.iloc[:12].copy()
    bars.index = bars.index[:6].append(bars.index[6:] + pd.Timedelta("1D"))
    reset = compute_features(bars, windows=(3,), groups=("returns", "trend"))
    allow = compute_features(bars, windows=(3,), groups=("returns", "trend"), gap_policy="allow")
    assert np.isnan(reset["returns.log"].iloc[6])
    assert reset["trend.sma_w3"].iloc[6:8].isna().all()
    assert np.isfinite(allow["returns.log"].iloc[6])


def test_flat_prices_zero_volume_and_short_windows_are_safe(fine_bars):
    bars = fine_bars.iloc[:25].copy()
    bars[["open", "high", "low", "close"]] = 100.0
    bars.volume = 0.0
    f = compute_features(bars, windows=(2, 5))
    assert not np.isinf(f.to_numpy()).any()
    assert f["candles.close_position"].isna().all()
    assert f["volume.vwap_proxy_w5"].isna().all()
    assert f["complexity.price_sample_entropy_w5"].iloc[-1] == 0


def test_previous_session_levels_and_opening_range_are_causal(fine_bars):
    first = fine_bars.iloc[:20]
    second = fine_bars.iloc[:20].copy()
    second.index += pd.Timedelta("1D")
    second *= 1.1
    bars = pd.concat([first, second])
    f = compute_features(bars, windows=(5,), groups=("calendar",), opening_range="5min")
    assert f["session.classic_p"].iloc[:20].isna().all()
    expected = (first.high.max() + first.low.min() + first.close.iloc[-1]) / 3
    assert f["session.classic_p"].iloc[20] == pytest.approx(expected)
    assert f["session.opening_range_high"].iloc[:4].isna().all()
    assert f["session.opening_range_high"].iloc[4] == first.high.iloc[:5].max()
    assert f["session.volume_same_slot_relative"].iloc[20] == pytest.approx(1.1)


def test_transforms_are_causal_and_reject_future_lags(fine_bars):
    f = compute_features(fine_bars, windows=(5,), groups=("returns",))
    prefix = transform_features(
        f.iloc[:90], windows=(5,), lags=(1, 3), interactions=(("returns.simple", "returns.log"),)
    )
    full = transform_features(
        f, windows=(5,), lags=(1, 3), interactions=(("returns.simple", "returns.log"),)
    )
    assert_frame_equal(prefix, full.iloc[:90])
    assert full["returns.simple__lag1"].iloc[2] == f["returns.simple"].iloc[1]
    with pytest.raises(ValueError, match="positive"):
        transform_features(f, lags=(-1,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"windows": (1,)},
        {"windows": (5, 5)},
        {"groups": ("unknown",)},
        {"groups": "returns"},
        {"gap_policy": "fill"},
        {"structure_threshold": 0},
        {"source_frequency": "0min"},
    ],
)
def test_bad_configuration_fails_explicitly(fine_bars, kwargs):
    with pytest.raises(ValueError):
        compute_features(fine_bars, **kwargs)


def test_catalog_can_describe_exact_output(fine_bars):
    f = compute_features(fine_bars, groups=("returns",), windows=(5,))
    assert list(feature_catalog(f).name) == list(f.columns)
    assert "CDLHAMMER" in set(feature_catalog().name)


def test_calendar_schema_before_session_open_is_stable(fine_bars):
    bars = fine_bars.copy()
    bars.index -= pd.Timedelta("1h")
    prefix = compute_features(bars.iloc[:30], windows=(5,), groups=("calendar",))
    full = compute_features(bars, windows=(5,), groups=("calendar",))
    assert_frame_equal(prefix, full.iloc[:30])


def test_zigzag_confirmation_is_not_backdated(fine_bars):
    bars = fine_bars.iloc[:5].copy()
    close = np.array([100, 100.5, 102, 101.5, 99.0])
    bars.close = close
    bars.open = close
    bars.high = close + 0.1
    bars.low = close - 0.1
    f = compute_features(bars, windows=(2,), groups=("structure",), structure_threshold=0.01)
    assert f["structure.zigzag_confirmed_pivot"].iloc[:2].isna().all()
    assert f["structure.zigzag_confirmed_pivot"].iloc[2] == 100
    assert f["structure.zigzag_confirmed_pivot"].iloc[4] == 102
    assert f["structure.zigzag_event"].iloc[4] == -1


def test_adaptive_average_handles_flat_halves_separated_by_price_gap(fine_bars):
    bars = fine_bars.iloc[:20].copy()
    price = np.repeat([100.0, 110.0, 90.0, 100.0], 5)
    for name in ("open", "high", "low", "close"):
        bars[name] = price
    features = compute_features(bars, windows=(10,), groups=("technical",))
    assert np.isfinite(features["technical.frama_w10"].iloc[-1])
