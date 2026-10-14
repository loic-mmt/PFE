"""Fine-to-coarse aggregation, availability, profiles and path preservation."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe import (
    build_features,
    enrich_bars,
    multi_timeframe_features,
    normalize_bars,
    price_time_profile,
)


def make_bars(closes, *, opens=None, start="2026-01-05 14:30", freq="1min", volume=None):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]] if opens is None else np.asarray(opens, dtype=float)
    return normalize_bars(
        pd.DataFrame(
            {
                "timestamp": pd.date_range(start, periods=len(c), freq=freq, tz="UTC"),
                "open": o,
                "high": np.maximum(o, c),
                "low": np.minimum(o, c),
                "close": c,
                "volume": np.ones(len(c)) if volume is None else volume,
            }
        ),
        frequency=freq,
    )


def test_aggregate_and_realized_variance_hand_calculation():
    b = make_bars([100, 102, 98, 101])
    f = enrich_bars(b, "4min", origin=b.index[0], segments=2, bins=4)
    row = f.iloc[0]
    assert list(row[["open", "high", "low", "close", "volume"]]) == [100, 102, 98, 101, 4]
    r = np.log(np.array([100 / 100, 102 / 100, 98 / 102, 101 / 98]))
    assert row["path.realized_variance"] == pytest.approx(np.sum(r**2))
    assert row["path.realized_volatility"] == pytest.approx(np.sqrt(np.sum(r**2)))
    assert row["path.directional_efficiency"] == pytest.approx(abs(np.sum(r)) / np.sum(np.abs(r)))
    assert row["path.max_drawdown"] == pytest.approx(1 - 98 / 102)
    assert row["path.turn_count"] == 2
    assert row["quality.complete"] == 1
    assert row["available_at"] == b.index[0] + pd.Timedelta("4min")


def test_same_ohlcv_different_path_has_different_features():
    a = make_bars([100, 102, 98, 101])
    b = make_bars([100, 98, 102, 101])
    af = enrich_bars(a, "4min", origin=a.index[0], segments=2)
    bf = enrich_bars(b, "4min", origin=b.index[0], segments=2)
    assert_frame_equal(
        af[["open", "high", "low", "close", "volume"]],
        bf[["open", "high", "low", "close", "volume"]],
    )
    assert af["path.realized_variance"].iloc[0] != bf["path.realized_variance"].iloc[0]
    assert (
        af["path.high_first_time_fraction"].iloc[0] != bf["path.high_first_time_fraction"].iloc[0]
    )


@pytest.mark.parametrize("frequency", ["3min", "7min", "13min", "45min", "2h", "1D"])
def test_arbitrary_target_granularity(frequency):
    step = pd.Timedelta(frequency)
    n = int(step / pd.Timedelta("1min")) * 2
    b = make_bars(100 + np.arange(n) * 0.001)
    f = enrich_bars(b, frequency, origin=b.index[0], bins=3, segments=3)
    assert len(f) == 2
    assert f["quality.expected_bars"].eq(n / 2).all()
    assert f.available_at.iloc[0] == b.index[0] + step


def test_subminute_source_and_arbitrary_multiple():
    b = make_bars([100, 101, 102, 101, 100, 101], freq="1s")
    f = enrich_bars(b, "3s", source_frequency="1s", origin=b.index[0])
    assert len(f) == 2
    assert f["quality.expected_bars"].eq(3).all()


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_group_and_aggregation_accept_coarser_datetime_storage(unit):
    from pfe import compute_feature, compute_features

    bars = make_bars([100, 101, 102, 101, 100, 101])
    coarse = bars.copy()
    coarse.index = coarse.index.as_unit(unit)
    expected = compute_features(bars, groups=("returns",), windows=(2,))
    assert_frame_equal(compute_features(coarse, groups=("returns",), windows=(2,)), expected)
    np.testing.assert_allclose(
        compute_feature(coarse, "returns.simple"), expected["returns.simple"], equal_nan=True
    )
    options = dict(origin=bars.index[0], internal_scales=(1,))
    assert_frame_equal(enrich_bars(coarse, "3min", **options), enrich_bars(bars, "3min", **options))


def test_gaps_are_reported_without_inventing_realized_volatility():
    b = make_bars([100, 101, 102, 103, 104, 105])
    b = b.drop(b.index[2])
    f = enrich_bars(b, "6min", origin=b.index[0])
    assert f["quality.coverage"].iloc[0] == pytest.approx(5 / 6)
    assert f["quality.missing_bars"].iloc[0] == 1
    assert f["quality.largest_gap_seconds"].iloc[0] == 120
    assert np.isnan(f["path.realized_variance"].iloc[0])
    assert np.isfinite(f["path.realized_variance_observed"].iloc[0])


def test_edge_bins_as_of_and_future_causality():
    b = make_bars(100 + np.sin(np.arange(60)))
    prefix = enrich_bars(b.iloc[:31], "7min", origin=b.index[0])
    altered = b.copy()
    altered.iloc[31:] *= 2
    full = enrich_bars(altered, "7min", origin=b.index[0])
    assert_frame_equal(prefix, full.loc[prefix.index])
    at = enrich_bars(b, "7min", origin=b.index[0], as_of=b.index[30] + pd.Timedelta("1min"))
    assert_frame_equal(prefix, at)
    kept = enrich_bars(b.iloc[:31], "7min", origin=b.index[0], partial="keep")
    assert kept["quality.closed"].iloc[-1] == 0
    assert kept["quality.complete"].iloc[-1] == 0
    with pytest.raises(ValueError, match="Truncated"):
        enrich_bars(b.iloc[:31], "7min", origin=b.index[0], partial="raise")


def test_equal_time_segments_preserve_missing_slot():
    b = make_bars([100, 101, 102, 103, 104, 105])
    b = b.drop(b.index[2:4])
    f = enrich_bars(b, "6min", origin=b.index[0], segments=3, bins=3)
    assert np.isnan(f["path.segment1_volume"].iloc[0])
    assert f.filter(regex=r"profile.tpo_bin\d+_segment1$").iloc[0].eq(0).all()
    assert f["path.segment2_volume"].iloc[0] == 2


def test_tpo_matrix_and_flat_profile():
    b = make_bars([100, 100, 100, 100], volume=[0, 0, 0, 0])
    matrix, zones, centers, segment_ids = price_time_profile(
        b, start=b.index[0], duration=pd.Timedelta("4min"), bins=4, segments=2
    )
    assert matrix.shape == (4, 2)
    assert matrix.sum() == 2
    assert (zones == 2).all()
    assert (centers == 100).all()
    assert list(segment_ids) == [0, 0, 1, 1]
    f = enrich_bars(b, "4min", origin=b.index[0], bins=4, segments=2)
    assert f["profile.tpo_poc"].iloc[0] == 100
    assert f["path.realized_variance"].iloc[0] == 0
    assert not np.isinf(f.select_dtypes("number").to_numpy()).any()


def test_build_combines_both_families_and_is_causal():
    b = make_bars(100 + np.sin(np.arange(140) / 4))
    kwargs = dict(
        frequency="7min",
        origin=b.index[0],
        windows=(3,),
        groups=("returns", "technical", "patterns"),
    )
    prefix = build_features(b.iloc[:70], **kwargs)
    full = build_features(b, **kwargs)
    assert_frame_equal(prefix, full.loc[prefix.index])
    assert "path.realized_variance" in full
    assert "technical.rsi_w3" in full
    assert "patterns.cdlhammer" in full
    assert len(full) == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency": "90s"},
        {"frequency": "0min"},
        {"frequency": "1s"},
        {"frequency": "5min", "bins": 0},
        {"frequency": "5min", "segments": -1},
        {"frequency": "5min", "partial": "fill"},
        {"frequency": "5min", "offset": "30s"},
        {"frequency": "5min", "as_of": "2026-01-05"},
        {"frequency": "5min", "as_of": "2027-01-05T00:00:00Z"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    b = make_bars([100] * 10)
    with pytest.raises(ValueError):
        enrich_bars(b, **kwargs)


def test_multiple_timeframes_join_only_after_target_close():
    b = make_bars(100 + np.arange(42) * 0.01)
    kwargs = dict(
        frequencies=("7min", "14min"), origin=b.index[0], windows=(2,), groups=("returns",)
    )
    prefix = multi_timeframe_features(b.iloc[:5], **kwargs)
    full = multi_timeframe_features(b, **kwargs)
    assert_frame_equal(prefix, full.iloc[:5])
    assert full["tf[7min].close"].iloc[:6].isna().all()
    assert full["tf[7min].close"].iloc[6] == b.close.iloc[6]
    assert full["tf[7min].close"].iloc[7] == b.close.iloc[6]
    assert full["tf[7min].age_seconds"].iloc[6] == 0
    assert full["tf[7min].age_seconds"].iloc[7] == 60
    assert full["tf[14min].close"].iloc[:13].isna().all()


def test_empty_aggregation_keeps_configured_schema():
    b = make_bars([100, 101, 102, 103])
    empty = enrich_bars(b.iloc[:2], "4min", origin=b.index[0], bins=3, segments=2)
    full = enrich_bars(b, "4min", origin=b.index[0], bins=3, segments=2)
    assert empty.empty
    assert list(empty.columns) == list(full.columns)
