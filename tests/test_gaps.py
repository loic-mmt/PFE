"""Opening returns, exact session endpoints and local-calendar weekend closures."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pfe import (
    compute_feature,
    compute_features,
    compute_selected_features,
    iter_features,
    normalize_bars,
)

NAMES = ("session.night_gap", "session.weekend_gap")


def make_bars(rows, timezone="America/New_York"):
    times, opening, close = zip(*rows, strict=True)
    return normalize_bars(
        pd.DataFrame(
            {
                "open": opening,
                "close": close,
                "high": np.maximum(opening, close) + 1,
                "low": np.minimum(opening, close) - 1,
                "volume": 100.0,
            },
            index=pd.DatetimeIndex(times).tz_localize(timezone).tz_convert("UTC"),
        )
    )


@pytest.fixture
def sessions():
    # DST starts on Sunday March 8: Monday's 09:30 is one UTC hour earlier.
    return make_bars(
        [
            ("2026-03-05 15:59", 99, 100),
            ("2026-03-06 09:30", 110, 111),
            ("2026-03-06 09:31", 112, 113),
            ("2026-03-06 15:59", 119, 120),
            ("2026-03-06 16:00", 990, 999),
            ("2026-03-09 09:30", 126, 127),
            ("2026-03-09 09:31", 128, 129),
            ("2026-03-09 15:59", 124, 125),
            ("2026-03-10 09:29", 500, 501),
            ("2026-03-10 09:30", 120, 121),
            ("2026-03-10 09:31", 122, 123),
        ]
    )


@pytest.mark.parametrize("policy", ["reset", "allow"])
def test_open_gap_alias_and_gap_policy(sessions, policy):
    full = compute_features(sessions, windows=(2,), groups=("returns",), gap_policy=policy)
    assert_series_equal(full["returns.open_gap"], full["returns.gap"], check_names=False)
    selected = compute_feature(sessions, "returns.open_gap", gap_policy=policy)
    assert_series_equal(selected, full["returns.open_gap"])
    assert selected.iloc[2] == pytest.approx(112 / 111 - 1)
    if policy == "reset":
        assert np.isnan(selected.iloc[5])
    else:
        assert selected.iloc[5] == pytest.approx(126 / 999 - 1)


@pytest.mark.parametrize("policy", ["reset", "allow"])
def test_session_gaps_hand_values_dst_extended_hours_and_persistence(sessions, policy):
    result = compute_selected_features(sessions, NAMES, gap_policy=policy)
    assert result.iloc[0].isna().all()
    assert result["session.night_gap"].iloc[1:4].to_numpy() == pytest.approx([0.1] * 3)
    assert result["session.weekend_gap"].iloc[1:4].isna().all()
    assert result["session.weekend_gap"].iloc[5:8].to_numpy() == pytest.approx([0.05] * 3)
    assert result["session.night_gap"].iloc[5:8].isna().all()
    assert result["session.night_gap"].iloc[9:].to_numpy() == pytest.approx([-0.04] * 2)
    assert result["session.weekend_gap"].iloc[9:].isna().all()
    assert result.iloc[[4, 8]].isna().all().all()
    assert sessions.index[1].hour == 14 and sessions.index[5].hour == 13


@pytest.mark.parametrize("policy", ["reset", "allow"])
def test_open_gap_streaming_across_batches_and_session_boundaries(sessions, policy):
    names = ("returns.open_gap", "returns.gap")
    expected = compute_selected_features(sessions, names, gap_policy=policy)
    batches = (sessions.iloc[i : i + 2] for i in range(0, len(sessions), 2))
    actual = pd.concat(iter_features(batches, names, gap_policy=policy))
    assert_frame_equal(actual, expected, check_freq=False)


def test_independent_and_group_gaps_match_and_are_causal(sessions):
    before = sessions.copy(deep=True)
    names = ("returns.open_gap", *NAMES)
    group = compute_features(sessions, windows=(2,), groups=("returns", "calendar"))
    selected = compute_selected_features(sessions, names)
    assert_frame_equal(selected, group[list(names)])
    for name in names:
        assert_series_equal(compute_feature(sessions, name), selected[name])
    for end in range(1, len(sessions)):
        prefix = compute_selected_features(sessions.iloc[:end], names)
        future = sessions.copy()
        future.iloc[end:] *= 3
        full = compute_selected_features(future, names)
        assert_frame_equal(prefix, full.iloc[:end])
    assert_frame_equal(before, sessions)


@pytest.mark.parametrize("missing_position", [3, 5])
def test_missing_exact_close_or_open_yields_nan(sessions, missing_position):
    bars = sessions.drop(sessions.index[missing_position])
    result = compute_selected_features(bars, NAMES)
    monday = result.index.tz_convert("America/New_York").day == 9
    assert result.loc[monday].isna().all().all()


def test_weekend_gap_includes_monday_holiday():
    bars = make_bars([("2026-05-22 15:59", 100, 100), ("2026-05-26 09:30", 103, 103)])
    result = compute_selected_features(bars, NAMES)
    assert result["session.weekend_gap"].iloc[1] == pytest.approx(0.03)
    assert np.isnan(result["session.night_gap"].iloc[1])


def test_custom_session_and_source_frequency_with_zero_gap():
    bars = make_bars(
        [("2026-01-05 09:55", 100, 100), ("2026-01-06 09:00", 100, 100)],
        timezone="UTC",
    )
    result = compute_selected_features(
        bars,
        NAMES,
        source_frequency="5min",
        session_timezone="UTC",
        session_start="09:00",
        session_end="10:00",
    )
    assert result["session.night_gap"].iloc[1] == 0
    assert np.isnan(result["session.weekend_gap"].iloc[1])


def test_selected_gap_skips_other_session_work_and_shares_dependencies(sessions, monkeypatch):
    import pfe.features.sessions as session_module

    calls = []
    original = session_module._session_gap_inputs

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unrelated session aggregation executed")

    monkeypatch.setattr(session_module, "_session_gap_inputs", tracked)
    monkeypatch.setattr(pd.DataFrame, "groupby", forbidden)
    # A fresh configuration ensures the planner captures the tracked function.
    result = compute_selected_features(sessions, NAMES, opening_range="7min")
    assert result["session.weekend_gap"].iloc[5] == pytest.approx(0.05)
    assert len(calls) == 1
