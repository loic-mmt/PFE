"""Sparse tombstones and calendar revisions on explicit intraday decisions."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe import compute_macro_features
from pfe.data.macro_calendar import macro_calendar_asof, normalize_macro_calendar_history


def evaluate(frame, decisions, **options):
    index = pd.to_datetime(decisions, utc=True) - pd.Timedelta("1ns")
    return compute_macro_features(index, calendar_history=frame, source_frequency="1ns", **options)


def version(at="2026-01-02T10:00:00Z", revision="1", **changes):
    return (
        dict(
            event_id="cpi:jan",
            event="CPI",
            scheduled_at="2026-01-02T12:00:00Z",
            available_at=at,
            revision_id=revision,
            unit="percent",
            forecast=2.0,
            forecast_available_at="2026-01-02T10:00:00Z",
            point_in_time=True,
        )
        | changes
    )


def test_calendar_reschedule_cancel_and_future_invariance():
    rows = pd.DataFrame(
        [
            version(),
            version("2026-01-02T11:00:00Z", "2", scheduled_at="2026-01-02T13:00:00Z"),
            version("2026-01-02T11:30:00Z", "3", status="cancelled"),
        ]
    )
    decisions = ["2026-01-02T10:00:00Z", "2026-01-02T11:00:00Z", "2026-01-02T11:30:00Z"]
    result = evaluate(rows, decisions, groups="calendar")
    assert result["macro.calendar.seconds_to_next_event"].iloc[:2].tolist() == [7200, 7200]
    assert np.isnan(result["macro.calendar.seconds_to_next_event"].iloc[-1])
    assert_frame_equal(result.iloc[:1], evaluate(rows.iloc[:1], decisions[:1], groups="calendar"))
    assert macro_calendar_asof(rows, decisions[-1]).status.tolist() == ["cancelled"]


def test_actual_revisions_freeze_forecast_before_real_release_and_withdrawal():
    release = "2026-01-02T12:00:00Z"
    rows = pd.DataFrame(
        [
            version(),
            version(
                release,
                "2",
                actual=3.0,
                actual_available_at=release,
                release_at=release,
                forecast=999.0,
                forecast_available_at=release,
                status="released",
            ),
            version(
                "2026-01-02T12:05:00Z",
                "3",
                actual=4.0,
                actual_available_at=release,
                release_at=release,
                status="released",
            ),
            version("2026-01-02T12:06:00Z", "4", actual=None, release_at=release),
        ]
    )
    result = evaluate(
        rows, [release, "2026-01-02T12:05:00Z", "2026-01-02T12:06:00Z"], groups="surprise"
    )
    assert result["macro.event[CPI].surprise"].iloc[:2].tolist() == [1.0, 2.0]
    assert np.isnan(result["macro.event[CPI].surprise"].iloc[-1])
    assert np.isnan(result["macro.event[CPI].actual"].iloc[-1])


def test_unknown_release_and_prerelease_forecast_withdrawal_mask_surprise():
    release = "2026-01-02T12:00:00Z"
    for withdrawal in (False, True):
        rows = [version()]
        if withdrawal:
            rows.append(version("2026-01-02T11:00:00Z", "2", forecast=None))
        rows.append(
            version(
                release,
                "3",
                actual=3.0,
                actual_available_at=release,
                release_at=release if withdrawal else None,
            )
        )
        result = evaluate(pd.DataFrame(rows), [release], groups="surprise")
        assert np.isnan(result["macro.event[CPI].surprise"].iloc[0])


def test_duplicate_revision_and_future_release_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_macro_calendar_history(pd.DataFrame([version(), version()]))
    with pytest.raises(ValueError, match="release_at"):
        normalize_macro_calendar_history(pd.DataFrame([version(release_at="2026-01-03T00:00:00Z")]))


def test_series_null_revision_restore_and_window_preserve_economic_periods():
    points = pd.DataFrame(
        dict(
            series=["x"] * 5,
            observation_at=pd.to_datetime(
                ["2025-01-01", "2025-02-01", "2025-02-01", "2025-01-01", "2025-02-01"], utc=True
            ),
            available_at=pd.date_range("2026-01-01", periods=5, tz="UTC"),
            value=[1.0, 3.0, None, None, 5.0],
        )
    )
    index = pd.date_range("2026-01-01", periods=5, tz="UTC")
    result = compute_macro_features(
        index, observations=points, source_frequency="1ns", windows=(2,)
    )
    assert result["macro.x.level"].iloc[:2].tolist() == [1, 3]
    assert result["macro.x.level"].iloc[2:4].isna().all()
    assert result["macro.x.level"].iloc[-1] == 5
    assert result["macro.x.mean_w2"].iloc[2:].isna().all()
    assert result["macro.x.is_revision"].iloc[2] == 1
