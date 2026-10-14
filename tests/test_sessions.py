"""Exchange date labels are separate from real session times."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe.data.sessions import exchange_schedule, normalize_session_schedule


def schedule():
    return pd.DataFrame(
        {
            "session_date": ["2026-03-06", "2026-03-09", "2026-03-10", "2026-03-11"],
            "open_at": [
                "2026-03-06T14:30:00Z",
                "2026-03-09T13:30:00Z",
                "2026-03-10T13:30:00Z",
                "2026-03-11T13:30:00Z",
            ],
            "close_at": [
                "2026-03-06T21:00:00Z",
                "2026-03-09T20:00:00Z",
                "2026-03-10T20:00:00Z",
                "2026-03-11T20:00:00Z",
            ],
        }
    )


def test_injected_calendar_preserves_real_dst_bounds_and_is_idempotent():
    raw = schedule().iloc[::-1]
    out = normalize_session_schedule(raw)
    assert out.index.tolist() == schedule().session_date.tolist()
    assert out.open_at.iloc[1].hour == 13
    assert out.open_at.iloc[0].hour == 14
    assert_frame_equal(normalize_session_schedule(out), out)
    assert raw.open_at.iloc[0] == "2026-03-11T13:30:00Z"
    subset = exchange_schedule("2026-03-07", "2026-03-10", calendar=out)
    assert subset.index.tolist() == ["2026-03-09", "2026-03-10"]


def test_real_nyse_calendar_dst_holiday_and_early_close():
    pytest.importorskip("exchange_calendars")
    march = exchange_schedule("2026-03-06", "2026-03-10", calendar="XNYS")
    assert march.index.tolist() == ["2026-03-06", "2026-03-09", "2026-03-10"]
    assert march.loc["2026-03-06", "open_at"] == pd.Timestamp("2026-03-06T14:30:00Z")
    assert march.loc["2026-03-09", "open_at"] == pd.Timestamp("2026-03-09T13:30:00Z")
    november = exchange_schedule("2026-11-25", "2026-11-30")
    assert "2026-11-26" not in november.index
    assert november.loc["2026-11-27", "close_at"] == pd.Timestamp("2026-11-27T18:00:00Z")
    assert exchange_schedule("2026-11-26", "2026-11-26").empty
    assert exchange_schedule("2026-03-07", "2026-03-08").empty


def test_unknown_named_calendar_has_a_public_validation_error():
    pytest.importorskip("exchange_calendars")
    with pytest.raises(ValueError, match="Cannot create exchange calendar 'UNKNOWN_CALENDAR'"):
        exchange_schedule("2026-03-06", "2026-03-10", calendar="UNKNOWN_CALENDAR")


@pytest.mark.parametrize(
    "change,message",
    [
        ({"session_date": "2026-03-06"}, "Duplicate"),
        ({"open_at": "2026-03-06"}, "timezone-aware"),
        ({"close_at": "2026-03-05T21:00:00Z"}, "later"),
        ({"open_at": "2026-03-06T20:00:00Z"}, "chronological order|overlap"),
        ({"session_date": "2026-02-30"}, "valid calendar date"),
        ({"session_date": "2026-03-06T14:30:00Z"}, "midnight"),
    ],
)
def test_invalid_schedule_rejected(change, message):
    raw = schedule()
    for key, value in change.items():
        raw.loc[1, key] = value
    with pytest.raises(ValueError, match=message):
        normalize_session_schedule(raw)


def test_breaks_are_explicit_validated_and_never_invented():
    raw = schedule().iloc[:1].copy()
    raw["break_start_at"] = "2026-03-06T17:00:00Z"
    with pytest.raises(ValueError, match="require break_start_at"):
        normalize_session_schedule(raw)
    raw["break_end_at"] = "2026-03-06T18:00:00Z"
    assert normalize_session_schedule(raw).break_start_at.notna().all()
    raw["break_end_at"] = "2026-03-06T21:00:00Z"
    with pytest.raises(ValueError, match="strictly inside"):
        normalize_session_schedule(raw)


def test_nonoverlapping_equal_boundaries_empty_and_invalid_bounds():
    raw = schedule().iloc[:2].copy()
    raw.loc[1, "open_at"] = raw.loc[0, "close_at"]
    assert len(normalize_session_schedule(raw)) == 2
    assert normalize_session_schedule(schedule().iloc[:0]).empty
    with pytest.raises(ValueError, match="start"):
        exchange_schedule("2026-03-10", "2026-03-06", calendar=schedule())
