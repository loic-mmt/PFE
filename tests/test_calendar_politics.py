"""Real civil dates, electoral boundaries and unavailable political outcomes."""

import numpy as np
import pandas as pd
import pytest
from test_daily_features import rows as daily_rows

from pfe import (
    compute_calendar_features,
    compute_daily_features,
    compute_features,
    compute_political_features,
    compute_selected_features,
    normalize_political_history,
    normalize_session_schedule,
    us_election_calendar,
)


def test_local_calendar_iso_year_leap_day_and_cyclic_encoding():
    times = pd.DatetimeIndex(["2021-01-01T00:30Z", "2021-01-01T15:00Z", "2024-02-29T15:00Z"])
    result = compute_calendar_features(times)
    assert result["calendar.year"].tolist() == [2020, 2021, 2024]
    assert result["calendar.iso_year"].tolist() == [2020, 2020, 2024]
    assert result["calendar.iso_week"].iloc[:2].tolist() == [53, 53]
    assert result["calendar.days_in_month"].iloc[-1] == 29
    assert result["calendar.is_month_end"].iloc[-1] == 1
    assert result["calendar.day_of_year"].iloc[-1] == 60
    assert result["calendar.day_of_year_sin"].iloc[-1] == pytest.approx(
        np.sin(2 * np.pi * 59 / 366)
    )
    assert np.isnan(result["calendar.sessions_remaining_month"].iloc[-1])
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_calendar_features(pd.date_range("2026-01-01", periods=2))


def test_ohlcv_exact_calendar_matches_group_and_future_extension():
    times = pd.date_range("2026-01-30T14:30Z", periods=4, freq="min", name="timestamp")
    bars = pd.DataFrame(dict(open=10.0, high=11.0, low=9.0, close=10.0, volume=1.0), index=times)
    names = ("calendar.month", "calendar.iso_week", "calendar.quarter", "calendar.month_cos")
    full = compute_features(bars, groups=("calendar",))
    selected = compute_selected_features(bars, names)
    pd.testing.assert_frame_equal(full.loc[:, names], selected, check_flags=False)
    prefix = compute_selected_features(bars.iloc[:2], names)
    pd.testing.assert_frame_equal(prefix, selected.iloc[:2])


def test_daily_calendar_uses_actual_dates_even_when_price_missing():
    days = ["2026-03-06", "2026-03-09", "2026-03-10"]
    schedule = normalize_session_schedule(
        pd.DataFrame(
            dict(
                session_date=days,
                open_at=[day + "T14:30Z" for day in days],
                close_at=[day + "T21:00Z" for day in days],
            )
        )
    )
    bars = daily_rows(schedule, prices=(100, 101, 102))
    bars.loc[1, ["open", "high", "low", "close", "volume"]] = np.nan
    result = compute_daily_features(
        bars, schedule=schedule, names=("calendar.year", "calendar.weekday", "returns.simple")
    )
    assert result["calendar.year"].tolist() == [2026, 2026, 2026]
    assert result["calendar.weekday"].tolist() == [4, 0, 1]
    assert result["mask.calendar.year"].all()
    assert result["returns.simple"].isna().all()
    with pytest.raises(ValueError, match="intraday"):
        compute_daily_features(bars, schedule=schedule, names="calendar.hour_sin")


def test_exchange_positions_require_full_period_and_respect_omitted_holiday():
    days = pd.bdate_range("2026-03-01", "2026-03-31").delete(3).strftime("%Y-%m-%d")
    schedule = normalize_session_schedule(
        pd.DataFrame(
            dict(
                session_date=days,
                open_at=[day + "T14:30Z" for day in days],
                close_at=[day + "T21:00Z" for day in days],
            )
        )
    )
    decisions = pd.DatetimeIndex([days[0] + "T21:00Z", days[-1] + "T21:00Z"])
    incomplete = compute_calendar_features(decisions, schedule=schedule)
    assert incomplete["calendar.session_index_month"].isna().all()
    schedule.attrs.update(start="2026-03-01", end="2026-03-31")
    complete = compute_calendar_features(decisions, schedule=schedule)
    assert complete["calendar.session_index_month"].tolist() == [1, len(days)]
    assert complete["calendar.sessions_remaining_month"].tolist() == [len(days) - 1, 0]
    assert complete["calendar.is_last_session_month"].tolist() == [0, 1]
    assert complete["calendar.session_index_quarter"].isna().all()


def state(kind, party, effective="2025-01-20T17:00Z", available=None, **changes):
    return (
        dict(
            record_id=kind + effective,
            state_kind=kind,
            party=party,
            effective_from=effective,
            available_at=available or effective,
            source_id="fixture",
            revision_id="1",
            point_in_time=True,
        )
        | changes
    )


def test_midterm_calendar_and_nine_calendar_month_boundary():
    dates = pd.DatetimeIndex(
        [
            "2022-11-08T23:00Z",
            "2022-11-09T04:59Z",
            "2022-11-09T05:00Z",
            "2023-08-08T16:00Z",
            "2023-08-09T16:00Z",
        ]
    )
    result = compute_political_features(dates)
    assert result["politics.us.post_midterm_9m"].tolist() == [0, 0, 1, 1, 0]
    assert result["politics.us.cycle_year"].tolist() == [2, 2, 2, 3, 3]
    calendar = us_election_calendar(2022, 2024)
    assert calendar.election_date.tolist() == ["2022-11-08", "2024-11-05"]
    assert not any("bullish" in name for name in result)
    assert result["politics.us.president.republican"].isna().all()


def test_party_transition_needs_effective_and_available_instants():
    history = pd.DataFrame(
        [
            state("president", "democratic", effective="2021-01-20T17:00Z"),
            state("president", "republican", available="2025-01-20T17:05Z"),
            state("house", "republican"),
            state("senate", "republican"),
        ]
    )
    decisions = pd.DatetimeIndex(["2025-01-20T16:59Z", "2025-01-20T17:00Z", "2025-01-20T17:05Z"])
    result = compute_political_features(decisions, history=history)
    assert result["politics.us.president.republican"].tolist() == [0, 0, 1]
    assert np.isnan(result["politics.us.unified_government"].iloc[0])
    assert result["politics.us.divided_government"].iloc[1] == 1
    assert result["politics.us.unified_government"].iloc[2] == 1
    future = state("house", "democratic", effective="2027-01-03T17:00Z")
    after = compute_political_features(
        decisions, history=pd.concat([history, pd.DataFrame([future])])
    )
    pd.testing.assert_frame_equal(result, after)


def test_political_vintage_ambiguity_and_non_ordinal_parties():
    first = state("president", "democratic", effective="2021-01-20T17:00Z")
    revised = first | dict(party="republican", revision_id="2", available_at="2026-01-02T12:00Z")
    decisions = pd.DatetimeIndex(["2026-01-01T12:00Z", "2026-01-02T12:00Z"])
    result = compute_political_features(decisions, history=pd.DataFrame([first, revised]))
    assert result["politics.us.president.democratic"].tolist() == [1, 0]
    conflicting = first | dict(source_id="other", record_id="other")
    result = compute_political_features(decisions, history=pd.DataFrame([first, conflicting]))
    assert result["politics.us.president.mask"].eq(0).all()
    assert result["politics.us.president.quality.ambiguous"].eq(1).all()
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_political_history(pd.DataFrame([first, first]))


def test_future_schedule_declarations_and_split_chamber_remain_missing():
    days = pd.bdate_range("2026-03-01", "2026-03-31").strftime("%Y-%m-%d")
    schedule = normalize_session_schedule(
        pd.DataFrame(
            dict(
                session_date=days,
                open_at=[day + "T14:30Z" for day in days],
                close_at=[day + "T21:00Z" for day in days],
                available_at=["2026-03-03T12:00Z"] * len(days),
            )
        )
    )
    schedule.attrs.update(start="2026-03-01", end="2026-03-31")
    out = compute_calendar_features(
        pd.DatetimeIndex(["2026-03-02T21:00Z", "2026-03-03T21:00Z"]), schedule=schedule
    )
    assert np.isnan(out["calendar.session_index_month"].iloc[0])
    assert out["calendar.session_index_month"].iloc[1] == 2
    history = pd.DataFrame(
        [state("president", "democratic"), state("house", "democratic"), state("senate", "split")]
    )
    political = compute_political_features(pd.DatetimeIndex(["2026-01-01T12:00Z"]), history=history)
    assert np.isnan(political["politics.us.unified_government"].iloc[0])
    assert political["politics.us.senate.split"].iloc[0] == 1
