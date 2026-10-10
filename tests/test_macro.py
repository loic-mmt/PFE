"""Macro vintage references, schedule causality and sparse/dense alignment checks."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pfe import (
    MACRO_FEATURE_GROUPS,
    compute_macro_feature,
    compute_macro_features,
    load_macro,
    normalize_macro_calendar,
    normalize_macro_series,
)


def stamp(value):
    return pd.Timestamp(value, tz="UTC")


@pytest.fixture
def observations():
    return pd.DataFrame(
        dict(
            series=["rate"] * 5,
            observation_at=[stamp(f"2025-{m:02}-01") for m in (1, 2, 3, 1, 3)],
            available_at=[stamp(f"2025-04-01 12:{m:02}") for m in (1, 3, 5, 7, 9)],
            value=[1.0, 2.0, 4.0, 3.0, 5.0],
        )
    )


@pytest.fixture
def bars():
    return pd.date_range("2025-04-01 12:00", periods=12, freq="min", tz="UTC")


@pytest.fixture
def events():
    return pd.DataFrame(
        dict(
            event_id=[1, 2, 3, 4],
            event=["CPI"] * 4,
            scheduled_at=[stamp(f"2025-04-01 12:{m:02}") for m in (2, 4, 6, 10)],
            known_at=[stamp(f"2025-04-01 12:{m:02}") for m in (0, 5, 1, 9)],
            actual=[2.0, 5.0, 10.0, 9.0],
            forecast=[1.0, 2.0, 3.0, 4.0],
            previous=[0.0, 1.0, 2.0, 3.0],
            actual_available_at=[stamp(f"2025-04-01 12:{m:02}") for m in (3, 5, 7, 11)],
            forecast_available_at=[stamp(f"2025-04-01 12:{m:02}") for m in (0, 5, 1, 9)],
            previous_available_at=[stamp(f"2025-04-01 12:{m:02}") for m in (0, 5, 1, 9)],
        )
    )


def test_series_reference_periods_revisions_and_receipt_ages(observations, bars):
    out = compute_macro_features(bars, observations=observations, windows=(3,))
    assert out["macro.rate.level"].tolist() == [1, 1, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5]
    assert out["macro.rate.change"].iloc[4] == 2
    # An older-period revision updates the rolling window but cannot regress the level.
    assert out["macro.rate.mean_w3"].iloc[6] == 3
    assert out["macro.rate.revision"].iloc[6] == 2
    assert out["macro.rate.age_seconds"].iloc[6] == pytest.approx(120)
    assert out["macro.rate.std_w3"].iloc[6] == pytest.approx(np.std([3, 2, 4]))
    assert out["macro.rate.slope_w3"].iloc[6] == pytest.approx(0.5)
    assert out["macro.rate.zscore_w3"].iloc[6] == pytest.approx(1 / np.std([3, 2, 4]))
    assert out["macro.rate.mean_w3"].iloc[:4].isna().all()
    assert out["macro.rate.is_revision"].iloc[8] == 1
    assert out.attrs["available_at"] == "index + source_frequency"
    assert pd.Timedelta(out.attrs["source_frequency"]) == pd.Timedelta("1min")


def test_simultaneous_snapshot_uses_latest_period_and_all_distinct_observations(observations):
    frame = observations.iloc[:3].copy()
    frame["available_at"] = stamp("2025-04-01 12:05")
    index = pd.date_range("2025-04-01 12:03", periods=2, freq="min", tz="UTC")
    out = compute_macro_features(index, observations=frame, windows=(3,))
    assert out.iloc[0].isna().all()
    assert out["macro.rate.level"].iloc[1] == 4
    assert out["macro.rate.mean_w3"].iloc[1] == pytest.approx(7 / 3)


def test_calendar_against_direct_known_event_reference(events, bars):
    out = compute_macro_features(bars, events=events, groups="calendar", event_window="3min")
    for i, now in enumerate(bars + pd.Timedelta("1min")):
        visible = events.loc[events.known_at <= now, "scheduled_at"]
        future, past = visible[visible > now], visible[visible <= now]
        row = out.iloc[i]
        expected_next = (future.min() - now).total_seconds() if len(future) else np.nan
        expected_previous = (now - past.max()).total_seconds() if len(past) else np.nan
        if np.isnan(expected_next):
            assert np.isnan(row["macro.calendar.seconds_to_next_event"])
        else:
            assert row["macro.calendar.seconds_to_next_event"] == pytest.approx(expected_next)
        if np.isnan(expected_previous):
            assert np.isnan(row["macro.calendar.seconds_since_event"])
        else:
            assert row["macro.calendar.seconds_since_event"] == pytest.approx(expected_previous)
        assert (
            row["macro.calendar.events_next_window"]
            == ((visible > now) & (visible <= now + pd.Timedelta("3min"))).sum()
        )
        assert (
            row["macro.calendar.events_previous_window"]
            == ((visible <= now) & (visible >= now - pd.Timedelta("3min"))).sum()
        )
        assert row["macro.calendar.event_now"] == (visible == now).sum()


def test_surprises_wait_for_each_input_and_use_previous_event_scales(events, bars):
    out = compute_macro_features(bars, events=events, windows=(2,), groups="surprise")
    assert out["macro.event[CPI].forecast"].iloc[0] == 3
    assert out["macro.event[CPI].actual"].iloc[:2].isna().all()
    assert out["macro.event[CPI].surprise"].iloc[2] == 1
    assert out["macro.event[CPI].surprise"].iloc[4] == 3
    assert out["macro.event[CPI].point_in_time"].iloc[4] == 0  # late forecast
    assert out["macro.event[CPI].surprise_zscore_w2"].iloc[6] == 5
    assert out["macro.event[CPI].actual_change"].iloc[6] == 8
    assert out["macro.event[CPI].surprise_relative"].iloc[6] == pytest.approx(7 / 3)
    assert out["macro.event[CPI].age_seconds"].iloc[7] == pytest.approx(60)


def test_late_forecast_cannot_backfill_surprise(events, bars):
    event = events.iloc[:1].copy()
    event["forecast_available_at"] = stamp("2025-04-01 12:08")
    out = compute_macro_features(bars, events=event, groups="surprise")
    assert out["macro.event[CPI].surprise"].iloc[:7].isna().all()
    assert out["macro.event[CPI].surprise"].iloc[7] == 1
    assert out["macro.event[CPI].point_in_time"].iloc[7] == 0


def test_simultaneous_receipts_do_not_train_surprise_scale_on_same_batch(events, bars):
    frame = events.copy()
    frame["actual_available_at"] = stamp("2025-04-01 12:11")
    out = compute_macro_feature(
        bars, "macro.event[CPI].surprise_zscore_w2", events=frame, windows=(2,)
    )
    assert out.isna().all()


def test_future_data_cannot_change_any_past_feature(observations, events, bars):
    options = dict(observations=observations, events=events, windows=(2, 3))
    full = compute_macro_features(bars, **options)
    cutoff = stamp("2025-04-01 12:06")
    early_events = events.loc[events.known_at <= cutoff].copy()
    for field in ("actual", "forecast", "previous"):
        early_events.loc[early_events[f"{field}_available_at"] > cutoff, field] = np.nan
    prefix = compute_macro_features(
        bars[:6],
        observations=observations.loc[observations.available_at <= cutoff],
        events=early_events,
        windows=(2, 3),
    )
    assert_frame_equal(full.iloc[:6], prefix)


def test_spreads_zero_denominators_max_age_and_arbitrary_bar_duration(observations):
    left = observations.iloc[:1].copy()
    right = left.assign(series="other", value=0.0)
    index = pd.date_range("2025-04-01 12:00", periods=3, freq="7min", tz="UTC")
    out = compute_macro_features(
        index,
        observations=pd.concat([left, right]),
        pairs=(("rate", "other"),),
        groups="spreads",
        source_frequency="7min",
        max_age="8min",
    )
    assert out["macro.pair[rate,other].spread"].iloc[0] == 1
    assert out["macro.pair[rate,other].spread"].iloc[1:].isna().all()
    assert out["macro.pair[rate,other].ratio"].isna().all()


def test_individual_avoids_unrequested_group_execution(monkeypatch, observations, bars):
    import pfe.features.macro as module

    expected = compute_macro_features(bars, observations=observations)["macro.rate.level"]
    original = module._series_history

    def history(frame, metrics):
        assert metrics == {"level"}
        return original(frame, metrics)

    def forbidden(*args):
        raise AssertionError("Unrequested group executed")

    monkeypatch.setattr(module, "_series_history", history)
    monkeypatch.setattr(module, "_schedule", forbidden)
    monkeypatch.setattr(module, "_event_history", forbidden)
    actual = compute_macro_feature(bars, "macro.rate.level", observations=observations)
    assert_series_equal(actual, expected)


@pytest.mark.parametrize("group", MACRO_FEATURE_GROUPS)
def test_groups_match_individual_selection(group, observations, events, bars):
    options = dict(
        observations=observations, events=events, windows=(2,), pairs=(("rate", "other"),)
    )
    options["observations"] = pd.concat([observations, observations.assign(series="other")])
    all_features = compute_macro_features(bars, **options)
    selected = compute_macro_features(bars, groups=group, **options)
    assert len(selected.columns)
    assert_frame_equal(selected, all_features[selected.columns])
    for name in selected:
        assert_series_equal(compute_macro_feature(bars, name, **options), selected[name])


def test_empty_market_and_empty_calendar():
    index = pd.DatetimeIndex([], tz="UTC")
    events = pd.DataFrame(columns=["event_id", "event", "scheduled_at", "known_at"])
    out = compute_macro_features(index, events=events, groups="calendar")
    assert out.shape == (0, 5)
    dense = compute_macro_features(
        pd.date_range("2025-01-01", periods=2, tz="UTC"), events=events, groups="calendar"
    )
    assert dense["macro.calendar.events_next_window"].eq(0).all()
    assert dense["macro.calendar.seconds_to_next_event"].isna().all()


@pytest.mark.parametrize(
    "options",
    [
        dict(names="bad"),
        dict(groups="bad"),
        dict(windows=(1,)),
        dict(windows=(3, 3)),
        dict(windows=(True,)),
        dict(names="macro.rate.level", groups="series"),
        dict(pairs=(("rate", "rate"),)),
        dict(max_age="0min"),
    ],
)
def test_bad_options(options, observations, bars):
    with pytest.raises(ValueError):
        compute_macro_features(bars, observations=observations, **options)


@pytest.mark.parametrize(
    "change",
    [
        dict(available_at="2025-04-01"),
        dict(value=np.inf),
        dict(value=True),
        dict(series=""),
        dict(point_in_time="yes"),
    ],
)
def test_bad_series(change, observations):
    with pytest.raises(ValueError):
        normalize_macro_series(observations.assign(**change))


def test_duplicate_vintage_and_event_rejected(observations, events):
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_macro_series(pd.concat([observations, observations.iloc[:1]]))
    with pytest.raises(ValueError, match="unique"):
        normalize_macro_calendar(pd.concat([events, events.iloc[:1]]))
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_macro_calendar(events.drop(columns="actual_available_at"))


def test_load_snapshot_never_claims_historical_availability(tmp_path):
    payload = dict(
        provider="LSE",
        snapshot_at="2026-01-01T00:00:00Z",
        limit_reached={"calendar": True},
        series={"rate": [dict(date="2025-01-01", value=2)]},
        calendar=[
            dict(
                id=1,
                event="Payroll",
                datetime="2025-01-02T13:30:00Z",
                actual="120K",
                consensus="110K",
                previous="100K",
                created_at="2025-12-01T00:00:00Z",
            )
        ],
    )
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(payload))
    data = load_macro(path)
    assert data.calendar.actual.iloc[0] == 120000
    assert not data.series.point_in_time.any()
    assert not data.calendar.point_in_time.any()
    assert data.limit_reached["calendar"]
    historical = pd.date_range("2025-01-01", periods=3, tz="UTC")
    past = compute_macro_features(historical, observations=data.series, events=data.calendar)
    assert past["macro.rate.level"].isna().all()
    assert past["macro.event[Payroll].surprise"].isna().all()
    assert past["macro.calendar.events_next_window"].eq(0).all()
    current = compute_macro_features(
        pd.DatetimeIndex([data.snapshot_at]), observations=data.series, events=data.calendar
    )
    assert current["macro.event[Payroll].surprise"].iloc[0] == 10000
    with pytest.raises(ValueError, match="max_bytes"):
        load_macro(path, max_bytes=10)


@pytest.mark.parametrize(
    "raw,expected", [("3.1%", 3.1), ("-1.2M", -1200000), ("1,200", 1200), (".4B", 400000000)]
)
def test_provider_numeric_units(raw, expected):
    from pfe.data.macro import _lse_number

    assert _lse_number(raw) == expected


def test_provider_unknown_numeric_format_rejected():
    from pfe.data.macro import _lse_number

    with pytest.raises(ValueError):
        _lse_number("unavailable text")


def test_pandas_repr_concat_and_chunk_alignment(monkeypatch, observations, events, bars):
    import pfe.features.macro as module

    options = dict(observations=observations, events=events, windows=(2, 3))
    full = compute_macro_features(bars, **options)
    monkeypatch.setattr(module, "_CHUNK", 2)
    assert_frame_equal(full, compute_macro_features(bars, **options))
    assert "macro" in repr(full)
    chunks = pd.concat(
        [compute_macro_features(bars[:6], **options), compute_macro_features(bars[6:], **options)]
    )
    assert_frame_equal(full, chunks)
