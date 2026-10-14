"""Daily source labels, availability, RTH coverage and explicit missingness."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe.data.daily import aggregate_session_bars, load_daily_bars, normalize_daily_bars
from pfe.data.sessions import normalize_session_schedule


@pytest.fixture
def calendar():
    return normalize_session_schedule(
        pd.DataFrame(
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
    )


def daily_rows():
    return pd.DataFrame(
        {
            "timestamp": ["2026-03-06", "2026-03-09"],
            "instrument_id": ["security:a"] * 2,
            "open": [99.0, 109.0],
            "high": [101.0, 111.0],
            "low": [98.0, 108.0],
            "close": [100.0, 110.0],
            "volume": [1000.0, 1200.0],
        }
    )


def intraday(calendar, sessions=None, instrument="security:a"):
    records = []
    for day, row in calendar.iterrows():
        if sessions is not None and day not in sessions:
            continue
        times = pd.date_range(row.open_at, row.close_at, freq="min", inclusive="left")
        for index, time in enumerate(times):
            records.append(
                dict(
                    instrument_id=instrument,
                    timestamp=time,
                    open=100.0 + index / 100,
                    high=100.1 + index / 100,
                    low=99.9 + index / 100,
                    close=100.05 + index / 100,
                    volume=10.0,
                )
            )
    return pd.DataFrame(records)


def test_native_daily_receipt_actual_close_metadata_and_no_inferred_coverage(calendar):
    out = normalize_daily_bars(daily_rows(), schedule=calendar, ingested_at="2026-10-12T10:00:00Z")
    assert out.session_date.tolist() == ["2026-03-06", "2026-03-09"]
    assert out.timestamp.tolist() == calendar.open_at.iloc[:2].tolist()
    assert out.session_close.tolist() == calendar.close_at.iloc[:2].tolist()
    assert out.available_at.eq(pd.Timestamp("2026-10-12T10:00:00Z")).all()
    assert out.entity_id.isna().all()
    assert out.currency.isna().all()
    assert out.price_adjustment.eq("unknown").all()
    assert out.session_scope.eq("unknown").all()
    assert out["quality.session_complete"].isna().all()
    assert not out.point_in_time.any()
    assert out["quality.feature_eligible"].all()
    assert_frame_equal(normalize_daily_bars(out, schedule=calendar), out)


def test_explicit_open_close_labels_and_historical_assumptions(calendar):
    raw = daily_rows()
    for label, values in (
        ("open", calendar.open_at.iloc[:2]),
        ("close", calendar.close_at.iloc[:2]),
    ):
        selected = raw.assign(timestamp=values.to_list())
        out = normalize_daily_bars(
            selected,
            schedule=calendar,
            label=label,
            availability_basis="event_time",
            latency="2s",
            price_adjustment="split_adjusted",
            volume_adjustment="unknown",
            currency="USD",
        )
        assert out.available_at.equals(out.session_close + pd.Timedelta("2s"))
        assert not out.point_in_time.any()
        assert out.price_adjustment.eq("split_adjusted").all()
    encoded = raw.assign(timestamp=pd.to_datetime(raw.timestamp, utc=True))
    assert (
        len(normalize_daily_bars(encoded, schedule=calendar, availability_basis="event_time")) == 2
    )


def test_missing_daily_values_keep_nan_and_quality_masks(calendar):
    raw = daily_rows()
    raw.loc[1, ["open", "high", "low", "close", "volume"]] = np.nan
    out = normalize_daily_bars(raw, schedule=calendar, availability_basis="event_time")
    assert out.loc[1, ["open", "high", "low", "close", "volume"]].isna().all()
    assert not out.loc[1, "quality.feature_eligible"]
    assert not out.loc[1, "quality.volume_known"]
    assert not out.loc[1, "quality.price_complete"]


def test_daily_input_calendar_and_temporal_validation(calendar):
    raw = daily_rows()
    changes = [
        ({"timestamp": "2026-03-07"}, "non-session"),
        ({"available_at": "2026-03-06T00:00:00Z"}, "real session close"),
        ({"session_close": "2026-03-06T20:00:00Z"}, "disagrees"),
        ({"timestamp": "2026-03-06T15:00:00Z"}, "midnight"),
        ({"price_adjustment": "adjusted"}, "price_adjustment"),
        ({"volume_adjustment": "total_return"}, "volume_adjustment"),
        ({"ingested_at": "2026-10-12"}, "timezone-aware"),
        ({"volume": -1}, "non-negative"),
        ({"high": 99.5}, "high"),
        ({"close": np.inf}, "finite"),
    ]
    for change, message in changes:
        selected = raw.iloc[:1].assign(**change)
        with pytest.raises(ValueError, match=message):
            normalize_daily_bars(selected, schedule=calendar, availability_basis="event_time")
    with pytest.raises(ValueError, match="ingested_at"):
        normalize_daily_bars(raw, schedule=calendar)
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_daily_bars(
            pd.concat([raw, raw.iloc[:1]]), schedule=calendar, availability_basis="event_time"
        )


def test_rth_full_grid_extended_exclusion_and_early_close():
    calendar = normalize_session_schedule(
        pd.DataFrame(
            {
                "session_date": ["2026-11-27"],
                "open_at": ["2026-11-27T14:30:00Z"],
                "close_at": ["2026-11-27T18:00:00Z"],
            }
        )
    )
    raw = intraday(calendar)
    extra = raw.iloc[:1].assign(timestamp=pd.Timestamp("2026-11-27T13:00:00Z"), high=9999.0)
    out = aggregate_session_bars(
        pd.concat([raw, extra]), schedule=calendar, availability_basis="event_time"
    )
    assert out.loc[0, "quality.observed_bars"] == 210
    assert out.loc[0, "quality.expected_bars"] == 210
    assert out.loc[0, "quality.session_complete"]
    assert out.loc[0, "high"] < 200
    assert out.loc[0, "volume"] == 2100
    assert out.loc[0, "available_at"] == pd.Timestamp("2026-11-27T18:00:00Z")
    assert out.attrs["excluded_extended_bars"] == {"security:a": 1}
    with pytest.raises(ValueError, match="outside-session"):
        aggregate_session_bars(
            pd.concat([raw, extra]),
            schedule=calendar,
            availability_basis="event_time",
            extended="raise",
        )


def test_incomplete_intraday_and_missing_session_are_explicit_without_filling(calendar):
    raw = intraday(calendar, sessions=["2026-03-06", "2026-03-10"])
    raw = raw.drop(raw.index[10])
    out = aggregate_session_bars(raw, schedule=calendar, availability_basis="event_time")
    assert out.session_date.tolist() == ["2026-03-06", "2026-03-09", "2026-03-10"]
    assert out["quality.observed_bars"].tolist() == [389, 0, 390]
    assert out["quality.missing_bars"].tolist() == [1, 390, 0]
    assert out["quality.session_complete"].tolist() == [False, False, True]
    assert out["quality.feature_eligible"].tolist() == [False, False, True]
    assert out.loc[1, ["open", "high", "low", "close", "volume"]].isna().all()
    dropped = aggregate_session_bars(
        raw, schedule=calendar, availability_basis="event_time", partial="drop"
    )
    assert dropped.session_date.tolist() == ["2026-03-10"]
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_session_bars(
            raw, schedule=calendar, availability_basis="event_time", partial="raise"
        )


def test_declared_listing_delisting_coverage_does_not_extend_end(calendar):
    raw = intraday(calendar, sessions=["2026-03-06", "2026-03-09"])
    coverage = pd.DataFrame(
        {
            "instrument_id": ["security:a"],
            "first_session": ["2026-03-06"],
            "last_session": ["2026-03-09"],
        }
    )
    out = aggregate_session_bars(
        raw, schedule=calendar, coverage=coverage, availability_basis="event_time"
    )
    assert out.session_date.tolist() == ["2026-03-06", "2026-03-09"]
    assert out["quality.coverage_declared"].all()
    coverage["last_session"] = "2026-03-10"
    with_missing = aggregate_session_bars(
        raw, schedule=calendar, coverage=coverage, availability_basis="event_time"
    )
    assert not with_missing.iloc[-1]["quality.session_complete"]
    assert with_missing.iloc[-1]["volume"] != 0


def test_trading_breaks_do_not_count_as_missing_source_bars():
    calendar = normalize_session_schedule(
        pd.DataFrame(
            {
                "session_date": ["2026-03-06"],
                "open_at": ["2026-03-06T01:00:00Z"],
                "close_at": ["2026-03-06T03:00:00Z"],
                "break_start_at": ["2026-03-06T01:30:00Z"],
                "break_end_at": ["2026-03-06T02:30:00Z"],
            }
        )
    )
    raw = intraday(calendar)
    outside = raw.timestamp.between(
        pd.Timestamp("2026-03-06T01:30:00Z"), pd.Timestamp("2026-03-06T02:30:00Z"), inclusive="left"
    )
    out = aggregate_session_bars(raw, schedule=calendar, availability_basis="event_time")
    assert out["quality.expected_bars"].iloc[0] == 60
    assert out["quality.observed_bars"].iloc[0] == 60
    assert out.attrs["excluded_extended_bars"] == {"security:a": int(outside.sum())}


def test_delayed_input_availability_receipt_and_invalid_grid(calendar):
    raw = intraday(calendar, sessions=["2026-03-06"])
    raw["available_at"] = raw.timestamp + pd.Timedelta("1min")
    raw.loc[0, "available_at"] = pd.Timestamp("2026-03-09T15:00:00Z")
    out = aggregate_session_bars(raw, schedule=calendar, availability_basis="event_time")
    assert out.available_at.iloc[0] == pd.Timestamp("2026-03-09T15:00:00Z")
    receipt = aggregate_session_bars(
        raw.drop(columns="available_at"), schedule=calendar, ingested_at="2026-10-12T10:00:00Z"
    )
    assert receipt.available_at.iloc[0] == pd.Timestamp("2026-10-12T10:00:00Z")
    assert not receipt.point_in_time.any()
    off_grid = raw.drop(columns="available_at").copy()
    off_grid.loc[0, "timestamp"] += pd.Timedelta("1s")
    with pytest.raises(ValueError, match="session grid"):
        aggregate_session_bars(off_grid, schedule=calendar, availability_basis="event_time")
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_session_bars(
            pd.concat([raw, raw.iloc[:1]]), schedule=calendar, availability_basis="event_time"
        )


def test_empty_daily_and_aggregate_keep_explicit_schema(calendar):
    empty = daily_rows().iloc[:0]
    assert normalize_daily_bars(empty, schedule=calendar).empty
    assert aggregate_session_bars(
        empty.assign(timestamp=pd.Series(dtype="datetime64[ns, UTC]")), schedule=calendar
    ).empty


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_daily_loader_preserves_ids_provenance_missingness_and_nullable_masks(
    calendar, tmp_path, suffix
):
    raw = daily_rows()
    raw["instrument_id"] = "001"
    raw["entity_id"] = "002"
    raw["revision_id"] = "003"
    raw["quality.session_complete"] = [True, pd.NA]
    raw.loc[1, "volume"] = np.nan
    expected = normalize_daily_bars(raw, schedule=calendar, ingested_at="2026-10-12T10:00:00Z")
    path = tmp_path / f"daily.{suffix}"
    if suffix == "csv":
        expected.to_csv(path, index=False)
    else:
        expected.to_parquet(path, index=False)
    actual = load_daily_bars(path, schedule=calendar)
    assert_frame_equal(actual, expected)
    assert actual.instrument_id.eq("001").all()
    assert actual.entity_id.eq("002").all()
    assert actual.revision_id.eq("003").all()


def test_daily_loader_rejects_unknown_boolean_strings_and_suffix(calendar, tmp_path):
    raw = daily_rows().iloc[:1].assign(point_in_time="maybe")
    path = tmp_path / "daily.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="booleans"):
        load_daily_bars(path, schedule=calendar, availability_basis="event_time")
    with pytest.raises(ValueError, match="csv or .parquet"):
        load_daily_bars(tmp_path / "daily.json", schedule=calendar)
