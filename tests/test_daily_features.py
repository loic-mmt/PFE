"""Session-count indicators, quality masks and per-instrument availability."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe.data.daily import normalize_daily_bars
from pfe.data.sessions import normalize_session_schedule
from pfe.features.daily import compute_daily_features


@pytest.fixture
def calendar():
    days = ["2026-03-06", "2026-03-09", "2026-03-10", "2026-03-11"]
    return normalize_session_schedule(
        pd.DataFrame(
            {
                "session_date": days,
                "open_at": [
                    f"{day}T{'14' if index == 0 else '13'}:30:00Z" for index, day in enumerate(days)
                ],
                "close_at": [
                    f"{day}T{'21' if index == 0 else '20'}:00:00Z" for index, day in enumerate(days)
                ],
            }
        )
    )


def rows(calendar, prices=(100, 110, 121, 133.1), instrument="security:a"):
    return normalize_daily_bars(
        pd.DataFrame(
            {
                "session_date": calendar.index[: len(prices)],
                "instrument_id": instrument,
                "open": np.asarray(prices) - 0.5,
                "high": np.asarray(prices) + 1,
                "low": np.asarray(prices) - 1,
                "close": prices,
                "volume": 1000.0,
            }
        ),
        schedule=calendar,
        availability_basis="event_time",
    )


def test_weekend_dst_and_holiday_count_sessions_without_reset(calendar):
    out = compute_daily_features(
        rows(calendar), schedule=calendar, windows=(2,), names=("returns.simple", "trend.sma_w2")
    )
    assert out["returns.simple"].iloc[1:].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert out["trend.sma_w2"].iloc[1] == 105
    assert out.timestamp.tolist() == calendar.open_at.tolist()
    assert out.available_at.tolist() == calendar.close_at.tolist()
    assert out.attrs["window_units"] == "exchange_sessions"
    # Injected actual calendar omits a holiday; consecutive sessions remain adjacent.
    holiday = calendar.iloc[[0, 2, 3]]
    selected = rows(calendar).iloc[[0, 2, 3]]
    known = compute_daily_features(selected, schedule=holiday, names="returns.simple")
    assert known["returns.simple"].iloc[1] == pytest.approx(0.21)


def test_missing_expected_session_or_ineligible_row_resets_only_that_instrument(calendar):
    raw = rows(calendar).iloc[[0, 1, 3]]
    out = compute_daily_features(raw, schedule=calendar, names="returns.simple")
    assert out["returns.simple"].iloc[1] == pytest.approx(0.1)
    assert np.isnan(out["returns.simple"].iloc[2])
    allowed = compute_daily_features(
        raw, schedule=calendar, names="returns.simple", gap_policy="allow"
    )
    assert allowed["returns.simple"].iloc[2] == pytest.approx(0.21)
    raw = rows(calendar)
    raw.loc[2, ["open", "high", "low", "close", "volume"]] = np.nan
    unknown = compute_daily_features(raw, schedule=calendar, names="returns.simple")
    assert unknown["returns.simple"].iloc[2:].isna().all()
    assert unknown.loc[2, "close"] != 110


def test_multi_instrument_windows_independent_and_native_scope_preserved(calendar):
    first = rows(calendar, (100, 110))
    second = rows(calendar, (1000, 900), instrument="security:b")
    out = compute_daily_features(
        pd.concat([second, first]), schedule=calendar, names="returns.simple"
    )
    assert out.loc[out.instrument_id == "security:a", "returns.simple"].iloc[-1] == pytest.approx(
        0.1
    )
    assert out.loc[out.instrument_id == "security:b", "returns.simple"].iloc[-1] == pytest.approx(
        -0.1
    )
    assert out.entity_id.isna().all()
    assert out.session_scope.eq("unknown").all()


def test_delayed_old_bar_cannot_be_used_before_actual_availability(calendar):
    raw = rows(calendar)
    late = pd.Timestamp("2026-03-12T10:00:00Z")
    raw.loc[0, "available_at"] = late
    out = compute_daily_features(raw, schedule=calendar, names="returns.simple")
    assert out.available_at.eq(late).all()
    assert out.bar_available_at.iloc[1] == calendar.close_at.iloc[1]
    historical = compute_daily_features(
        raw, schedule=calendar, names="returns.simple", decision_at="2026-03-09T20:00:00Z"
    )
    assert historical.session_date.tolist() == ["2026-03-09"]
    assert historical["returns.simple"].isna().all()


def test_future_rows_cannot_change_existing_daily_features(calendar):
    raw = rows(calendar)
    options = dict(schedule=calendar, windows=(2,), groups=("returns", "trend"))
    prefix = compute_daily_features(raw.iloc[:2], **options)
    full = compute_daily_features(raw, **options)
    assert_frame_equal(prefix, full.iloc[:2].reset_index(drop=True))


def test_partial_bars_masked_until_explicit_opt_in_and_all_missing_schema(calendar):
    raw = rows(calendar, (100, 110))
    raw["quality.session_complete"] = False
    out = compute_daily_features(raw, schedule=calendar, names="returns.simple")
    assert out["returns.simple"].isna().all()
    accepted = compute_daily_features(
        raw, schedule=calendar, names="returns.simple", allow_partial=True
    )
    assert accepted["returns.simple"].iloc[-1] == pytest.approx(0.1)


def test_no_synthetic_calendar_features_and_no_implicit_convention_change(calendar):
    raw = rows(calendar)
    for options in (
        dict(groups=("calendar",)),
        dict(names="session.progress"),
        dict(names="calendar.weekday"),
    ):
        with pytest.raises(ValueError, match="intraday"):
            compute_daily_features(raw, schedule=calendar, **options)
    raw.loc[1, "price_adjustment"] = "split_adjusted"
    with pytest.raises(ValueError, match="cannot change"):
        compute_daily_features(raw, schedule=calendar, names="returns.simple")


def test_masks_distinguish_warmup_and_ages_use_real_session_times(calendar):
    raw = rows(calendar, (100, 110))
    out = compute_daily_features(raw, schedule=calendar, names=("returns.simple", "trend.sma_w2"))
    assert out["mask.returns.simple"].tolist() == [False, True]
    assert out["mask.trend.sma_w2"].tolist() == [False, True]
    assert out["quality.age_seconds"].tolist() == [0, 0]
    assert out["quality.receipt_age_seconds"].isna().all()
    asof = compute_daily_features(
        raw, schedule=calendar, names="returns.simple", decision_at="2026-03-09T21:00:00Z"
    )
    assert asof["quality.age_seconds"].tolist() == [259200, 3600]
    without = compute_daily_features(
        raw, schedule=calendar, names="returns.simple", include_masks=False
    )
    assert not any(name.startswith("mask.") for name in without.columns)


def test_availability_lag_and_future_backfill_receipt_age(calendar):
    raw = rows(calendar, (100, 110))
    raw["available_at"] += pd.Timedelta("2min")
    raw["ingested_at"] = pd.Timestamp("2026-10-12T00:00:00Z")
    raw["availability_basis"] = "reconstructed"
    out = compute_daily_features(raw, schedule=calendar, names="returns.simple")
    assert out["quality.availability_lag_seconds"].tolist() == [120, 120]
    assert out["quality.age_seconds"].tolist() == [120, 120]
    assert out["quality.receipt_age_seconds"].isna().all()
