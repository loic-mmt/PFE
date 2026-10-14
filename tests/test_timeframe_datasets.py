"""Selected arbitrary timeframes reach datasets only after close and receipt."""

import numpy as np
import pandas as pd
import pytest

from pfe.datasets.timeframes import build_timeframe_dataset, build_timeframe_observations


def bars(count=30, frequency="1min", start="2026-01-01T10:00:00Z"):
    index = pd.date_range(start, periods=count, freq=frequency)
    close = 100.0 + np.arange(count)
    frame = pd.DataFrame(
        dict(open=close, high=close + 1, low=close - 1, close=close, volume=1.0), index=index
    )
    frame["available_at"] = index + pd.Timedelta(frequency)
    return frame


def specs(frequencies=("1min", "5min")):
    return [
        dict(
            frequency=frequency,
            names=["close", "returns.simple"],
            units={"close": "USD", "returns.simple": "fraction"},
        )
        for frequency in frequencies
    ]


def test_multitimeframe_windows_and_labels_use_decision_grid_not_days():
    frame = bars()
    result = build_timeframe_dataset(
        {"A": frame},
        specs(),
        decisions=frame.index + pd.Timedelta("1min"),
        lookback=3,
        source_frequency="1min",
        source_id="fixture",
        revision_id="1",
        label_feature="tf[1min].close",
        label_horizon=2,
    )
    assert result.panel.values.shape == (30, 1, 4)
    assert not result.panel.mask[:4, 0, 2].any()
    assert result.panel.values[4, 0, 2] == 104
    assert result.panel.values[5, 0, 2] == 104
    assert result.sequences.y[0] == pytest.approx(104 / 102 - 1)
    assert not result.sequences.y_mask[-2:].any()


def test_late_receipt_future_invariance_and_incomplete_target_mask():
    frame = bars()
    frame.loc[frame.index[1], "available_at"] = frame.index[10]
    bundle = build_timeframe_observations(
        {"A": frame},
        specs(("5min",)),
        source_frequency="1min",
        source_id="fixture",
        revision_id="1",
    )
    points = bundle.observations
    assert points.available_at.iloc[0] >= frame.index[10]
    damaged = frame.drop(frame.index[2])
    bundle = build_timeframe_observations(
        {"A": damaged},
        specs(("5min",)),
        source_frequency="1min",
        source_id="fixture",
        revision_id="1",
    )
    first = bundle.observations.loc[
        bundle.observations.observation_date.eq("2026-01-01T10:05:00+00:00")
    ]
    assert first.value.isna().all() and not first["quality.feature_eligible"].any()
    options = dict(
        decisions=frame.index[:10] + pd.Timedelta("1min"),
        lookback=2,
        source_frequency="1min",
        source_id="fixture",
        revision_id="1",
    )
    one = build_timeframe_dataset({"A": frame}, specs(), **options)
    changed = frame.copy()
    changed.loc[frame.index[10] :, ["open", "high", "low", "close"]] *= 2
    two = build_timeframe_dataset({"A": changed}, specs(), **options)
    np.testing.assert_array_equal(one.panel.values, two.panel.values)


@pytest.mark.parametrize("source,target", [("1s", "7s"), ("5min", "1h"), ("1h", "1D")])
def test_arbitrary_fixed_source_and_target(source, target):
    count = int(pd.Timedelta(target) / pd.Timedelta(source)) * 2
    frame = bars(count, source, start="2026-01-01T00:00:00Z")
    bundle = build_timeframe_observations(
        {"A": frame},
        specs((target,)),
        source_frequency=source,
        source_id="fixture",
        revision_id="1",
    )
    assert len(bundle.observations) == 4


def test_explicit_source_availability_and_no_upsampling():
    frame = bars().drop(columns="available_at")
    options = dict(source_frequency="1min", source_id="fixture", revision_id="1")
    with pytest.raises(ValueError, match="available_at"):
        build_timeframe_observations({"A": frame}, specs(), **options)
    bundle = build_timeframe_observations(
        {"A": frame}, specs(), availability_policy="bar_close", **options
    )
    assert not bundle.observations.point_in_time.any()
    with pytest.raises(ValueError, match="integer multiple"):
        build_timeframe_observations(
            {"A": frame}, specs(("1s",)), availability_policy="bar_close", **options
        )


def test_session_daily_and_intraday_share_a_dataset_with_real_closes():
    from pfe.data.sessions import normalize_session_schedule

    friday = bars(3, start="2026-03-06T14:30:00Z")
    monday = bars(2, start="2026-03-09T13:30:00Z")
    frame = pd.concat([friday, monday])
    schedule = normalize_session_schedule(
        pd.DataFrame(
            dict(
                session_date=["2026-03-06", "2026-03-09"],
                open_at=[friday.index[0], monday.index[0]],
                close_at=[
                    friday.index[-1] + pd.Timedelta("1min"),
                    monday.index[-1] + pd.Timedelta("1min"),
                ],
            )
        )
    )
    daily = specs(("daily",))[0] | dict(mode="session")
    result = build_timeframe_dataset(
        {"A": frame},
        [specs(("1min",))[0], daily],
        decisions=pd.DatetimeIndex(schedule.close_at),
        lookback=1,
        source_frequency="1min",
        source_id="fixture",
        revision_id="1",
        schedules={"A": schedule},
        session_metadata={
            "A": dict(
                instrument_id="stock:A",
                currency="USD",
                price_adjustment="raw",
                volume_adjustment="raw",
            )
        },
    )
    assert result.panel.values[:, 0, 2].tolist() == [102, 101]
    assert result.panel.values[1, 0, 3] == pytest.approx(101 / 102 - 1)
