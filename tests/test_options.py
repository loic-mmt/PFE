"""Numerical options references, sparse availability, expiry and missing-field guards."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pfe import (
    OPTION_FEATURE_GROUPS,
    compute_option_feature,
    compute_option_features,
    load_options,
    normalize_option_candles,
    normalize_option_chain,
    normalize_option_flow,
)


def stamp(value):
    return pd.Timestamp(value, tz="UTC")


@pytest.fixture
def chain():
    rows = []
    for i, (side, strike, expiry, iv, delta) in enumerate(
        [
            ("call", 100, "2025-01-31", 0.20, 0.50),
            ("put", 100, "2025-01-31", 0.24, -0.50),
            ("call", 110, "2025-01-31", 0.22, 0.25),
            ("put", 90, "2025-01-31", 0.30, -0.25),
            ("call", 100, "2025-02-28", 0.28, 0.55),
            ("put", 100, "2025-02-28", 0.32, -0.55),
            ("call", 100, "2024-12-31", 10.0, 0.50),
        ]
    ):
        rows.append(
            dict(
                underlying="QQQ",
                ticker=f"CONTRACT{i}",
                contract_type=side,
                strike=strike,
                expiry=expiry,
                iv=iv,
                delta=delta,
                gamma=0.01,
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                underlying_price=100.0,
                snapshot_at=stamp("2025-01-14 12:00"),
                available_at=stamp("2025-01-14 12:02"),
                quote_at=stamp("2025-01-14 11:59"),
                volume=10 * (i + 1),
                premium=100 * (i + 1),
                open_interest=100.0,
            )
        )
    frame = pd.DataFrame(rows)
    frame.attrs["limit_reached"] = True
    return frame


@pytest.fixture
def bars():
    return pd.date_range("2025-01-14 12:00", periods=12, freq="min", tz="UTC")


@pytest.fixture
def flow():
    return pd.DataFrame(
        [
            dict(
                underlying="QQQ",
                ticker="CALL",
                contract_type="call",
                strike=100,
                expiry="2025-01-31",
                trade_id=1,
                trade_at=stamp("2025-01-14 12:01"),
                available_at=stamp("2025-01-14 12:02"),
                volume=10.0,
                premium=100.0,
                iv=0.2,
                delta=0.5,
            ),
            dict(
                underlying="QQQ",
                ticker="PUT",
                contract_type="put",
                strike=100,
                expiry="2025-01-31",
                trade_id=2,
                trade_at=stamp("2025-01-14 12:02"),
                available_at=stamp("2025-01-14 12:02"),
                volume=20.0,
                premium=200.0,
                iv=0.3,
                delta=-0.5,
            ),
            dict(
                underlying="QQQ",
                ticker="CALL",
                contract_type="call",
                strike=100,
                expiry="2025-01-31",
                trade_id=3,
                trade_at=stamp("2025-01-14 12:02"),
                available_at=stamp("2025-01-14 12:09"),
                volume=999.0,
                premium=999.0,
                iv=0.9,
                delta=0.1,
            ),
        ]
    )


@pytest.fixture
def candles():
    return pd.DataFrame(
        [
            dict(
                underlying="QQQ",
                ticker="CALL",
                contract_type="call",
                strike=100,
                expiry="2025-01-31",
                bar_at=stamp(f"2025-01-14 12:{minute:02}"),
                available_at=stamp(f"2025-01-14 12:{minute + 1:02}"),
                open=price,
                high=price + 0.2,
                low=price - 0.1,
                close=price + 0.1,
                volume=10.0,
                premium=100.0,
                print_count=2,
                iv=0.2,
                delta=0.5,
            )
            for minute, price in [(0, 1.0), (1, 1.1), (4, 1.5)]
        ]
    )


def test_snapshot_activity_greeks_and_expired_contract_exclusion(chain, bars):
    out = compute_option_features(bars, chain=chain)
    assert out.iloc[0].isna().all()
    assert out["options.activity.call_volume"].iloc[1] == 90
    assert out["options.activity.put_volume"].iloc[1] == 120
    assert out["options.activity.put_call_volume_ratio"].iloc[1] == pytest.approx(4 / 3)
    assert out["options.activity.volume_imbalance"].iloc[1] == pytest.approx(-1 / 7)
    assert out["options.greeks.delta_mean"].iloc[1] == pytest.approx(0.0)
    expected = np.dot([0.5, -0.5, 0.25, -0.25, 0.55, -0.55], [10, 20, 30, 40, 50, 60]) / 210
    assert out["options.greeks.delta_volume_weighted"].iloc[1] == pytest.approx(expected)
    assert out["options.quality.active_contract_count"].iloc[1] == 6
    assert out["options.quality.expired_fraction"].iloc[1] == pytest.approx(1 / 7)
    assert out["options.quality.limit_reached"].iloc[1] == 1
    assert out["options.quality.snapshot_age_seconds"].iloc[1] == pytest.approx(120)
    assert out["options.quality.oldest_trade_age_seconds"].iloc[1] == pytest.approx(180)


def test_smile_term_structure_and_expected_move_references(chain, bars):
    out = compute_option_features(bars, chain=chain, groups="volatility")
    row = out.iloc[1]
    assert row["options.volatility.atm_call_iv"] == 0.2
    assert row["options.volatility.atm_put_iv"] == 0.24
    assert row["options.volatility.atm_iv"] == pytest.approx(0.22)
    assert row["options.volatility.risk_reversal"] == pytest.approx(-0.08)
    assert row["options.volatility.butterfly"] == pytest.approx(0.04)
    assert row["options.volatility.term_spread"] == pytest.approx(0.08)
    assert row["options.volatility.term_slope"] == pytest.approx(0.08 / 28)
    assert row["options.volatility.expected_move_relative"] == pytest.approx(
        0.22 * np.sqrt(17 / 365)
    )
    assert row["options.volatility.iv_std"] == pytest.approx(
        np.std([0.2, 0.24, 0.22, 0.3, 0.28, 0.32])
    )


def test_missing_oi_quotes_and_partial_greek_data_are_not_zero(chain, bars):
    frame = chain.drop(columns="open_interest").copy()
    frame.loc[0, "delta"] = np.nan
    out = compute_option_features(bars, chain=frame)
    assert out.filter(like="options.positioning.").iloc[1].isna().all()
    assert np.isnan(out["options.greeks.delta_volume_weighted"].iloc[1])
    assert out["options.quality.open_interest_coverage"].iloc[1] == 0
    assert out["options.quality.quote_coverage"].iloc[1] == 0


def test_oi_strike_concentration_and_unsigned_gamma(chain, bars):
    out = compute_option_features(bars, chain=chain, groups="positioning")
    assert out["options.positioning.put_call_open_interest_ratio"].iloc[1] == 1
    assert out["options.positioning.oi_concentration"].iloc[1] == pytest.approx(
        (4 / 6) ** 2 + 2 * (1 / 6) ** 2
    )
    assert out["options.positioning.call_oi_peak_strike"].iloc[1] == 100
    assert out["options.positioning.unsigned_gamma_oi"].iloc[1] == 6


def test_surface_tolerances_no_extrapolation_and_zero_dte(chain, bars):
    frame = chain.copy()
    frame.loc[frame.strike == 100, "underlying_price"] = 200.0
    out = compute_option_features(bars, chain=frame, groups="volatility", delta_tolerance=0.001)
    assert np.isnan(out["options.volatility.atm_iv"].iloc[1])
    # A non-target delta cannot be substituted silently.
    frame["delta"] = 0.5
    out = compute_option_features(bars, chain=frame, groups="volatility")
    assert out["options.volatility.risk_reversal"].isna().all()
    frame = chain.iloc[:2].assign(expiry="2025-01-14")
    out = compute_option_features(bars, chain=frame, groups="volatility")
    assert out["options.volatility.expected_move_relative"].isna().all()


def test_new_day_and_max_age_invalidate_chain_but_keep_quality(chain):
    bars = pd.DatetimeIndex([stamp("2025-01-14 12:02"), stamp("2025-01-14 23:59")])
    out = compute_option_features(bars, chain=chain, max_age="1min")
    assert out["options.volatility.iv_mean"].iloc[0] > 0
    assert np.isnan(out["options.volatility.iv_mean"].iloc[1])
    assert out["options.quality.contract_count"].iloc[1] == 7


def test_late_older_snapshot_does_not_regress_current_state(chain, bars):
    older = chain.assign(
        snapshot_at=stamp("2025-01-14 11:59"), available_at=stamp("2025-01-14 12:05"), iv=9.0
    )
    newer = chain.assign(
        snapshot_at=stamp("2025-01-14 12:03"), available_at=stamp("2025-01-14 12:04"), iv=0.4
    )
    out = compute_option_feature(
        bars, "options.volatility.iv_mean", chain=pd.concat([chain, newer, older])
    )
    assert out.iloc[3:].to_numpy() == pytest.approx(0.4)


def test_flow_window_against_direct_event_and_availability_reference(flow, bars):
    out = compute_option_features(bars, flow=flow, flow_window="3min")
    for i, now in enumerate(bars + pd.Timedelta("1min")):
        known = flow.loc[
            (flow.available_at <= now)
            & (flow.trade_at <= now)
            & (flow.trade_at > now - pd.Timedelta("3min"))
        ]
        call = known.loc[known.contract_type == "call", "volume"].sum()
        put = known.loc[known.contract_type == "put", "volume"].sum()
        assert out["options.flow.call_volume"].iloc[i] == call
        assert out["options.flow.put_volume"].iloc[i] == put
        assert out["options.flow.print_count"].iloc[i] == len(known)
        assert out["options.flow.delta_contracts"].iloc[i] == pytest.approx(
            (known.delta * known.volume).sum()
        )
    assert out["options.flow.put_call_volume_ratio"].iloc[1] == 2
    assert out["options.flow.iv_volume_weighted"].iloc[1] == pytest.approx(8 / 30)
    assert out["options.flow.average_premium"].iloc[1] == 150


def test_missing_greek_and_late_delivery_never_contaminate_prior_windows(flow, bars):
    flow.loc[0, "iv"] = np.nan
    out = compute_option_feature(
        bars, "options.flow.iv_volume_weighted", flow=flow, flow_window="3min"
    )
    assert np.isnan(out.iloc[1])
    assert out.iloc[3] == 0.3  # missing-IV print has left the window
    assert out.iloc[4:].isna().all()  # late print is already outside its event-time window


def test_candles_use_close_times_and_do_not_double_count_prints(candles, flow, bars):
    prints = compute_option_features(
        bars, flow=flow, candles=candles, groups="flow", flow_window="3min"
    )
    from_prints = compute_option_features(bars, flow=flow, groups="flow", flow_window="3min")
    assert_frame_equal(prints, from_prints)
    out = compute_option_features(
        bars, flow=flow, candles=candles, groups="flow", flow_source="candles", flow_window="3min"
    )
    assert out["options.flow.print_count"].iloc[0] == 2
    assert out["options.flow.total_volume"].iloc[1] == 20
    out = compute_option_features(bars, candles=candles, groups="contract")
    assert out["options.contract[CALL].log_return"].iloc[1] == pytest.approx(np.log(1.2 / 1.1))
    assert np.isnan(out["options.contract[CALL].log_return"].iloc[4])  # gap reset
    assert out["options.contract[CALL].range_relative"].iloc[0] == pytest.approx(0.3 / 1.1)


def test_future_prefix_and_small_internal_chunks(monkeypatch, chain, flow, candles, bars):
    import pfe.features.options as module

    options = dict(chain=chain, flow=flow, candles=candles, flow_window="3min")
    full = compute_option_features(bars, **options)
    monkeypatch.setattr(module, "_CHUNK", 2)
    assert_frame_equal(full, compute_option_features(bars, **options))
    cutoff = stamp("2025-01-14 12:04")
    prefix = compute_option_features(
        bars[:4],
        chain=chain,
        flow=flow.loc[flow.available_at <= cutoff],
        candles=candles.loc[candles.available_at <= cutoff],
        flow_window="3min",
    )
    assert_frame_equal(full.iloc[:4], prefix)
    assert_frame_equal(
        full,
        pd.concat(
            [
                compute_option_features(bars[:6], **options),
                compute_option_features(bars[6:], **options),
            ]
        ),
    )
    assert "options" in repr(full)


def test_exact_selection_never_executes_other_groups(monkeypatch, chain, flow, bars):
    import pfe.features.options as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Unrequested feature family executed")

    monkeypatch.setattr(module, "_volatility", forbidden)
    monkeypatch.setattr(module, "_positioning", forbidden)
    monkeypatch.setattr(module, "_flow_values", forbidden)
    monkeypatch.setattr(module, "normalize_option_flow", forbidden)
    actual = compute_option_feature(bars, "options.activity.call_volume", chain=chain, flow=flow)
    assert actual.iloc[1] == 90


@pytest.mark.parametrize("group", OPTION_FEATURE_GROUPS)
def test_groups_match_exact_outputs(group, chain, flow, candles, bars):
    options = dict(chain=chain, flow=flow, candles=candles)
    full = compute_option_features(bars, **options)
    selected = compute_option_features(bars, groups=group, **options)
    assert len(selected.columns)
    assert_frame_equal(selected, full[selected.columns])
    for name in selected:
        assert_series_equal(compute_option_feature(bars, name, **options), selected[name])


@pytest.mark.parametrize(
    "change",
    [
        dict(strike=0),
        dict(contract_type="unknown"),
        dict(iv=-0.1),
        dict(delta=2),
        dict(volume=-1),
        dict(snapshot_at="2025-01-14"),
        dict(available_at="2025-01-01"),
    ],
)
def test_invalid_chain(change, chain):
    with pytest.raises(ValueError):
        normalize_option_chain(chain.assign(**change))


def test_duplicate_atomic_snapshot_and_candle_close_checks(chain, flow, candles):
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_option_chain(pd.concat([chain, chain.iloc[:1]]))
    bad = chain.copy()
    bad.loc[0, "available_at"] = stamp("2025-01-14 12:03")
    with pytest.raises(ValueError, match="atomic"):
        normalize_option_chain(bad)
    with pytest.raises(ValueError, match="unique"):
        normalize_option_flow(pd.concat([flow, flow.iloc[:1]]))
    with pytest.raises(ValueError, match="bar close"):
        normalize_option_candles(candles.assign(available_at=candles.bar_at))


@pytest.mark.parametrize(
    "options",
    [
        dict(min_dte=-1),
        dict(max_dte=True),
        dict(atm_tolerance=-1),
        dict(delta_target=1),
        dict(groups="unknown"),
        dict(names="unknown"),
        dict(groups="flow", names="options.flow.print_count"),
        dict(flow_source="both"),
    ],
)
def test_invalid_options(options, chain, bars):
    with pytest.raises(ValueError):
        compute_option_features(bars, chain=chain, **options)


def test_multi_instrument_requires_explicit_underlying(chain, bars):
    other = chain.assign(underlying="SPY", ticker=chain.ticker + "SPY")
    with pytest.raises(ValueError, match="underlying"):
        compute_option_features(bars, chain=pd.concat([chain, other]))
    qqq = compute_option_features(bars, chain=chain)
    selected = compute_option_features(bars, chain=pd.concat([chain, other]), underlying="qqq")
    assert_frame_equal(qqq, selected)


def test_lse_loader_receipt_and_explicit_event_time_assumption(tmp_path):
    identity = dict(
        underlying="QQQ",
        ticker="CALL",
        contract_type="call",
        strike=100,
        expiry="2025-01-31",
        underlying_price=100,
    )
    payload = dict(
        provider="LSE",
        snapshot_at="2025-01-14T12:10:00Z",
        limit=1,
        chain=[
            dict(
                **identity,
                volume_today=3,
                premium_today=300,
                iv=0.2,
                last_trade_at="2025-01-14T12:00:00Z",
                last_price=1.0,
            )
        ],
        flow=[dict(**identity, id=1, ts="2025-01-14T12:00:00Z", premium=300, volume=3, iv=0.2)],
        candles=[
            dict(
                **identity,
                minute="2025-01-14T12:00:00Z",
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=3,
                premium=300,
                print_count=1,
                iv_avg=0.2,
            )
        ],
    )
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(payload))
    data = load_options(path)
    assert data.chain.volume.iloc[0] == 3
    assert data.chain.open_interest.isna().all()
    assert data.chain.available_at.iloc[0] == data.snapshot_at
    assert data.flow.available_at.iloc[0] == data.snapshot_at
    assert data.candles.available_at.iloc[0] == data.snapshot_at
    assert not data.flow.point_in_time.any()
    historical = load_options(path, availability="event", latency="1s")
    assert historical.chain.available_at.iloc[0] == data.snapshot_at
    assert historical.flow.available_at.iloc[0] == stamp("2025-01-14 12:00:01")
    assert historical.candles.available_at.iloc[0] == stamp("2025-01-14 12:01:01")
    assert historical.limit_reached["flow"]
    with pytest.raises(ValueError, match="max_bytes"):
        load_options(path, max_bytes=10)


def test_empty_inputs():
    chain = pd.DataFrame(
        columns=[
            "underlying",
            "ticker",
            "contract_type",
            "strike",
            "expiry",
            "snapshot_at",
            "available_at",
        ]
    )
    bars = pd.DatetimeIndex([], tz="UTC")
    out = compute_option_features(bars, chain=chain)
    assert out.shape[0] == 0
    assert out.shape[1] > 0
