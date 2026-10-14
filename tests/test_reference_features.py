"""Known periods, exact benchmark horizons, quotation direction and exposure weights."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe.data.daily import normalize_daily_bars
from pfe.data.reference_assets import normalize_reference_assets, reference_assets_from_daily
from pfe.data.sessions import exchange_schedule
from pfe.features.reference import compute_reference_features


def row(instrument, day, value, **changes):
    return (
        dict(
            entity_id=f"factor:{instrument}",
            instrument_id=instrument,
            symbol=instrument,
            asset_class="benchmark",
            source_id="local",
            revision_id="v1",
            schema_version="1",
            metric="close",
            unit="USD/share",
            currency="USD",
            calendar="XNYS",
            frequency="daily",
            price_adjustment="split_adjusted",
            observation_date=f"2026-01-{day:02}",
            observed_at=f"2026-01-{day:02}T21:00:00Z",
            value=value,
            published_at=None,
            available_at=f"2026-01-{day:02}T21:01:00Z",
        )
        | changes
    )


def decisions(*stamps):
    return pd.DatetimeIndex(stamps)


def test_availability_boundary_and_late_old_period_revision_are_causal():
    original = [row("SPY", 5, 100), row("SPY", 6, 110)]
    old_revision = row("SPY", 5, 105, revision_id="v2", available_at="2026-01-08T21:00:00Z")
    future = row("SPY", 9, 1000)
    now = decisions("2026-01-06T21:00:59Z", "2026-01-06T21:01:00Z")
    options = dict(windows=(2,), groups=("levels", "returns"))
    before = compute_reference_features(now, references=pd.DataFrame(original), **options)
    full = compute_reference_features(
        now, references=pd.DataFrame([*original, old_revision, future]), **options
    )
    assert_frame_equal(before, full)
    assert before["reference[SPY].level"].tolist() == [100, 110]
    assert before["reference[SPY].return"].iloc[1] == pytest.approx(0.1)
    revised = compute_reference_features(
        decisions("2026-01-08T21:00:00Z"),
        references=pd.DataFrame([*original, old_revision]),
        **options,
    )
    assert revised["reference[SPY].level"].iloc[0] == 110
    assert revised["reference[SPY].return"].iloc[0] == pytest.approx(110 / 105 - 1)


def test_simultaneous_periods_are_a_single_visible_batch_and_sparse_windows():
    simultaneous = pd.DataFrame([row("SPY", 5, 100), row("SPY", 6, 110)])
    simultaneous["available_at"] = "2026-01-07T21:00:00Z"
    out = compute_reference_features(
        decisions("2026-01-07T20:59:59Z", "2026-01-07T21:00:00Z", "2026-01-07T21:05:00Z"),
        references=simultaneous,
        windows=(2,),
        names=("reference[SPY].mean_w2",),
    )
    assert np.isnan(out["reference[SPY].mean_w2"].iloc[0])
    assert out["reference[SPY].mean_w2"].iloc[1:].tolist() == [105, 105]
    assert out["mask.reference[SPY].mean_w2"].tolist() == [False, True, True]


def test_benchmark_beta_corr_and_relative_strength_use_matched_horizons():
    reference = pd.DataFrame([row("SPY", 5, 100), row("SPY", 6, 110), row("SPY", 7, 121)])
    # Nonconstant benchmark returns 10% and 20%, target exactly twice these.
    reference.loc[2, "value"] = 132
    target = pd.DataFrame([row("STOCK", 5, 100), row("STOCK", 6, 120), row("STOCK", 7, 168)])
    out = compute_reference_features(
        decisions("2026-01-07T21:01:00Z"),
        references=reference,
        target=target,
        windows=(2,),
        groups=("risk", "relative"),
    )
    assert out["reference[SPY].sample_count_w2"].iloc[0] == 2
    assert out["reference[SPY].beta_w2"].iloc[0] == pytest.approx(2)
    assert out["reference[SPY].corr_w2"].iloc[0] == pytest.approx(1)
    assert out["reference[SPY].relative_strength_w2"].iloc[0] == pytest.approx(1.68 / 1.32 - 1)
    reference["observed_at"] = pd.to_datetime(reference.observed_at) - pd.Timedelta("1h")
    unmatched = compute_reference_features(
        decisions("2026-01-07T21:01:00Z"),
        references=reference,
        target=target,
        windows=(2,),
        groups=("risk",),
    )
    assert unmatched["reference[SPY].sample_count_w2"].iloc[0] == 0
    assert np.isnan(unmatched["reference[SPY].beta_w2"].iloc[0])


def test_fx_direction_and_energy_exposure_are_explicit_and_missing_stays_missing():
    refs = pd.DataFrame(
        [
            row(
                "USDJPY",
                5,
                100,
                asset_class="fx",
                currency="JPY",
                unit="JPY/USD",
                base_currency="USD",
                quote_currency="JPY",
                price_adjustment="raw",
            ),
            row(
                "USDJPY",
                6,
                110,
                asset_class="fx",
                currency="JPY",
                unit="JPY/USD",
                base_currency="USD",
                quote_currency="JPY",
                price_adjustment="raw",
            ),
            row("OIL", 5, 50, asset_class="energy", unit="USD/barrel", price_adjustment="raw"),
            row("OIL", 6, 55, asset_class="energy", unit="USD/barrel", price_adjustment="raw"),
        ]
    )
    out = compute_reference_features(
        decisions("2026-01-05T21:01:00Z", "2026-01-06T21:01:00Z"),
        references=refs,
        windows=(2,),
        groups=("exposure",),
        fx_exposures={"USDJPY": 0.5},
        energy_exposures={"OIL": -0.25},
        reporting_currency="USD",
    )
    assert np.isnan(out["reference[USDJPY].fx_effect"].iloc[0])
    assert out["reference[USDJPY].fx_effect"].iloc[1] == pytest.approx((100 / 110 - 1) * 0.5)
    assert out["reference[OIL].energy_effect"].iloc[1] == pytest.approx(-0.025)


def test_expiry_and_unknown_units_do_not_fabricate_values():
    refs = pd.DataFrame([row("SPY", 5, 100), row("SPY", 6, 110)])
    out = compute_reference_features(
        decisions("2026-01-07T21:01:00Z"),
        references=refs,
        windows=(2,),
        max_age="1h",
        groups=("levels", "returns"),
    )
    assert np.isnan(out["reference[SPY].level"].iloc[0])
    assert out["reference[SPY].age_seconds"].iloc[0] == 86400
    mixed = pd.concat([refs, pd.DataFrame([row("SPY", 7, 120, unit="EUR/share", currency="EUR")])])
    with pytest.raises(ValueError, match="Reconcile reference"):
        normalize_reference_assets(mixed)
    fx = pd.DataFrame([row("USDJPY", 5, 110, asset_class="fx", currency="JPY")])
    with pytest.raises(ValueError, match="base_currency"):
        normalize_reference_assets(fx)


def test_missing_period_is_not_forward_filled_into_return():
    refs = pd.DataFrame([row("SPY", 5, 100), row("SPY", 6, None), row("SPY", 7, 110)])
    out = compute_reference_features(
        decisions("2026-01-07T21:01:00Z"), references=refs, windows=(2,), groups=("returns",)
    )
    assert out["reference[SPY].return"].isna().all()


def test_daily_adapter_preserves_actual_market_close_and_masks():
    schedule = exchange_schedule("2026-03-06", "2026-03-09")
    raw = pd.DataFrame(
        dict(
            session_date=schedule.index,
            open=[100, 105],
            high=[101, 106],
            low=[99, 104],
            close=[100, 105],
            volume=1000,
        )
    )
    daily = normalize_daily_bars(
        raw,
        schedule=schedule,
        instrument_id="SPY",
        entity_id="fund:spy",
        source_id="local",
        currency="USD",
        availability_basis="event_time",
    )
    adapted = reference_assets_from_daily(
        daily, asset_class="benchmark", unit="USD/share", symbol="SPY"
    )
    assert adapted.observed_at.tolist() == schedule.close_at.tolist()
    assert adapted.available_at.tolist() == schedule.close_at.tolist()
    assert adapted.calendar.eq("XNYS").all()


def test_ambiguous_closes_and_currency_contradictions_are_rejected():
    ambiguous = pd.DataFrame([row("SPY", 5, 100), row("SPY", 6, 110)])
    ambiguous.loc[1, "observed_at"] = ambiguous.loc[0, "observed_at"]
    with pytest.raises(ValueError, match="different economic periods"):
        normalize_reference_assets(ambiguous)
    with pytest.raises(ValueError, match="Monetary unit"):
        normalize_reference_assets(pd.DataFrame([row("SPY", 5, 100, unit="EUR/share")]))


def test_context_bridge_retains_frequency_and_has_no_invented_market_close():
    from pfe import feature_catalog, reference_assets_from_observations
    from pfe.data.world_bank import parse_pink_sheet

    monthly = parse_pink_sheet(
        pd.DataFrame({"Month": ["2026-01", "2026-02"], "Oil": [50, 55]}),
        period_column="Month",
        column_specs={"Oil": {"entity_id": "oil", "unit": "USD/barrel", "currency": "USD"}},
        received_at="2026-03-10T12:00:00Z",
        revision_id="fixture-file-1",
    )
    references = reference_assets_from_observations(
        monthly,
        instrument_id="OIL",
        symbol="Oil",
        asset_class="energy",
    )
    assert references.frequency.tolist() == ["monthly", "monthly"]
    assert references.observed_at.isna().all()
    out = compute_reference_features(
        decisions("2026-03-10T11:59:59Z", "2026-03-10T12:00:00Z"),
        references=references,
        windows=(2,),
        groups=("returns", "risk"),
    )
    assert np.isnan(out["reference[OIL].return"].iloc[0])
    assert out["reference[OIL].return"].iloc[1] == pytest.approx(0.1)
    assert out["reference[OIL].beta_w2"].isna().all()
    assert feature_catalog(out).availability.eq("explicit decision_at").all()
    with pytest.raises(ValueError, match="Existing frequency differs"):
        reference_assets_from_observations(
            monthly,
            instrument_id="OIL",
            symbol="Oil",
            asset_class="energy",
            frequency="daily",
        )


def test_benchmark_expiry_uses_the_matched_endpoint_not_a_newer_unmatched_price():
    capture = "2026-01-10T21:01:00Z"
    references = pd.DataFrame(
        [
            row("SPY", 5, 100),
            row("SPY", 6, 110),
            row("SPY", 7, 132),
            row("SPY", 8, 140, available_at=capture),
        ]
    )
    target = pd.DataFrame(
        [
            row("STOCK", 5, 100, available_at=capture),
            row("STOCK", 6, 120, available_at=capture),
            row("STOCK", 7, 168, available_at=capture),
        ]
    )
    options = dict(references=references, target=target, windows=(2,), groups=("risk", "relative"))
    known = compute_reference_features(decisions(capture), **options)
    assert known["reference[SPY].beta_w2"].iloc[0] == pytest.approx(2)
    expired = compute_reference_features(decisions(capture), max_age="1h", **options)
    assert expired["reference[SPY].beta_w2"].isna().all()
    assert expired["reference[SPY].relative_strength_w2"].isna().all()
    assert not expired["mask.reference[SPY].beta_w2"].any()
