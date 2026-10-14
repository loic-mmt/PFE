"""Compatible actual periods, additive YTD derivation and causal revisions."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe.features.fundamentals import compute_fundamental_features


def fact(
    role,
    value,
    *,
    start="2025-01-01",
    end="2025-03-31",
    available="2025-05-01T20:00:00Z",
    kind="quarter",
    **changes,
):
    concept = role
    earnings_basis = changes.get("eps_basis", "diluted") if role == "eps" else "na"
    row = dict(
        entity_id="company:a",
        source_id="sec",
        observation_date=end,
        metric=(
            f"{concept}:{start}:{changes.get('basis', 'us_gaap')}:"
            f"{changes.get('unit', 'USD')}:{earnings_basis}"
        ),
        value=value,
        unit="USD",
        revision_id=f"{role}:{start}:{available}",
        schema_version="1",
        published_at=available,
        available_at=available,
        financial_metric=role,
        taxonomy="us-gaap",
        concept=concept,
        period_start=start,
        period_end=end,
        period_kind=kind,
        basis="us_gaap",
        eps_basis="diluted" if role == "eps" else None,
        entity_scope="entity",
        currency="USD",
        statement_type="reported_actual",
    )
    return row | changes


def decisions(*times):
    return pd.DatetimeIndex(times)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT])
@pytest.mark.parametrize("role", ["revenue", "eps", "assets"])
def test_nullable_period_start_preserves_known_level_and_missing_growth(missing, role):
    facts = pd.DataFrame(
        [fact(role, 100, start=missing, kind="instant" if role == "assets" else "quarter")]
    )
    out = compute_fundamental_features(decisions("2025-05-01T21:00:00Z"), facts=facts)
    assert out[f"fundamentals.{role}"].iloc[0] == 100
    if role in ("revenue", "eps"):
        assert out[f"fundamentals.{role}_growth_yoy"].isna().all()


def test_actual_levels_ratios_fcf_and_metric_ages_hand_calculated():
    rows = [
        fact("revenue", 100),
        fact("gross_profit", 40),
        fact("operating_income", 20),
        fact("net_income", 15),
        fact("operating_cash_flow", 30),
        fact("capex", 10),
        fact("assets", 200, start=None, kind="instant"),
        fact("liabilities", 100, start=None, kind="instant"),
        fact("equity", 100, start=None, kind="instant"),
    ]
    out = compute_fundamental_features(
        decisions("2025-05-01T20:00:00Z", "2025-05-02T20:00:00Z"), facts=pd.DataFrame(rows)
    )
    assert out["fundamentals.gross_margin"].tolist() == [0.4, 0.4]
    assert out["fundamentals.operating_margin"].iloc[0] == 0.2
    assert out["fundamentals.capex_intensity"].iloc[0] == 0.1
    assert out["fundamentals.liabilities_to_assets"].iloc[0] == 0.5
    assert out["fundamentals.free_cash_flow"].iloc[0] == 20
    assert out["fundamentals.free_cash_flow_derived"].iloc[0] == 1
    assert out["fundamentals.revenue_age_seconds"].iloc[1] == 86400
    assert out["fundamentals.debt_to_assets"].isna().all()


def test_future_revision_does_not_change_historical_decision_or_yoy():
    previous = fact(
        "revenue", 80, start="2024-01-01", end="2024-03-31", available="2024-05-01T20:00:00Z"
    )
    current = fact("revenue", 100)
    revised = previous | dict(
        value=90,
        revision_id="revision",
        published_at="2025-06-01T20:00:00Z",
        available_at="2025-06-01T20:00:00Z",
    )
    index = decisions("2025-05-01T20:00:00Z")
    prefix = compute_fundamental_features(index, facts=pd.DataFrame([previous, current]))
    full = compute_fundamental_features(index, facts=pd.DataFrame([previous, current, revised]))
    assert_frame_equal(prefix, full)
    assert full["fundamentals.revenue_growth_yoy"].iloc[0] == 0.25
    later = compute_fundamental_features(
        decisions("2025-06-01T20:00:00Z"), facts=pd.DataFrame([previous, current, revised])
    )
    assert later["fundamentals.revenue_growth_yoy"].iloc[0] == pytest.approx(100 / 90 - 1)


def test_additive_quarter_from_ytd_and_eps_not_additive():
    rows = [
        fact(
            "revenue",
            100,
            start="2025-01-01",
            end="2025-06-30",
            available="2025-08-01T20:00:00Z",
            kind="ytd",
        ),
        fact(
            "revenue",
            160,
            start="2025-01-01",
            end="2025-09-30",
            available="2025-11-01T20:00:00Z",
            kind="ytd",
        ),
        fact(
            "eps",
            1,
            start="2025-01-01",
            end="2025-06-30",
            available="2025-08-01T20:00:00Z",
            kind="ytd",
            unit="USD/shares",
        ),
        fact(
            "eps",
            1.8,
            start="2025-01-01",
            end="2025-09-30",
            available="2025-11-01T20:00:00Z",
            kind="ytd",
            unit="USD/shares",
        ),
    ]
    out = compute_fundamental_features(decisions("2025-11-01T20:00:00Z"), facts=pd.DataFrame(rows))
    assert out["fundamentals.revenue"].iloc[0] == 60
    assert out["fundamentals.derived_quarter"].iloc[0] == 1
    assert out["fundamentals.eps"].isna().all()
    delayed = rows[0] | dict(
        published_at="2025-12-01T20:00:00Z", available_at="2025-12-01T20:00:00Z"
    )
    early = compute_fundamental_features(
        decisions("2025-11-01T20:00:00Z"), facts=pd.DataFrame([delayed, *rows[1:]])
    )
    assert early["fundamentals.revenue"].isna().all()


def test_fourth_quarter_annual_minus_nine_month_ytd_with_masks():
    rows = [
        fact("revenue", 300, end="2025-09-30", available="2025-11-01T20:00:00Z", kind="ytd"),
        fact("revenue", 450, end="2025-12-31", available="2026-02-01T20:00:00Z", kind="annual"),
        fact(
            "eps",
            4,
            end="2025-09-30",
            available="2025-11-01T20:00:00Z",
            kind="ytd",
            unit="USD/shares",
        ),
        fact(
            "eps",
            6,
            end="2025-12-31",
            available="2026-02-01T20:00:00Z",
            kind="annual",
            unit="USD/shares",
        ),
    ]
    out = compute_fundamental_features(decisions("2026-02-01T20:00:00Z"), facts=pd.DataFrame(rows))
    assert out["fundamentals.revenue"].iloc[0] == 150
    assert out["fundamentals.eps"].isna().all()
    assert out["mask.fundamentals.revenue"].iloc[0]
    assert not out["mask.fundamentals.eps"].iloc[0]
    assert out.available_at.equals(out.decision_at)


def test_max_age_does_not_resurrect_an_old_period_revised_recently():
    old = fact("revenue", 100)
    current = fact(
        "revenue", 120, start="2025-04-01", end="2025-06-30", available="2025-07-01T20:00:00Z"
    )
    revision = old | dict(
        value=105,
        revision_id="old-revised",
        published_at="2025-07-20T20:00:00Z",
        available_at="2025-07-20T20:00:00Z",
    )
    out = compute_fundamental_features(
        decisions("2025-07-22T20:00:00Z"),
        facts=pd.DataFrame([old, current, revision]),
        max_age="7D",
    )
    assert out["fundamentals.revenue"].isna().all()
    assert not out["mask.fundamentals.revenue"].any()


def test_current_age_mask_preserves_historical_yoy_comparatives():
    rows = [
        fact("revenue", 80, start="2024-01-01", end="2024-03-31", available="2024-05-01T20:00:00Z"),
        fact("revenue", 100),
    ]
    out = compute_fundamental_features(
        decisions("2025-05-02T20:00:00Z"), facts=pd.DataFrame(rows), max_age="7D"
    )
    assert out["fundamentals.revenue_growth_yoy"].iloc[0] == 0.25


def test_no_mixing_period_currency_accounting_basis_or_eps_variant():
    rows = [
        fact("revenue", 100),
        fact("gross_profit", 30, start="2024-01-01", kind="annual"),
        fact("operating_income", 20, unit="EUR", currency="EUR"),
        fact("net_income", 10, basis="non_gaap"),
        fact("eps", 2, eps_basis="basic", unit="USD/shares"),
        fact("eps", 1.5, eps_basis="diluted", unit="USD/shares"),
    ]
    out = compute_fundamental_features(decisions("2025-05-01T20:00:00Z"), facts=pd.DataFrame(rows))
    assert out["fundamentals.gross_margin"].isna().all()
    assert out["fundamentals.operating_margin"].isna().all()
    assert out["fundamentals.net_income"].isna().all()
    assert out["fundamentals.eps"].iloc[0] == 1.5


def test_ambiguous_revenue_tags_need_explicit_concept_and_guidance_excluded():
    one = fact("revenue", 100)
    other = fact("revenue", 105, metric="different-concept", concept="other_revenue")
    guide = fact(
        "revenue", 150, start="2025-04-01", end="2025-06-30", statement_type="management_guidance"
    )
    rows = pd.DataFrame([one, other, guide])
    out = compute_fundamental_features(decisions("2025-05-01T20:00:00Z"), facts=rows)
    assert out["fundamentals.revenue"].isna().all()
    assert out["fundamentals.quality.ambiguous"].iloc[0] == 1
    selected = compute_fundamental_features(
        decisions("2025-05-01T20:00:00Z"), facts=rows, concepts={"revenue": "us-gaap:revenue"}
    )
    assert selected["fundamentals.revenue"].iloc[0] == 100


def test_entity_decisions_zero_denominator_missing_and_age_mask():
    rows = pd.DataFrame(
        [fact("revenue", 0), fact("capex", 2), fact("revenue", 20, entity_id="company:b")]
    )
    times = pd.DataFrame(
        dict(entity_id=["company:a", "company:b"], decision_at=["2025-05-01T20:00:00Z"] * 2)
    )
    out = compute_fundamental_features(times, facts=rows)
    assert out["fundamentals.revenue"].tolist() == [0, 20]
    assert out["fundamentals.capex_intensity"].isna().all()
    stale = compute_fundamental_features(
        decisions("2025-06-01T20:00:00Z"), facts=rows, entity_id="company:a", max_age="1D"
    )
    assert stale["fundamentals.revenue"].isna().all()
    with pytest.raises(ValueError, match="aware"):
        compute_fundamental_features(
            pd.DatetimeIndex(["2025-05-01"]), facts=rows, entity_id="company:a"
        )
