"""Explicit IFRS/custom contexts, fiscal dates and safe debt component replay."""

import pandas as pd
import pytest
from test_fundamentals import fact

from pfe.data.financial_mappings import apply_financial_mappings, derive_debt_totals
from pfe.features.fundamentals import compute_fundamental_features


def test_explicit_ifrs_custom_mapping_and_segment_selection():
    row = fact("revenue", 100, start="2025-01-01")
    row.update(
        taxonomy="issuer",
        concept="CloudSales",
        basis="ifrs",
        financial_metric=None,
        entity_scope="segment",
        context_json='{"segment":"cloud"}',
    )
    original = pd.DataFrame([row])
    mapped = apply_financial_mappings(
        original,
        [
            dict(
                taxonomy="issuer",
                concept="CloudSales",
                financial_metric="revenue",
                basis="ifrs",
                reference="filing:cloud_note",
                available_at="2025-05-02T00:00:00Z",
                context={"segment": "cloud"},
            )
        ],
    )
    assert mapped.available_at.iloc[0] >= pd.Timestamp("2025-05-02T00:00:00Z")
    assert mapped.source_revision_id.iloc[0] == row["revision_id"]
    result = compute_fundamental_features(
        pd.DatetimeIndex(["2025-05-03T00:00:00Z"]),
        facts=mapped,
        basis="ifrs",
        entity_scope="segment",
        context={"segment": "cloud"},
    )
    assert result["fundamentals.revenue"].iloc[0] == 100
    with pytest.raises(ValueError, match="context"):
        compute_fundamental_features(
            pd.DatetimeIndex(["2025-05-03T00:00:00Z"]),
            facts=mapped,
            basis="ifrs",
            entity_scope="segment",
        )


def test_declared_fiscal_period_does_not_use_filing_year_or_backdate_mapping():
    row = fact("revenue", 100, start="2024-10-01", end="2025-03-31")
    row["period_kind"] = "unknown"
    definition = dict(
        entity_id=row["entity_id"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        period_kind="ytd",
        fiscal_year=2025,
        label="H1",
        reference="filing:fiscal_note",
        available_at="2025-05-03T00:00:00Z",
    )
    result = apply_financial_mappings(pd.DataFrame([row]), [], periods=[definition])
    assert result.declared_fiscal_period.iloc[0] == "H1"
    assert result.period_kind.iloc[0] == "ytd" and not result["quality.period_inferred"].iloc[0]


def debt_rows():
    first = fact("long_term_debt_current", 10, start=None)
    second = fact("long_term_debt_noncurrent", 40, start=None)
    for row in (first, second):
        row["period_kind"] = "instant"
        row["source_document_id"] = "filing:one"
    return pd.DataFrame([first, second])


RECIPE = dict(
    financial_metric="long_term_debt",
    components=["long_term_debt_current", "long_term_debt_noncurrent"],
    reference="filing:disjoint_debt_note",
    available_at="2025-04-30T00:00:00Z",
    components_disjoint=True,
)


def test_disjoint_debt_recipe_withdrawal_future_invariance_and_direct_precedence():
    rows = debt_rows()
    derived = derive_debt_totals(rows, RECIPE)
    assert derived.loc[derived.taxonomy.eq("derived"), "value"].iloc[-1] == 50
    withdrawal = rows.iloc[:1].copy()
    withdrawal["value"] = None
    withdrawal["revision_id"] = "withdrawal"
    withdrawal["available_at"] = "2025-06-01T00:00:00Z"
    changed = derive_debt_totals(pd.concat([rows, withdrawal]), RECIPE)
    outputs = changed.loc[changed.taxonomy.eq("derived")]
    assert outputs.value.iloc[0] == 50 and outputs.value.iloc[-1] is None
    previous = outputs.iloc[:1].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        previous, derived.loc[derived.taxonomy.eq("derived")].reset_index(drop=True)
    )
    direct = fact("long_term_debt", 60, start=None)
    direct.update(period_kind="instant", available_at="2025-06-01T00:00:00Z")
    merged = derive_debt_totals(pd.concat([rows, pd.DataFrame([direct])]), RECIPE)
    result = compute_fundamental_features(pd.DatetimeIndex(["2025-06-02T00:00:00Z"]), facts=merged)
    assert result["fundamentals.long_term_debt"].iloc[0] == 60


def test_debt_ambiguous_units_filing_and_overlapping_recipe():
    rows = debt_rows()
    rows.loc[1, "source_document_id"] = "filing:other"
    assert derive_debt_totals(rows, RECIPE).query("taxonomy == 'derived'").value.isna().all()
    with pytest.raises(ValueError, match="overlapping"):
        derive_debt_totals(
            rows, RECIPE | dict(components=["long_term_debt", "long_term_debt_current"])
        )
    with pytest.raises(ValueError, match="proof"):
        derive_debt_totals(rows, RECIPE | dict(components_disjoint=False))
