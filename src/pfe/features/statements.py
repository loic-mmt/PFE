"""Reviewed textual actuals/guidance/estimates remain separate, in original units."""

import numpy as np
import pandas as pd

from pfe.data.contracts import _instant
from pfe.features.graph import decision_grid


def compute_statement_features(decisions, statements, *, entity_ids, specifications):
    """Specs map output names to metric/statement_type/currency/unit_scale/segment_name.

    No GAAP assumption, currency conversion, midpoint, surprise or consensus
    aggregation. Latest known compatible fiscal period is selected; ambiguous
    simultaneous sources/segments are missing. Period labels remain in metadata.
    """
    index = decision_grid(decisions, entity_ids)
    required = {"metric", "statement_type", "currency", "unit_scale", "segment_name"}
    allowed = required | {
        "fiscal_period_label",
        "fiscal_period_start",
        "fiscal_period_end",
        "source_id",
    }
    if not specifications or any(
        not required.issubset(spec) or not set(spec).issubset(allowed)
        for spec in specifications.values()
    ):
        raise ValueError(f"Statement specifications require {sorted(required)}")
    data = statements.copy()
    data["available_at"] = data.available_at.map(lambda value: _instant(value, "available_at"))
    data["statement_origin_at"] = data.statement_origin_at.map(
        lambda value: _instant(value, "statement_origin_at")
    )
    rows = []
    for decision in index:
        known = data.loc[data.available_at <= decision].sort_values("available_at", kind="stable")
        known = known.drop_duplicates("fact_id", keep="last")
        known = known.loc[known.review_status.eq("approved")]
        for entity in entity_ids:
            row = {"decision_at": decision, "entity_id": entity}
            for name, spec in specifications.items():
                subset = known.loc[known.entity_id.eq(entity)]
                for key, value in spec.items():
                    subset = subset.loc[
                        subset[key].isna() if value is None else subset[key].eq(value)
                    ]
                if spec["statement_type"] == "reported_actual":
                    subset = subset.loc[
                        subset.fiscal_period_end.notna()
                        & (subset.fiscal_period_end <= decision.date().isoformat())
                    ]
                    if len(subset):
                        subset = subset.loc[
                            subset.fiscal_period_end.eq(subset.fiscal_period_end.max())
                        ]
                elif spec["statement_type"] not in ("management_guidance", "analyst_estimate"):
                    raise ValueError("Invalid textual financial statement_type")
                periods = subset[
                    ["fiscal_period_start", "fiscal_period_end", "fiscal_period_label"]
                ].drop_duplicates()
                ambiguous = len(periods) > 1 or subset.source_id.nunique() > 1
                if len(subset) and not ambiguous:
                    subset = subset.loc[
                        subset.statement_origin_at.eq(subset.statement_origin_at.max())
                    ]
                    subset = subset.drop_duplicates("fact_id", keep="last")
                point = subset.iloc[0] if len(subset) == 1 and not ambiguous else None
                prefix = "statements." + name
                for key in ("value", "lower_bound", "upper_bound"):
                    row[prefix + "." + key] = (
                        float(point[key]) if point is not None and pd.notna(point[key]) else np.nan
                    )
                row[prefix + ".age_seconds"] = (
                    (decision - point.available_at).total_seconds() if point is not None else np.nan
                )
                row[prefix + ".source_age_seconds"] = (
                    (decision - point.statement_origin_at).total_seconds()
                    if point is not None
                    else np.nan
                )
                row[prefix + ".period_age_days"] = (
                    (decision.date() - pd.Timestamp(point.fiscal_period_end).date()).days
                    if point is not None and pd.notna(point.fiscal_period_end)
                    else np.nan
                )
                row[prefix + ".mask"] = float(point is not None)
                row[prefix + ".quality.ambiguous"] = float(ambiguous or len(subset) > 1)
                row[prefix + ".quality.point_in_time"] = (
                    float(point.point_in_time) if point is not None else np.nan
                )
                row[prefix + ".quality.period_known"] = (
                    float(pd.notna(point.fiscal_period_end)) if point is not None else np.nan
                )
            rows.append(row)
    out = (
        pd.DataFrame(rows).set_index(["decision_at", "entity_id"])
        if rows
        else pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [index, entity_ids], names=["decision_at", "entity_id"]
            )
        )
    )
    out.attrs.update(
        timestamp_semantics="explicit_decision", statement_specifications=specifications
    )
    return out
