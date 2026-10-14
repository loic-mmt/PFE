"""Comparable analyst snapshots and strictly pre-announcement earnings surprise."""

import numpy as np
import pandas as pd

from pfe.data.contracts import _instant, normalize_observations, observations_asof
from pfe.data.estimates import ESTIMATE_CONTEXT
from pfe.features.graph import decision_grid


def _consensus(frame):
    points = frame.loc[frame.value.map(lambda v: v["statement_type"] == "analyst_estimate")]
    points = points.copy()
    points["_snapshot"] = points.value.map(lambda v: _instant(v["snapshot_at"], "snapshot_at"))
    points = points.sort_values(["_snapshot", "available_at"], kind="stable")
    explicit = points.loc[points.value.map(lambda v: v["statistic"] == "mean")]
    if len(explicit):
        row = explicit.iloc[-1]
        value = row.value
        return (
            value["value"] if value["active"] else None,
            value["stddev"],
            value["analyst_count"],
            row.available_at,
            "provider_mean",
        )
    detail = points.loc[points.value.map(lambda v: v["statistic"] == "detail")]
    detail = detail.copy()
    detail["_analyst"] = detail.value.map(lambda v: v["analyst_id"])
    detail = detail.drop_duplicates("_analyst", keep="last")
    detail = detail.loc[detail.value.map(lambda v: v["active"] and v["value"] is not None)]
    if len(detail):
        values = np.array([v["value"] for v in detail.value])
        return (
            values.mean(),
            values.std(ddof=0),
            len(values),
            detail.available_at.max(),
            "available_analysts_only",
        )
    return None, None, None, None, None


def compute_estimate_features(decisions, observations, *, entity_id, source_id, context):
    if set(context) != set(ESTIMATE_CONTEXT):
        raise ValueError(
            "Estimate context must specify metric, fiscal period and all unit/accounting bases"
        )
    frame = normalize_observations(observations)
    selected = frame.loc[
        frame.entity_id.eq(entity_id)
        & frame.source_id.eq(source_id)
        & frame.value.map(
            lambda v: isinstance(v, dict) and all(v.get(k) == value for k, value in context.items())
        )
    ]
    rows = []
    for decision in decision_grid(decisions, [entity_id]):
        known = observations_asof(selected, decision)
        mean, std, count, at, basis = _consensus(known)
        row = dict(
            mean=np.nan if mean is None else mean,
            stddev=np.nan if std is None else std,
            analyst_count=np.nan if count is None else count,
            dispersion=np.nan,
            revision=np.nan,
            revision_fraction=np.nan,
            surprise=np.nan,
            surprise_fraction=np.nan,
            mask=float(mean is not None),
            surprise_mask=0.0,
            age_seconds=(decision - at).total_seconds() if at is not None else np.nan,
            population=basis,
            consensus_source_document_id=None,
            actual_source_document_id=None,
            coverage_complete=0.0,
            point_in_time=float(known.point_in_time.all()) if len(known) else np.nan,
        )
        if mean is not None:
            if std is not None and mean != 0:
                row["dispersion"] = std / abs(mean)
            earlier = selected.loc[selected.available_at < at]
            if len(earlier):
                prior = observations_asof(earlier, earlier.available_at.max())
                old, _, _, _, old_basis = _consensus(prior)
                if old is not None and old_basis == basis:
                    row["revision"] = mean - old
                    row["revision_fraction"] = (mean - old) / abs(old) if old else np.nan
        actuals = known.loc[
            known.value.map(lambda v: v["statement_type"] == "reported_actual")
        ].copy()
        if len(actuals):
            actuals["_snapshot"] = actuals.value.map(
                lambda v: _instant(v["snapshot_at"], "snapshot_at")
            )
            actual = actuals.sort_values(["_snapshot", "available_at"], kind="stable").iloc[-1]
            announcement = (
                _instant(actual.value["announcement_at"], "announcement_at")
                if actual.value.get("announcement_at") is not None
                and actual.value["active"]
                and actual.value["value"] is not None
                else pd.NaT
            )
            if pd.notna(announcement) and announcement <= decision:
                before = selected.loc[
                    (selected.available_at < announcement)
                    & selected.value.map(
                        lambda v, cutoff=announcement: (
                            _instant(v["snapshot_at"], "snapshot_at") < cutoff
                        )
                    )
                ]
                baseline = observations_asof(before, announcement)
                estimate, _, _, _, _ = _consensus(baseline)
                if estimate is not None:
                    row.update(
                        surprise=actual.value["value"] - estimate,
                        surprise_fraction=(actual.value["value"] - estimate) / abs(estimate)
                        if estimate
                        else np.nan,
                        surprise_mask=1.0,
                        actual_source_document_id=actual.source_document_id,
                    )
                    relevant = baseline.loc[
                        baseline.value.map(lambda v: v["statement_type"] == "analyst_estimate")
                    ]
                    row["consensus_source_document_id"] = "|".join(
                        sorted(set(relevant.source_document_id.dropna()))
                    )
        rows.append({"estimates." + k: v for k, v in row.items()})
    return pd.DataFrame(rows, index=decision_grid(decisions, [entity_id]))
