"""Causal ranks and standardized comparisons across a declared universe."""

import numpy as np
import pandas as pd

from pfe.data.classifications import classifications_asof, normalize_classifications
from pfe.data.security_sources import security_universe_asof


def compute_cross_sectional_features(
    decisions,
    observations,
    *,
    entity_ids,
    registry,
    security_master=None,
    universe_source=None,
    classifications=None,
    classification_source=None,
    min_count=2,
    require_same_observation=True,
    limits=None,
):
    """Compare exact metric/unit/source/horizon specifications at each decision.

    Default requires the same economic observation time within each comparison
    group, avoiding a rank mixing yesterday's stale close with today's close.
    Percentile uses average ranks / N; std is population; constant groups have
    missing z-scores. Group history and universe are resolved as-of. Missing input
    never contributes a zero. The caller's universe is explicit if no master is
    provided. Each output family includes its full input known-time gate.
    """
    from pfe.datasets.core import NAT, build_feature_panel

    if type(min_count) is not int or min_count < 2 or type(require_same_observation) is not bool:
        raise ValueError("Require min_count >= 2 and a boolean observation policy")
    panel = build_feature_panel(
        decisions, observations, entity_ids=entity_ids, registry=registry, limits=limits
    )
    groups = normalize_classifications(classifications) if classifications is not None else None
    records = []
    for i, stamp in enumerate(panel.decisions_ns):
        decision = pd.Timestamp(stamp, tz="UTC")
        membership = {entity: None for entity in entity_ids}
        member_clock = NAT
        eligible = set(entity_ids)
        if security_master is not None:
            universe = security_universe_asof(security_master, decision, source_id=universe_source)
            eligible &= set(universe.entity_id)
            if len(universe):
                member_clock = universe.available_at.max().value
        group_clock = {}
        if groups is not None:
            known_groups = classifications_asof(groups, decision, source_id=classification_source)
            for row in known_groups.itertuples():
                membership[row.entity_id] = row.group_id
                group_clock[row.entity_id] = row.available_at.value
        rows = [dict(decision_at=decision, entity_id=entity) for entity in entity_ids]
        for k, spec in enumerate(panel.registry.features):
            valid = panel.mask[i, :, k] & np.array([entity in eligible for entity in entity_ids])
            for scope in ("universe", "group") if groups is not None else ("universe",):
                buckets = {"universe": list(range(len(entity_ids)))} if scope == "universe" else {}
                if scope == "group":
                    for j, entity in enumerate(entity_ids):
                        if membership[entity] is not None:
                            buckets.setdefault(membership[entity], []).append(j)
                prefix = f"cross_section.{spec.name}.{scope}"
                for row in rows:
                    row.update(
                        {
                            f"{prefix}.{name}": np.nan
                            for name in ("rank_pct", "zscore", "demeaned", "mean", "std")
                        }
                    )
                    row[f"{prefix}.count"], row[f"{prefix}.mask"] = 0, False
                    row[f"{prefix}.available_at"] = pd.NaT
                for locations in buckets.values():
                    selected = np.array([j for j in locations if valid[j]], dtype=int)
                    count = len(selected)
                    if not count:
                        continue
                    same = len(np.unique(panel.observation_at_ns[i, selected, k])) == 1
                    usable = count >= min_count and (same or not require_same_observation)
                    values = panel.values[i, selected, k]
                    known_at = max(
                        member_clock,
                        int(panel.available_at_ns[i, selected, k].max()),
                        max((group_clock.get(entity_ids[j], NAT) for j in selected), default=NAT)
                        if scope == "group"
                        else NAT,
                    )
                    mean, std = values.mean(), values.std(ddof=0)
                    ranks = pd.Series(values).rank(method="average", pct=True).to_numpy()
                    for position, j in enumerate(selected):
                        rows[j][f"{prefix}.count"] = count
                        rows[j][f"{prefix}.available_at"] = pd.Timestamp(known_at, tz="UTC")
                        if usable:
                            rows[j].update(
                                {
                                    f"{prefix}.rank_pct": ranks[position],
                                    f"{prefix}.zscore": (values[position] - mean) / std
                                    if std > 0
                                    else np.nan,
                                    f"{prefix}.demeaned": values[position] - mean,
                                    f"{prefix}.mean": mean,
                                    f"{prefix}.std": std,
                                    f"{prefix}.mask": True,
                                }
                            )
        records.extend(rows)
    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["decision_at", "entity_id"])
        )
    result = result.set_index(["decision_at", "entity_id"])
    result.attrs.update(
        timestamp_semantics="explicit_decision",
        registry=panel.registry.to_dict(),
        universe="security_master_asof" if security_master is not None else "caller_declared",
    )
    return result
