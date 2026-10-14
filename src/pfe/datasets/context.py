"""Explicit-unit context features -> canonical observations for existing datasets."""

import numpy as np
import pandas as pd

from pfe.data.contracts import OBSERVATION_COLUMNS, normalize_observations
from pfe.features.calendar import calendar_index


def context_feature_observations(
    features,
    *,
    units,
    source_id,
    revision_id,
    entity_ids=None,
    point_in_time=False,
    max_cells=1_000_000,
):
    """Convert selected columns; broadcast global context only to declared entities.

    Regular index: aware decisions plus explicit entity_ids. MultiIndex: exact
    decision_at/entity_id rows, with no broadcast. Observation and availability
    are the calculation's decision instant. Values remain missing; no fill or
    timestamp reconstruction. Upstream inputs must already be causally computed.
    """
    if not isinstance(features, pd.DataFrame) or not features.columns.is_unique:
        raise ValueError("Context features require a DataFrame with unique columns")
    if not isinstance(units, dict) or not units or not set(units) <= set(features):
        raise ValueError("Context units must explicitly select available feature columns")
    if (
        type(point_in_time) is not bool
        or type(max_cells) is not int
        or max_cells < 1
        or any(not isinstance(v, str) or not v for v in (source_id, revision_id))
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in units.items()
        )
    ):
        raise ValueError("Context source/revision/units and budgets must be explicit")
    if isinstance(features.index, pd.MultiIndex):
        if (
            features.index.names != ["decision_at", "entity_id"]
            or not features.index.is_unique
            or entity_ids is not None
        ):
            raise ValueError(
                "Entity context requires a unique decision_at/entity_id index, without broadcast"
            )
        decisions = features.index.get_level_values("decision_at")
        calendar_index(decisions.unique())
        count = len(features)
        positions = ((i, decision, entity) for i, (decision, entity) in enumerate(features.index))
    else:
        decisions = calendar_index(features.index)
        if (
            not entity_ids
            or isinstance(entity_ids, str)
            or len(set(entity_ids)) != len(entity_ids)
            or any(not isinstance(v, str) or not v for v in entity_ids)
        ):
            raise ValueError("Global context requires distinct explicit entity_ids")
        count = len(decisions) * len(entity_ids)
        positions = (
            (i, decision, entity) for i, decision in enumerate(decisions) for entity in entity_ids
        )
    if count * len(units) > max_cells:
        raise ValueError("Context observations exceed cell budget")
    for name in units:
        if not (
            pd.api.types.is_numeric_dtype(features[name])
            or pd.api.types.is_bool_dtype(features[name])
        ):
            raise ValueError("Context feature columns must be numeric")
    rows = []
    for position, decision, entity in positions:
        for name, unit in units.items():
            value = features.iloc[position][name]
            rows.append(
                dict(
                    entity_id=entity,
                    observation_date=decision,
                    available_at=decision,
                    published_at=None,
                    value=float(value) if pd.notna(value) and np.isfinite(value) else None,
                    source_id=source_id,
                    revision_id=revision_id,
                    metric=name,
                    unit=unit,
                    schema_version="1",
                    point_in_time=point_in_time,
                )
            )
    return normalize_observations(
        pd.DataFrame(rows, columns=[*OBSERVATION_COLUMNS, "point_in_time"])
    )
