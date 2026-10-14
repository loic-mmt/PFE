"""Release-aware COT levels and compatible net positioning."""

import numpy as np
import pandas as pd

from pfe.data.contracts import normalize_observations, observations_asof
from pfe.features.graph import decision_grid


def compute_cot_features(
    decisions,
    observations,
    *,
    entity_id,
    report,
    long_metric,
    short_metric,
    open_interest_metric="open_interest",
    max_age="14D",
    max_observation_age="21D",
):
    frame = normalize_observations(observations)
    index = decision_grid(decisions, [entity_id])
    age_limit = pd.Timedelta(max_age)
    if pd.isna(age_limit) or age_limit < pd.Timedelta(0):
        raise ValueError("COT max_age must be nonnegative")
    observation_limit = pd.Timedelta(max_observation_age)
    if pd.isna(observation_limit) or observation_limit < pd.Timedelta(0):
        raise ValueError("COT max_observation_age must be nonnegative")
    names = [f"cot.{report}.{name}" for name in (long_metric, short_metric, open_interest_metric)]
    if len(set(names)) != len(names):
        raise ValueError("COT long, short and open interest metrics must be distinct")
    selected = frame.loc[
        frame.entity_id.eq(entity_id)
        & frame.source_id.eq("cftc")
        & frame.metric.isin(names)
        & frame.unit.eq("contracts")
    ]
    rows = []
    for decision in index:
        known = observations_asof(selected, decision)
        known = known.loc[known.observation_date <= decision.date().isoformat()]
        row = dict(
            net=np.nan,
            net_fraction_oi=np.nan,
            net_change_1w=np.nan,
            age_seconds=np.nan,
            observation_age_seconds=np.nan,
            mask=0.0,
            point_in_time=np.nan,
        )
        if len(known):
            day = known.observation_date.max()
            current = known.loc[known.observation_date.eq(day)]
            if (
                len(current) == len(names)
                and set(current.metric) == set(names)
                and current.revision_id.nunique() == 1
            ):
                values = current.set_index("metric").value
                age = (decision - current.available_at.min()).total_seconds()
                row.update(
                    age_seconds=age,
                    observation_age_seconds=(
                        decision - pd.Timestamp(day, tz="UTC")
                    ).total_seconds(),
                    point_in_time=float(current.point_in_time.all()),
                )
                if (
                    values.notna().all()
                    and age <= age_limit.total_seconds()
                    and row["observation_age_seconds"] <= observation_limit.total_seconds()
                ):
                    net = values[names[0]] - values[names[1]]
                    row.update(
                        net=net,
                        net_fraction_oi=net / values[names[2]] if values[names[2]] > 0 else np.nan,
                        mask=1.0,
                    )
                    prior_day = (pd.Timestamp(day) - pd.Timedelta("7D")).date().isoformat()
                    prior = known.loc[
                        known.observation_date.eq(prior_day) & known.metric.isin(names[:2])
                    ]
                    if (
                        len(prior) == 2
                        and set(prior.metric) == set(names[:2])
                        and prior.revision_id.nunique() == 1
                        and prior.value.notna().all()
                    ):
                        previous = prior.set_index("metric").value
                        row["net_change_1w"] = net - (previous[names[0]] - previous[names[1]])
        rows.append({"cot." + k: v for k, v in row.items()})
    return pd.DataFrame(rows, index=index)
