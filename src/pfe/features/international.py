"""Foreign factors on explicit aware US decisions, with source/link masks.

Sector context lists only help choose sources. A company exposure requires a
known graph edge. No foreign close, currency conversion or supplier is inferred.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfe.data.contracts import normalize_observations, observations_asof
from pfe.data.graph import graph_asof
from pfe.features.graph import decision_grid

INTERNATIONAL_FEATURE_GROUPS = ("series", "links", "detections", "quality")
SECTOR_CONTEXTS = {
    "semiconductors": ("TW", "KR", "JP", "NL", "CN"),
    "automotive": ("CN", "EU", "JP", "commodities", "energy"),
    "energy": ("OPEC", "global_oil", "gas", "US_stocks_production"),
    "banking_insurance": ("US_rates", "credit_spreads", "EU", "global_liquidity"),
    "industrial_aerospace": ("supply_chains", "EU", "JP", "energy"),
    "consumer": ("CN", "EU", "FX", "inflation", "imports"),
    "software_cloud": ("technology_capex", "corporate_spending", "customer_results"),
}


@dataclass(frozen=True)
class InternationalSeries:
    name: str
    entity_id: str
    metric: str
    unit: str
    source_id: str
    frequency: str
    max_age: str | None = None
    max_observation_age: str | None = None

    def __post_init__(self):
        for field in ("name", "entity_id", "metric", "unit", "source_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("International series identifiers must be explicit")
        if self.frequency not in ("daily", "monthly", "quarterly", "annual", "fiscal"):
            raise ValueError("International series needs a supported explicit frequency")
        for value in (self.max_age, self.max_observation_age):
            if value is not None and (
                pd.isna(pd.Timedelta(value)) or pd.Timedelta(value) < pd.Timedelta(0)
            ):
                raise ValueError("International age bounds must be nonnegative")


def compute_international_features(decisions, observations, *, series):
    """Latest known economic period and exact previous/year-ago period changes.

    Revision of an older period cannot displace the newer economic period.
    Unknown/null latest value remains missing. Receipt, publication and economic
    ages are separate; no annual/monthly indicator becomes a daily observation.
    Fiscal periods expose levels only; explicit accounting joins belong upstream.
    """
    selections = [InternationalSeries(**s) if isinstance(s, dict) else s for s in series]
    if (
        not selections
        or any(not isinstance(s, InternationalSeries) for s in selections)
        or len({s.name for s in selections}) != len(selections)
    ):
        raise ValueError("Provide distinct InternationalSeries selections")
    index = decision_grid(decisions, [s.name for s in selections])
    frame = normalize_observations(observations)
    for column in ("period_end", "frequency"):
        if column not in frame:
            raise ValueError("International observations require period_end and frequency")
    rows = []
    for decision in index:
        row = {}
        for spec in selections:
            prefix = "international." + spec.name + "."
            values = {
                "level": np.nan,
                "change": np.nan,
                "pct_change": np.nan,
                "yoy": np.nan,
                "mask.available": 0.0,
                "mask.value": 0.0,
                "mask.fresh": 0.0,
                "quality.frequency_match": 0.0,
                "quality.point_in_time": np.nan,
                "age_seconds": np.nan,
                "publication_age_seconds": np.nan,
                "observation_age_seconds": np.nan,
                "source_id": None,
                "revision_id": None,
                "source_document_id": None,
                "observation_period": None,
            }
            subset = frame.loc[
                frame.entity_id.eq(spec.entity_id)
                & frame.metric.eq(spec.metric)
                & frame.source_id.eq(spec.source_id)
                & frame.unit.eq(spec.unit)
            ]
            known = observations_asof(subset, decision)
            known = known.loc[known.period_end <= decision.date().isoformat()]
            if len(known):
                latest = known.loc[known.period_end.eq(known.period_end.max())]
                if len(latest) == 1:
                    point = latest.iloc[0]
                    matches = point.frequency == spec.frequency
                    age = (decision - point.available_at).total_seconds()
                    econ_age = (decision - pd.Timestamp(point.period_end, tz="UTC")).total_seconds()
                    pub_age = (
                        (decision - point.published_at).total_seconds()
                        if pd.notna(point.published_at)
                        else np.nan
                    )
                    fresh = (
                        spec.max_age is None or age <= pd.Timedelta(spec.max_age).total_seconds()
                    ) and (
                        spec.max_observation_age is None
                        or econ_age <= pd.Timedelta(spec.max_observation_age).total_seconds()
                    )
                    values.update(
                        {
                            "mask.available": 1.0,
                            "mask.fresh": float(fresh),
                            "quality.frequency_match": float(matches),
                            "age_seconds": age,
                            "observation_age_seconds": econ_age,
                            "publication_age_seconds": pub_age,
                            "quality.point_in_time": float(point.get("point_in_time", False)),
                            "source_id": point.source_id,
                            "revision_id": point.revision_id,
                            "source_document_id": point.get("source_document_id"),
                            "observation_period": point.get("observation_period"),
                        }
                    )
                    numeric = (
                        isinstance(point.value, (int, float, np.number))
                        and not isinstance(point.value, (bool, np.bool_))
                        and pd.notna(point.value)
                    )
                    if numeric and matches and fresh:
                        values.update(level=float(point.value), **{"mask.value": 1.0})
                        freq = {"daily": "D", "monthly": "M", "quarterly": "Q", "annual": "Y"}.get(
                            spec.frequency
                        )
                        if freq:
                            current = pd.Period(point.period_end, freq=freq)
                            for distance, metric in (
                                (1, "pct_change"),
                                (
                                    {"daily": 0, "monthly": 12, "quarterly": 4, "annual": 1}[
                                        spec.frequency
                                    ],
                                    "yoy",
                                ),
                            ):
                                if not distance:
                                    continue
                                end = (current - distance).end_time.date().isoformat()
                                previous = known.loc[
                                    known.period_end.eq(end) & known.frequency.eq(spec.frequency)
                                ]
                                if (
                                    len(previous) == 1
                                    and isinstance(previous.iloc[0].value, (int, float, np.number))
                                    and pd.notna(previous.iloc[0].value)
                                ):
                                    old = float(previous.iloc[0].value)
                                    if metric == "pct_change":
                                        values["change"] = float(point.value) - old
                                    values[metric] = (
                                        float(point.value) / old - 1 if old != 0 else np.nan
                                    )
            row.update({prefix + k: v for k, v in values.items()})
        rows.append(row)
    return pd.DataFrame(rows, index=index)


def compute_linked_international_features(
    decisions,
    observations,
    graph_history,
    *,
    series,
    entity_ids,
    relations=("supplier", "customer", "factor_exposure", "country_exposure", "commodity_exposure"),
    graph_max_age=None,
):
    """Attach foreign factors only when a known, active documented edge exists.

    No weights synthesized from sector/country membership. Values remain in
    their original units; masks distinguish missing links from missing sources.
    """
    selections = [InternationalSeries(**s) if isinstance(s, dict) else s for s in series]
    factors = compute_international_features(decisions, observations, series=selections)
    decision_grid(decisions, entity_ids)
    rows = []
    for decision, factor_row in factors.iterrows():
        graph = graph_asof(graph_history, decision, max_age=graph_max_age)
        for entity in entity_ids:
            row = {"decision_at": decision, "entity_id": entity}
            for spec in selections:
                prefix = "international." + spec.name + "."
                links = graph.edges.loc[
                    graph.edges.relation_type.isin(relations)
                    & (
                        (
                            graph.edges.source_entity_id.eq(entity)
                            & graph.edges.target_entity_id.eq(spec.entity_id)
                        )
                        | (
                            graph.edges.target_entity_id.eq(entity)
                            & graph.edges.source_entity_id.eq(spec.entity_id)
                        )
                    )
                ]
                linked = bool(len(links))
                row.update({k: v for k, v in factor_row.items() if k.startswith(prefix)})
                for metric in ("level", "change", "pct_change", "yoy"):
                    if not linked:
                        row[prefix + metric] = np.nan
                row[prefix + "mask.link"] = float(linked)
                row[prefix + "mask.value"] *= float(linked)
                row[prefix + "link_source_documents"] = (
                    "|".join(sorted(set(links.source_document_id))) if linked else None
                )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["decision_at", "entity_id"])


def compute_gdelt_features(decisions, observations, *, entity_id, windows=("7D", "30D")):
    """Counts in the supplied discovery corpus; provider classification/tone only.

    Repeated captures never reset first availability. Zero means no detections
    in this corpus, not proof no real event. No company link from search keywords.
    """
    index = decision_grid(decisions, [entity_id])
    windows = tuple(windows)
    if (
        not windows
        or len(set(windows)) != len(windows)
        or any(pd.isna(pd.Timedelta(w)) or pd.Timedelta(w) <= pd.Timedelta(0) for w in windows)
    ):
        raise ValueError("GDELT windows must be distinct positive durations")
    frame = normalize_observations(observations)
    frame = frame.loc[
        frame.entity_id.eq(entity_id) & frame.source_id.isin(("gdelt", "gdelt_events"))
    ]
    rows = []
    for decision in index:
        known = frame.loc[frame.available_at <= decision].sort_values("available_at", kind="stable")
        known = known.drop_duplicates(["source_id", "metric"], keep="first")
        row = {
            "gdelt.quality.complete_feed": 0.0,
            "gdelt.quality.verified_events": 0.0,
            "gdelt.mask.source_available": float(bool(len(known))),
        }
        for window in windows:
            subset = known.loc[known.available_at > decision - pd.Timedelta(window)]
            row["gdelt.article_detections.count_" + window] = float(
                subset.source_id.eq("gdelt").sum()
            )
            row["gdelt.cameo_events.count_" + window] = float(
                subset.source_id.eq("gdelt_events").sum()
            )
        row["gdelt.age_seconds"] = (
            (decision - known.available_at.max()).total_seconds() if len(known) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows, index=index)
