"""Conflict activity in the known supplied corpus, distinct from annual war status."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pfe.data.captures import ObservationCapture
from pfe.data.conflicts import normalize_conflict_observations
from pfe.data.contracts import _instant
from pfe.data.graph import graph_asof
from pfe.features.calendar import calendar_index
from pfe.features.graph import decision_grid

VIOLENCE_TYPES = {1: "state_based", 2: "non_state", 3: "one_sided"}


@dataclass(frozen=True)
class ConflictFeatureLimits:
    max_history_rows: int = 25_000
    max_decisions: int = 10_000
    max_output_cells: int = 2_000_000
    max_work: int = 5_000_000

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Conflict feature budgets must be positive integers")


def _windows(windows):
    if isinstance(windows, str):
        raise ValueError("Conflict windows must be an iterable of day durations")
    values = [(str(w), pd.Timedelta(w)) for w in windows]
    if (
        not values
        or len({label for label, _ in values}) != len(values)
        or any(pd.isna(v) or v <= pd.Timedelta(0) or v % pd.Timedelta("1D") for _, v in values)
    ):
        raise ValueError("Conflict windows require distinct positive whole-day durations")
    return values


def _labels(values, label):
    if isinstance(values, str) or len(set(values)) != len(values):
        raise ValueError(f"{label} must contain distinct explicit selections")
    return tuple(values)


def _flatten(frame):
    columns = [
        "record_type",
        "conflict_id",
        "event_id",
        "country_id",
        "region",
        "type_of_violence",
        "date_start",
        "date_end",
        "best",
        "low",
        "high",
        "year",
        "intensity_level",
        "active",
    ]
    values = pd.DataFrame(frame.value.tolist(), columns=columns, index=frame.index)
    values["active"] = values.active.map(lambda v: True if pd.isna(v) else bool(v)).astype(bool)
    return pd.concat([frame.drop(columns="value"), values], axis=1)


def _summary(current, previous, first_times, decision, duration):
    values = {
        "events": float(len(current)),
        "active_conflicts": float(current.conflict_id.nunique()),
        "affected_countries": float(current.country_id.nunique()),
        "events_change": float(len(current) - len(previous)),
        "active_conflicts_change": float(
            current.conflict_id.nunique() - previous.conflict_id.nunique()
        ),
        "newly_observed_conflicts": float(
            sum(
                at > decision - duration
                for conflict, at in first_times.items()
                if conflict in set(current.conflict_id)
            )
        ),
    }
    for field in ("best", "low", "high"):
        values["fatalities_" + field] = (
            float(current[field].sum()) if current[field].notna().all() else np.nan
        )
    values["fatalities_best_change"] = (
        float(current.best.sum() - previous.best.sum())
        if current.best.notna().all() and previous.best.notna().all()
        else np.nan
    )
    values["fatalities_coverage"] = float(current.best.notna().mean()) if len(current) else np.nan
    return values


def compute_conflict_features(
    decisions,
    observations,
    *,
    source_id="ucdp",
    windows=("30D", "90D"),
    countries=(),
    regions=(),
    limits=None,
):
    """Counts describe known records, not proof of a complete world news universe.

    Window membership uses the economic date_end label in UTC civil dates,
    AFTER available_at filtering. All violent events and distinct conflict IDs
    are counted separately by violence type; one-sided violence is not a war.
    Newly observed means first receipt in this corpus, not actual conflict onset.
    Annual wars use only available classifications of a completed economic year.
    No-year/no-source remains missing; no annual hindsight or deduced live war label.
    """
    index = calendar_index(decisions)
    windows = _windows(windows)
    countries, regions = _labels(countries, "countries"), _labels(regions, "regions")
    if any(type(v) is not int or v < 1 for v in countries):
        raise ValueError("Conflict countries must use positive integer GW codes")
    if any(not isinstance(v, str) or not v.strip() for v in regions):
        raise ValueError("Conflict regions require explicit nonempty labels")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("Conflict source_id must be explicit")
    limits = (
        ConflictFeatureLimits(**limits)
        if isinstance(limits, dict)
        else limits or ConflictFeatureLimits()
    )
    capture = observations if isinstance(observations, ObservationCapture) else None
    capture_at = None
    if capture is not None and capture.metadata["source_id"] == source_id:
        capture_at = _instant(capture.metadata["received_at"], "received_at")
        publication = capture.metadata["request"]["query"].get("published_at")
        if publication is not None:
            capture_at = max(capture_at, _instant(publication, "published_at"))
    frame = normalize_conflict_observations(capture.observations if capture else observations)
    frame = frame.loc[frame.source_id.eq(source_id)]
    scopes = [("global", None, None)]
    scopes += [(f"country[{v}]", "country_id", v) for v in countries]
    scopes += [(f"region[{v}]", "region", v) for v in regions]
    cells = len(index) * (10 + len(scopes) * (6 + len(windows) * 33))
    if (
        len(frame) > limits.max_history_rows
        or len(index) > limits.max_decisions
        or cells > limits.max_output_cells
        or len(index) * max(1, len(frame)) * len(scopes) * len(windows) > limits.max_work
    ):
        raise ValueError("Conflict history/output exceeds budget; use smaller batches")
    data = _flatten(frame)
    rows = []
    for decision in index:
        known_all = data.loc[data.available_at <= decision]
        known = known_all.drop_duplicates("metric", keep="last")
        known = known.loc[known.active & (known.observation_date <= decision.date().isoformat())]
        events = known.loc[known.record_type.eq("event")]
        event_observed = bool(known_all.record_type.eq("event").any()) or bool(
            capture
            and capture.metadata["source_id"] == source_id
            and capture.metadata["request"]["query"]["resource"] == "gedevents"
            and capture_at is not None
            and capture_at <= decision
        )
        observed = bool(len(known_all)) or bool(capture_at is not None and capture_at <= decision)
        row = {
            "conflicts.mask.source_available": float(observed),
            "conflicts.mask.event_source_available": float(event_observed),
            "conflicts.quality.complete_global_universe": 0.0,
            "conflicts.quality.complete_supplied_query": float(capture.complete)
            if observed and capture
            else 0.0,
            "conflicts.quality.point_in_time": float(known.point_in_time.all())
            if len(known)
            else np.nan,
        }
        for scope, column, value in scopes:
            subset = events if column is None else events.loc[events[column].eq(value)]
            all_events = known_all.loc[known_all.record_type.eq("event")]
            if column is not None:
                all_events = all_events.loc[all_events[column].eq(value)]
            prefix = "conflicts." + scope
            row[prefix + ".age_seconds"] = (
                (decision - subset.available_at.max()).total_seconds() if len(subset) else np.nan
            )
            row[prefix + ".observation_age_days"] = (
                (pd.Timestamp(decision.date()) - pd.Timestamp(subset.date_end.max())).days
                if len(subset)
                else np.nan
            )
            for label, duration in windows:
                end = pd.Timestamp(decision.date())
                lower, previous_lower = end - duration, end - 2 * duration
                for type_id, name in VIOLENCE_TYPES.items():
                    typed = subset.loc[subset.type_of_violence.eq(type_id)]
                    dates = pd.to_datetime(typed.date_end)
                    current = typed.loc[(dates > lower) & (dates <= end)]
                    previous = typed.loc[(dates > previous_lower) & (dates <= lower)]
                    first = (
                        all_events.loc[all_events.type_of_violence.eq(type_id)]
                        .groupby("conflict_id")
                        .available_at.min()
                        .to_dict()
                    )
                    values = _summary(current, previous, first, decision, duration)
                    row.update(
                        {
                            f"{prefix}.{name}.{metric}_{label}": v if event_observed else np.nan
                            for metric, v in values.items()
                        }
                    )
        annual = known.loc[known.record_type.eq("annual")]
        for name in ("year", "armed_conflicts", "wars", "age_seconds"):
            row["conflicts.annual." + name] = np.nan
        row["conflicts.annual.mask"] = float(bool(len(annual)))
        if len(annual):
            year = annual.year.max()
            selected = annual.loc[annual.year.eq(year)]
            row.update(
                {
                    "conflicts.annual.year": float(year),
                    "conflicts.annual.armed_conflicts": float(selected.conflict_id.nunique()),
                    "conflicts.annual.wars": float(
                        selected.loc[selected.intensity_level.eq(2), "conflict_id"].nunique()
                    ),
                    "conflicts.annual.age_seconds": (
                        decision - selected.available_at.max()
                    ).total_seconds(),
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows, index=index)
    result.attrs.update(
        timestamp_semantics="explicit_decision",
        source_id=source_id,
        activity_scope="supplied known corpus",
        window_clock="UTC economic date_end labels",
        war_definition="annual UCDP state-based intensity_level=2, never a live prediction",
    )
    return result


def compute_graph_conflict_features(
    decisions,
    observations,
    history,
    *,
    entity_ids,
    country_nodes,
    windows=("30D", "90D"),
    source_id="ucdp",
    graph_max_age=None,
    limits=None,
):
    """Propagate declared country exposure fractions, without inferring locations.

    country_nodes maps GW IDs to known graph country IDs. Revenue, production,
    supplier, operating_cost and raw_material stay separate. Only compatible
    weight periods/scopes and scalar percentages participate; no renormalization.
    Weighted activity is a sum of country counts times known fractions, not a
    probability or a distinct global conflict count. Missing source stays missing.
    """
    index = decision_grid(decisions, entity_ids)
    if (
        not isinstance(country_nodes, dict)
        or not country_nodes
        or any(
            type(k) is not int or k < 1 or not isinstance(v, str) or not v
            for k, v in country_nodes.items()
        )
        or len(set(country_nodes.values())) != len(country_nodes)
    ):
        raise ValueError("country_nodes must map distinct GW codes to explicit graph country IDs")
    windows = _windows(windows)
    limits = (
        ConflictFeatureLimits(**limits)
        if isinstance(limits, dict)
        else limits or ConflictFeatureLimits()
    )
    if (
        len(history.edges) > limits.max_history_rows
        or len(index) * len(entity_ids) * (31 + 10 * len(windows)) > limits.max_output_cells
        or len(index) * len(entity_ids) * max(1, len(history.edges)) > limits.max_work
    ):
        raise ValueError("Graph conflict history/output exceeds budget")
    factors = compute_conflict_features(
        index,
        observations,
        countries=tuple(country_nodes),
        windows=tuple(w for w, _ in windows),
        source_id=source_id,
        limits=limits,
    )
    reverse = {v: k for k, v in country_nodes.items()}
    rows = []
    for decision, factor in factors.iterrows():
        graph = graph_asof(history, decision, max_age=graph_max_age)
        known_countries = set(graph.nodes.loc[graph.nodes.node_type.eq("country"), "entity_id"])
        for entity in entity_ids:
            row = {"decision_at": decision, "entity_id": entity}
            row["graph.conflicts.quality.point_in_time"] = float(
                factor["conflicts.quality.point_in_time"] == 1 and graph.edges.point_in_time.all()
            )
            for kind in ("revenue", "production", "supplier", "operating_cost", "raw_material"):
                linked = graph.edges.loc[
                    graph.edges.owner_entity_id.eq(entity)
                    & graph.edges.relation_type.eq("exposure." + kind)
                    & graph.edges.target_entity_id.isin(known_countries)
                ]
                edges = linked.loc[linked.target_entity_id.isin(reverse)]
                weights = edges.loc[edges.exposure_percentage.notna()]
                compatible = len(weights[["weight_scope", "weight_period"]].drop_duplicates()) <= 1
                # Multiple declarations per country are ambiguous, even at equal percentages.
                unique = not weights.target_entity_id.duplicated().any()
                valid = compatible and unique and bool(len(weights))
                prefix = f"graph.conflicts.{kind}"
                row[prefix + ".linked_country_count"] = float(edges.target_entity_id.nunique())
                row[prefix + ".known_weight_count"] = float(len(weights))
                row[prefix + ".country_mapping_coverage"] = (
                    edges.target_entity_id.nunique() / linked.target_entity_id.nunique()
                    if len(linked)
                    else np.nan
                )
                row[prefix + ".known_weight_sum"] = (
                    float(weights.exposure_percentage.sum() / 100) if valid else np.nan
                )
                row[prefix + ".quality.ambiguous_weights"] = float(not compatible or not unique)
                row[prefix + ".mask"] = float(
                    valid and bool(factor["conflicts.mask.event_source_available"])
                )
                for label, _ in windows:
                    for metric in ("active_conflicts", "fatalities_best"):
                        values = []
                        for edge in weights.itertuples():
                            country = reverse[edge.target_entity_id]
                            name = f"conflicts.country[{country}].state_based.{metric}_{label}"
                            values.append(factor[name] * edge.exposure_percentage / 100)
                        row[f"{prefix}.{metric}_weighted_{label}"] = (
                            float(sum(values))
                            if valid and all(np.isfinite(v) for v in values)
                            else np.nan
                        )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["decision_at", "entity_id"])
