"""Known declared topology, partial weights and compatible neighbor feature joins."""

import numpy as np
import pandas as pd

from pfe.data.contracts import _instant, normalize_observations, observations_asof
from pfe.data.graph import NODE_TYPES, graph_asof
from pfe.extraction.schemas import RELATIONS

GRAPH_FEATURE_GROUPS = (
    "degree",
    "dependency",
    "exposure",
    "neighbors",
    "quality",
    "channels",
    "two_hop",
    "topology",
    "dynamics",
)


def decision_grid(decisions, entity_ids):
    index = pd.DatetimeIndex(
        [_instant(value, "decision_at") for value in decisions], name="decision_at"
    )
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Decisions must be unique and increasing aware timestamps")
    if (
        not entity_ids
        or len(set(entity_ids)) != len(entity_ids)
        or any(not isinstance(value, str) or not value.strip() for value in entity_ids)
    ):
        raise ValueError("entity_ids must contain distinct nonempty IDs")
    return index


def compute_graph_features(
    decisions,
    history,
    *,
    entity_ids,
    node_features=None,
    node_feature_units=None,
    neighbor_relations=("supplier", "customer"),
    neighbor_node_types=("company",),
    exposure_targets=(),
    max_age=None,
    node_max_age=None,
    node_max_observation_age=None,
    require_same_observation=True,
    groups=None,
    neighbor_direction="both",
    topology_relations=("supplier", "customer"),
    topology_node_types=("company",),
    dynamics_windows=("30D",),
    limits=None,
):
    """Return (decision_at,entity_id) features; zeros describe only supplied corpus.

    Weights belong to the declared source company. Unknown/range weights are not
    zero and never renormalized. Weighted propagation is a sum of known fractions.
    Neighbor means need matching units/metric horizons AND economic observation
    dates; mixed periods or sources produce missing output with explicit masks.
    """
    index = decision_grid(decisions, entity_ids)
    from pfe.features.graph_analytics import (
        GraphFeatureLimits,
        graph_dynamics,
        topology_features,
        two_hop_features,
    )

    limits = (
        GraphFeatureLimits(**limits) if isinstance(limits, dict) else limits or GraphFeatureLimits()
    )
    selected = set(GRAPH_FEATURE_GROUPS if groups is None else groups)
    if isinstance(groups, str) or selected - set(GRAPH_FEATURE_GROUPS):
        raise ValueError("Unknown graph feature groups")
    if neighbor_direction not in ("both", "upstream", "downstream"):
        raise ValueError("Choose both/upstream/downstream neighbor direction")
    if type(require_same_observation) is not bool:
        raise ValueError("require_same_observation must be boolean")
    if len(index) > limits.max_decisions:
        raise ValueError("Graph decisions exceed budget")
    if not neighbor_node_types or any(kind not in NODE_TYPES for kind in neighbor_node_types):
        raise ValueError("neighbor_node_types must use explicit graph node types")
    units = node_feature_units or {"momentum": "fraction", "volatility": "fraction"}
    if not isinstance(units, dict):
        raise ValueError("node_feature_units must be a dictionary")
    estimated_columns = 120 + 30 * len(units) * (4 if "channels" in selected else 1)
    if (
        len(history.edges) > limits.max_history_rows
        or len(index) * len(entity_ids) * estimated_columns > limits.max_output_cells
    ):
        raise ValueError("Graph history/output exceeds budget; select smaller batches")
    if not units or any(
        not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in units.items()
    ):
        raise ValueError("node_feature_units must map explicit metric horizons to units")
    features = normalize_observations(node_features) if node_features is not None else None
    if (
        features is not None
        and features.value.map(lambda v: v is not None and type(v) not in (int, float)).any()
    ):
        raise ValueError("Neighbor node features must be numeric or null")
    age_limit = pd.Timedelta(node_max_age) if node_max_age is not None else None
    if age_limit is not None and (pd.isna(age_limit) or age_limit < pd.Timedelta(0)):
        raise ValueError("node_max_age must be nonnegative")
    economic_age = (
        pd.Timedelta(node_max_observation_age) if node_max_observation_age is not None else None
    )
    if economic_age is not None and (pd.isna(economic_age) or economic_age < pd.Timedelta(0)):
        raise ValueError("node_max_observation_age must be nonnegative")
    rows = []
    for decision in index:
        snapshot = graph_asof(history, decision, max_age=max_age)
        if len(snapshot.nodes) > limits.max_nodes or len(snapshot.edges) > limits.max_edges:
            raise ValueError("Graph exceeds node/edge budget")
        topology = (
            topology_features(
                snapshot,
                relations=topology_relations,
                node_types=topology_node_types,
                limits=limits,
            )
            if "topology" in selected
            else {}
        )
        dynamics = (
            graph_dynamics(
                decision,
                history,
                entity_ids=entity_ids,
                windows=dynamics_windows,
                max_age=max_age,
                limits=limits,
            )
            if "dynamics" in selected
            else {}
        )
        known_ids = set(snapshot.nodes.entity_id)
        known_features = observations_asof(features, decision) if features is not None else None
        for entity in entity_ids:
            edges = snapshot.edges
            incident = edges.loc[
                edges.source_entity_id.eq(entity) | edges.target_entity_id.eq(entity)
            ]
            own = incident.loc[incident.owner_entity_id.eq(entity)]
            row = {
                "decision_at": decision,
                "entity_id": entity,
                "graph.mask.entity_known": float(entity in known_ids),
                "graph.quality.declared_edges": float(len(incident)),
                "graph.quality.complete_network": 0.0,
                "graph.quality.point_in_time": float(incident.point_in_time.all())
                if len(incident)
                else np.nan,
            }
            row["graph.quality.topology.point_in_time"] = row["graph.quality.point_in_time"]
            row["graph.evidence_age_seconds"] = (
                max(
                    (decision - _instant(value, "evidence_at")).total_seconds()
                    for value in incident.evidence_at
                )
                if len(incident)
                else np.nan
            )
            for relation in (
                *RELATIONS,
                "supply_flow",
                "industry_member",
                "factor_exposure",
                "country_exposure",
                "commodity_exposure",
            ):
                subset = incident.loc[incident.relation_type.eq(relation)]
                if relation == "supply_flow":
                    subset = incident.loc[incident.relation_type.isin(("supplier", "customer"))]
                for direction, key in (("in", "target_entity_id"), ("out", "source_entity_id")):
                    opposite = "source_entity_id" if direction == "in" else "target_entity_id"
                    row[f"graph.degree.{relation}.{direction}"] = (
                        float(subset.loc[subset[key].eq(entity), opposite].nunique())
                        if entity in known_ids
                        else np.nan
                    )
            for kind in ("revenue", "supply"):
                column = kind + "_dependency_pct"
                subset = own.loc[own.source_role.isin(("supplier", "customer"))]
                weights = subset[column].dropna().astype(float) / 100
                weighted = subset.loc[subset[column].notna()]
                mixed_scope = len(weighted[["weight_scope", "weight_period"]].drop_duplicates()) > 1
                row[f"graph.dependency.{kind}.known_count"] = float(len(weights))
                row[f"graph.dependency.{kind}.coverage_fraction"] = (
                    len(weights) / len(subset) if len(subset) else np.nan
                )
                row[f"graph.dependency.{kind}.known_weight_sum"] = (
                    float(weights.sum()) if len(weights) and not mixed_scope else np.nan
                )
                row[f"graph.dependency.{kind}.concentration_known"] = (
                    float((weights**2).sum()) if len(weights) and not mixed_scope else np.nan
                )
                row[f"graph.mask.dependency.{kind}"] = float(bool(len(weights)) and not mixed_scope)
                row[f"graph.quality.{kind}.mixed_weight_scope"] = float(mixed_scope)
                row[f"graph.quality.{kind}.weight_sum_exceeds_one"] = (
                    float(weights.sum() > 1) if len(weights) else np.nan
                )
            for kind in ("country", "commodity", "industry", "factor"):
                subset = own.loc[own.target_node_type.eq(kind)]
                row[f"graph.exposure.{kind}.count"] = float(subset.target_entity_id.nunique())
                row[f"graph.exposure.{kind}.known_weight_count"] = float(
                    subset.exposure_percentage.notna().sum()
                )
            for target in exposure_targets:
                subset = own.loc[own.target_entity_id.eq(target)]
                values = subset.exposure_percentage.dropna()
                # Distinct revenue/production/input-cost exposures are never summed.
                for relation in (
                    "revenue",
                    "production",
                    "supplier",
                    "operating_cost",
                    "raw_material",
                ):
                    specific = subset.loc[subset.relation_type.eq("exposure." + relation)]
                    values = specific.exposure_percentage.dropna()
                    row[f"graph.exposure.{relation}.{target}.percentage"] = (
                        float(values.iloc[0]) if len(values) == 1 else np.nan
                    )
                    row[f"graph.mask.exposure.{relation}.{target}"] = float(len(values) == 1)
            related = incident.loc[incident.relation_type.isin(neighbor_relations)]
            if neighbor_direction == "upstream":
                related = related.loc[related.target_entity_id.eq(entity)]
            elif neighbor_direction == "downstream":
                related = related.loc[related.source_entity_id.eq(entity)]
            neighbors = (set(related.source_entity_id) | set(related.target_entity_id)) - {entity}
            allowed_neighbors = set(
                snapshot.nodes.loc[snapshot.nodes.node_type.isin(neighbor_node_types), "entity_id"]
            )
            neighbors &= allowed_neighbors
            row["graph.neighbors.count"] = float(len(neighbors))
            row["graph.neighbors.entity_known"] = float(entity in known_ids)
            for metric, unit in units.items():
                values, weights, periods, availabilities = {}, {}, [], []
                weight_contexts, provenance = set(), []
                if known_features is not None:
                    candidates = known_features.loc[
                        known_features.entity_id.isin(neighbors)
                        & known_features.metric.eq(metric)
                        & known_features.unit.eq(unit)
                    ]
                    economic = pd.to_datetime(candidates.observation_date, utc=True, format="mixed")
                    candidates = candidates.loc[economic <= decision].copy()
                    candidates["_economic"] = economic.loc[candidates.index]
                    for neighbor, group in candidates.groupby("entity_id"):
                        current = group.loc[group._economic.eq(group._economic.max())]
                        if len(current) != 1 or current.iloc[0].value is None:
                            continue
                        point = current.iloc[0]
                        if age_limit is not None and decision - point.available_at > age_limit:
                            continue
                        if economic_age is not None and decision - point._economic > economic_age:
                            continue
                        values[neighbor] = float(point.value)
                        periods.append(point._economic)
                        availabilities.append(point.available_at)
                        provenance.append(bool(point.point_in_time))
                        links = own.loc[
                            (own.source_entity_id.eq(neighbor) | own.target_entity_id.eq(neighbor))
                            & own.relation_type.isin(neighbor_relations)
                        ]
                        choices = []
                        for edge in links.itertuples():
                            if neighbor_direction == "upstream" and edge.source_role != "customer":
                                continue
                            if (
                                neighbor_direction == "downstream"
                                and edge.source_role != "supplier"
                            ):
                                continue
                            weight = (
                                edge.revenue_dependency_pct
                                if edge.source_role == "supplier"
                                else edge.supply_dependency_pct
                                if edge.source_role == "customer"
                                else edge.exposure_percentage
                                if edge.source_role in ("exposure", "declaration")
                                else None
                            )
                            if pd.notna(weight):
                                choices.append(float(weight) / 100)
                                weight_contexts.add((edge.weight_scope, edge.weight_period))
                        if len(set(choices)) == 1:
                            weights[neighbor] = choices[0]
                mixed = len(set(periods)) > 1
                if mixed and require_same_observation:
                    values, weights = {}, {}
                mixed_weights = len(weight_contexts) > 1
                if mixed_weights:
                    weights = {}
                prefix = f"graph.neighbors.{metric}"
                row[prefix + ".mean"] = float(np.mean(list(values.values()))) if values else np.nan
                for name, function in (("std", np.std), ("min", np.min), ("max", np.max)):
                    row[prefix + "." + name] = (
                        float(function(list(values.values()))) if values else np.nan
                    )
                row[prefix + ".known_weighted_sum"] = (
                    sum(values[key] * weight for key, weight in weights.items())
                    if weights
                    else np.nan
                )
                row[prefix + ".coverage_fraction"] = (
                    len(values) / len(neighbors) if neighbors else np.nan
                )
                row[prefix + ".weighted_coverage_fraction"] = (
                    len(weights) / len(neighbors) if neighbors else np.nan
                )
                row[prefix + ".age_seconds"] = (
                    (decision - min(availabilities)).total_seconds() if values else np.nan
                )
                row[prefix + ".available_at"] = (
                    max(
                        max(availabilities),
                        max(
                            (_instant(v, "edge_available_at") for v in related.available_at),
                            default=max(availabilities),
                        ),
                    )
                    if values
                    else pd.NaT
                )
                row[prefix + ".period_age_days"] = (
                    max((decision.date() - pd.Timestamp(period).date()).days for period in periods)
                    if values
                    else np.nan
                )
                row[f"graph.mask.neighbors.{metric}"] = float(bool(values))
                row[f"graph.quality.neighbors.{metric}.mixed_periods"] = float(mixed)
                row[f"graph.quality.neighbors.{metric}.mixed_weight_scope"] = float(mixed_weights)
                row[f"graph.quality.neighbors.{metric}.point_in_time"] = (
                    float(all(provenance) and incident.point_in_time.all()) if values else np.nan
                )
                if values and not all(provenance):
                    row["graph.quality.point_in_time"] = 0.0
            if "topology" in selected:
                names = (
                    "in_degree",
                    "out_degree",
                    "degree_centrality",
                    "harmonic",
                    "closeness",
                    "pagerank",
                    "pagerank_converged",
                    "pagerank_iterations",
                    "known_node_count",
                    "known_arc_count",
                    "betweenness",
                )
                row.update(
                    {
                        "graph.topology." + name: topology.get(entity, {}).get(name, np.nan)
                        for name in names
                    }
                )
            if "two_hop" in selected:
                two_hop = two_hop_features(snapshot, entity, limits=limits)
                row.update(
                    {
                        name: value if entity in known_ids else np.nan
                        for name, value in two_hop.items()
                    }
                )
            if "dynamics" in selected:
                row.update(dynamics[entity])
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
        timestamp_semantics="explicit_decision",
        partial_network=True,
        neighbor_units=units,
        weight_semantics="known fractions; no renormalization",
    )
    if "channels" in selected:
        for channel, direction, relations in (
            ("suppliers", "upstream", ("supplier", "customer")),
            ("customers", "downstream", ("supplier", "customer")),
            ("competitors", "both", ("competitor",)),
        ):
            frame = compute_graph_features(
                index,
                history,
                entity_ids=entity_ids,
                node_features=node_features,
                node_feature_units=units,
                neighbor_relations=relations,
                neighbor_node_types=neighbor_node_types,
                max_age=max_age,
                node_max_age=node_max_age,
                node_max_observation_age=node_max_observation_age,
                require_same_observation=require_same_observation,
                groups=("neighbors",),
                neighbor_direction=direction,
                limits=limits,
            )
            frame.columns = [
                name.replace("neighbors.", f"neighbors.{channel}.") for name in frame.columns
            ]
            out = out.join(frame)
    prefixes = {
        "degree": ("graph.degree.",),
        "dependency": ("graph.dependency.", "graph.mask.dependency."),
        "exposure": ("graph.exposure.", "graph.mask.exposure."),
        "neighbors": ("graph.neighbors.", "graph.mask.neighbors.", "graph.quality.neighbors."),
        "quality": ("graph.quality.", "graph.evidence_age_seconds", "graph.mask.entity_known"),
        "channels": tuple(
            f"graph.{prefix}neighbors.{channel}."
            for prefix in ("", "mask.", "quality.")
            for channel in ("suppliers", "customers", "competitors")
        ),
        "two_hop": ("graph.two_hop.",),
        "topology": ("graph.topology.",),
        "dynamics": ("graph.dynamics.",),
    }
    keep = tuple(p for group in selected for p in prefixes[group])
    return out.loc[:, [name.startswith(keep) for name in out.columns]]
