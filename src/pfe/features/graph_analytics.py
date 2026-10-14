"""Bounded graph analytics on declared, known histories; no inferred business links."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfe.data.contracts import _instant
from pfe.data.graph import graph_asof


@dataclass(frozen=True)
class GraphFeatureLimits:
    max_nodes: int = 512
    max_edges: int = 8192
    max_operations: int = 2_000_000
    max_epochs: int = 512
    max_decisions: int = 4096
    max_pairs: int = 4096
    max_history_rows: int = 50_000
    max_output_cells: int = 2_000_000

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ValueError("Graph feature budgets must be positive integers")


def adjacency(snapshot, relations=("supplier", "customer"), limits=None):
    limits = limits or GraphFeatureLimits()
    if (
        isinstance(relations, str)
        or not relations
        or any(not isinstance(v, str) or not v for v in relations)
    ):
        raise ValueError("Select explicit graph relations")
    ids = sorted(set(snapshot.nodes.entity_id))
    if len(ids) > limits.max_nodes or len(snapshot.edges) > limits.max_edges:
        raise ValueError("Graph topology exceeds node/edge budget")
    outgoing, incoming = {key: set() for key in ids}, {key: set() for key in ids}
    for edge in snapshot.edges.itertuples():
        a, b = edge.source_entity_id, edge.target_entity_id
        if edge.relation_type in relations and a in outgoing and b in outgoing and a != b:
            outgoing[a].add(b)
            incoming[b].add(a)
    return outgoing, incoming


def neighbor_channels(snapshot, entity):
    upstream, downstream, competitors = set(), set(), set()
    for edge in snapshot.edges.itertuples():
        a, b = edge.source_entity_id, edge.target_entity_id
        if a == b:
            continue
        if edge.relation_type in ("supplier", "customer"):
            if b == entity:
                upstream.add(a)
            if a == entity:
                downstream.add(b)
        elif edge.relation_type == "competitor" and entity in (a, b):
            competitors.add(b if a == entity else a)
    return dict(suppliers=upstream, customers=downstream, competitors=competitors)


def topology_features(
    snapshot,
    *,
    relations=("supplier", "customer"),
    damping=0.85,
    tolerance=1e-10,
    max_iterations=100,
    limits=None,
    node_types=("company",),
):
    """Directed unweighted PageRank, degree, harmonic/closeness and betweenness.

    Duplicate evidence is one arc; dangling PageRank mass is uniform. Centralities
    refer only to selected observed relations and known nodes, never the full economy.
    """
    limits = limits or GraphFeatureLimits()
    if (
        not 0 < damping < 1
        or not 0 < tolerance < 1
        or type(max_iterations) is not int
        or max_iterations < 1
    ):
        raise ValueError("Invalid PageRank parameters")
    from pfe.data.graph import NODE_TYPES, GraphHistory

    if not node_types or isinstance(node_types, str) or set(node_types) - set(NODE_TYPES):
        raise ValueError("Select explicit topology node types")
    selected = GraphHistory(
        snapshot.nodes.loc[snapshot.nodes.node_type.isin(node_types)],
        snapshot.edges,
        snapshot.issues,
    )
    outgoing, incoming = adjacency(selected, relations, limits)
    ids, n = list(outgoing), len(outgoing)
    m = sum(map(len, outgoing.values()))
    if (n + max_iterations) * (n + m) > limits.max_operations:
        raise ValueError("Graph centralities exceed operation budget")
    if not n:
        return {}
    rank = dict.fromkeys(ids, 1 / n)
    converged = False
    for _iteration in range(max_iterations):
        dangling = sum(rank[a] for a in ids if not outgoing[a])
        updated = {b: (1 - damping) / n + damping * dangling / n for b in ids}
        for a in ids:
            for b in sorted(outgoing[a]):
                updated[b] += damping * rank[a] / len(outgoing[a])
        error = sum(abs(updated[a] - rank[a]) for a in ids)
        rank = updated
        if error <= tolerance:
            converged = True
            break
    between, result = dict.fromkeys(ids, 0.0), {}
    for root in ids:
        predecessors = {a: [] for a in ids}
        paths, distance = dict.fromkeys(ids, 0.0), {root: 0}
        paths[root] = 1.0
        queue, order = deque([root]), []
        while queue:
            a = queue.popleft()
            order.append(a)
            for b in sorted(outgoing[a]):
                if b not in distance:
                    distance[b] = distance[a] + 1
                    queue.append(b)
                if distance[b] == distance[a] + 1:
                    paths[b] += paths[a]
                    predecessors[b].append(a)
        reachable = len(distance) - 1
        total_distance = sum(distance.values())
        result[root] = dict(
            in_degree=len(incoming[root]),
            out_degree=len(outgoing[root]),
            degree_centrality=(len(incoming[root]) + len(outgoing[root])) / (n - 1)
            if n > 1
            else 0.0,
            harmonic=sum(1 / d for d in distance.values() if d) / (n - 1) if n > 1 else 0.0,
            closeness=(reachable / total_distance) * (reachable / (n - 1))
            if total_distance and n > 1
            else 0.0,
            pagerank=rank[root] if converged else np.nan,
            pagerank_converged=float(converged),
            pagerank_iterations=_iteration + 1,
            known_node_count=n,
            known_arc_count=m,
        )
        dependency = dict.fromkeys(ids, 0.0)
        for b in reversed(order):
            for a in predecessors[b]:
                dependency[a] += paths[a] / paths[b] * (1 + dependency[b])
            if b != root:
                between[b] += dependency[b]
    for a in ids:
        result[a]["betweenness"] = between[a] / ((n - 1) * (n - 2)) if n > 2 else 0.0
    return result


def _dependency_edges(snapshot, kind):
    field, role = (
        ("supply_dependency_pct", "customer")
        if kind == "suppliers"
        else ("revenue_dependency_pct", "supplier")
    )
    choices = {}
    for edge in snapshot.edges.itertuples():
        if edge.source_role != role:
            continue
        owner = edge.owner_entity_id
        other = edge.source_entity_id if role == "customer" else edge.target_entity_id
        value = getattr(edge, field)
        if pd.notna(value):
            choices.setdefault((owner, other), set()).add(
                (
                    float(value) / 100,
                    None if pd.isna(edge.weight_scope) else edge.weight_scope,
                    None if pd.isna(edge.weight_period) else edge.weight_period,
                )
            )
    return {key: next(iter(values)) for key, values in choices.items() if len(values) == 1}


def two_hop_features(snapshot, entity, *, limits=None):
    """Strict second-hop supply paths and documented path coefficients.

    Products require same known fiscal weight period/scope and same dependency
    kind. They are mathematical path coefficients, NOT consolidated percentages.
    """
    limits = limits or GraphFeatureLimits()
    outgoing, incoming = adjacency(snapshot, limits=limits)
    result = {}
    for kind, links in (("suppliers", incoming), ("customers", outgoing)):
        weights = _dependency_edges(snapshot, kind)
        direct = links.get(entity, set())
        paths, known, products = [], 0, []
        for middle in sorted(direct):
            for target in sorted(links[middle] - direct - {entity}):
                paths.append(target)
                if len(paths) > limits.max_operations:
                    raise ValueError("Two-hop paths exceed operation budget")
                first, second = weights.get((entity, middle)), weights.get((middle, target))
                if first and second and first[1:] == second[1:] and first[2] is not None:
                    products.append(first[0] * second[0])
                    known += 1
        prefix = f"graph.two_hop.{kind}"
        result.update(
            {
                prefix + ".unique_count": len(set(paths)),
                prefix + ".path_count": len(paths),
                prefix + ".known_path_count": known,
                prefix + ".known_path_coefficient_sum": sum(products) if products else np.nan,
                prefix + ".weight_coverage": known / len(paths) if paths else np.nan,
            }
        )
    return result


def compute_graph_pair_features(
    decisions, history, *, pairs, relations=("supplier", "customer"), max_age=None, limits=None
):
    """Explicit pairs only: common neighbors, shared suppliers/customers/exposures."""
    limits = (
        GraphFeatureLimits(**limits) if isinstance(limits, dict) else limits or GraphFeatureLimits()
    )
    pairs = tuple(pairs)
    if any(isinstance(pair, str) for pair in pairs):
        raise ValueError("Graph pairs must contain two entity IDs")
    pairs = tuple(tuple(pair) for pair in pairs)
    if (
        len(pairs) > limits.max_pairs
        or len(set(pairs)) != len(pairs)
        or any(
            len(pair) != 2
            or any(not isinstance(v, str) or not v for v in pair)
            or pair[0] == pair[1]
            for pair in pairs
        )
    ):
        raise ValueError("Provide distinct bounded pairs of entity IDs")
    from pfe.features.graph import decision_grid

    index = decision_grid(decisions, sorted({v for pair in pairs for v in pair}))
    if len(index) > limits.max_decisions:
        raise ValueError("Graph decisions exceed budget")
    if (
        len(history.edges) > limits.max_history_rows
        or len(index) * len(pairs) * 32 > limits.max_output_cells
    ):
        raise ValueError("Graph pair history/output exceeds budget")
    rows = []
    for decision in index:
        snapshot = graph_asof(history, decision, max_age=max_age)
        outgoing, incoming = adjacency(snapshot, relations, limits)
        for left, right in pairs:
            known = left in outgoing and right in outgoing
            a = outgoing.get(left, set()) | incoming.get(left, set())
            b = outgoing.get(right, set()) | incoming.get(right, set())
            common, union = (a & b) - {left, right}, (a | b) - {left, right}
            row = dict(
                decision_at=decision,
                left_entity_id=left,
                right_entity_id=right,
                **{
                    "graph.pair.mask": bool(known),
                    "graph.pair.quality.complete_network": 0.0,
                    "graph.pair.common_neighbors": len(common) if known else np.nan,
                    "graph.pair.jaccard": len(common) / len(union) if known and union else np.nan,
                    "graph.pair.shared_suppliers": len(
                        incoming.get(left, set()) & incoming.get(right, set())
                    )
                    if known
                    else np.nan,
                    "graph.pair.shared_customers": len(
                        outgoing.get(left, set()) & outgoing.get(right, set())
                    )
                    if known
                    else np.nan,
                },
            )
            for kind in ("country", "commodity", "industry", "factor"):
                exposures = snapshot.edges.loc[snapshot.edges.target_node_type.eq(kind)]
                one = set(exposures.loc[exposures.owner_entity_id.eq(left), "target_entity_id"])
                two = set(exposures.loc[exposures.owner_entity_id.eq(right), "target_entity_id"])
                row[f"graph.pair.shared_{kind}"] = len(one & two) if known else np.nan
                row[f"graph.pair.shared_{kind}.jaccard"] = (
                    len(one & two) / len(one | two) if known and one | two else np.nan
                )
                weighted = {}
                for owner in (left, right):
                    selected = exposures.loc[
                        exposures.owner_entity_id.eq(owner) & exposures.exposure_percentage.notna()
                    ]
                    grouped = {}
                    for edge in selected.itertuples():
                        context = (
                            edge.target_entity_id,
                            edge.relation_type,
                            edge.weight_scope,
                            edge.weight_period,
                        )
                        if pd.notna(edge.weight_period):
                            grouped.setdefault(context, set()).add(
                                float(edge.exposure_percentage) / 100
                            )
                    weighted[owner] = {
                        key: next(iter(values))
                        for key, values in grouped.items()
                        if len(values) == 1
                    }
                common_weights = weighted[left].keys() & weighted[right].keys()
                row[f"graph.pair.shared_{kind}.known_weight_overlap"] = (
                    sum(min(weighted[left][key], weighted[right][key]) for key in common_weights)
                    if known and common_weights
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["decision_at", "left_entity_id", "right_entity_id"])


def graph_dynamics(decision, history, *, entity_ids, windows=("30D",), max_age=None, limits=None):
    """Count observed state transitions at knowledge/economic/expiry epochs.

    Expiry/retraction is a disappearance, not a claimed business termination.
    Multiple changes between two output decisions are counted individually.
    """
    limits = limits or GraphFeatureLimits()
    decision = _instant(decision, "decision")
    result = {entity: {} for entity in entity_ids}
    for label in windows:
        duration = pd.Timedelta(label)
        if pd.isna(duration) or duration <= pd.Timedelta(0):
            raise ValueError("Graph dynamics windows must be positive")
        start = decision - duration
        epochs = {decision}
        for edge in history.edges.itertuples():
            known = _instant(edge.available_at, "available_at")
            if known > decision:
                continue
            candidates = [known]
            for field in ("effective_from", "effective_to", "state_date"):
                value = getattr(edge, field)
                if pd.notna(value):
                    candidates.append(max(known, pd.Timestamp(value, tz="UTC")))
            if max_age is not None:
                candidates.append(
                    max(
                        known,
                        _instant(edge.evidence_at, "evidence_at")
                        + pd.Timedelta(max_age)
                        + pd.Timedelta("1ns"),
                    )
                )
            epochs.update(t for t in candidates if start < t <= decision)
        if len(epochs) > limits.max_epochs:
            raise ValueError("Graph dynamics exceeds epoch budget")
        if len(epochs) * max(1, len(history.edges)) > limits.max_operations:
            raise ValueError("Graph dynamics exceeds operation budget")
        counts = {
            entity: dict(new_edges=0, disappeared_edges=0, terminated_edges=0, weight_changes=0)
            for entity in entity_ids
        }

        def state(at):
            snapshot = graph_asof(history, at, max_age=max_age)
            return {row.edge_id: row for row in snapshot.edges.itertuples()}

        previous = state(start)
        for epoch in sorted(epochs):
            current = state(epoch)
            for key in previous.keys() | current.keys():
                before, after = previous.get(key), current.get(key)
                row = after if after is not None else before
                affected = {row.source_entity_id, row.target_entity_id, row.owner_entity_id}
                for entity in affected & counts.keys():
                    if before is None:
                        counts[entity]["new_edges"] += 1
                    elif after is None:
                        counts[entity]["disappeared_edges"] += 1
                        visible = history.edges.loc[
                            history.edges.edge_id.eq(key)
                            & history.edges.available_at.map(
                                lambda v, at=epoch: _instant(v, "available_at") <= at
                            )
                        ]
                        visible = visible.sort_values(
                            "available_at", kind="stable"
                        ).drop_duplicates(["edge_id", "fact_id"], keep="last")
                        if len(visible):
                            visible = visible.loc[visible.state_date.eq(visible.state_date.max())]
                            visible = visible.loc[visible.evidence_at.eq(visible.evidence_at.max())]
                        end = before.effective_to
                        explicit_end = pd.notna(end) and pd.Timestamp(end, tz="UTC") <= epoch
                        termination = (
                            visible.state.eq("terminated")
                            & visible.review_status.isin(("approved", "source_qualified"))
                            & (visible.state_date <= epoch.date().isoformat())
                        )
                        if explicit_end or termination.any():
                            counts[entity]["terminated_edges"] += 1
                    else:
                        fields = (
                            "revenue_dependency_pct",
                            "supply_dependency_pct",
                            "exposure_percentage",
                            "weight_scope",
                            "weight_period",
                        )

                        def values(item, selected_fields=fields):
                            return tuple(
                                None if pd.isna(getattr(item, field)) else getattr(item, field)
                                for field in selected_fields
                            )

                        if values(before) != values(after):
                            counts[entity]["weight_changes"] += 1
            previous = current
        for entity in entity_ids:
            result[entity].update(
                {f"graph.dynamics.{name}_w{label}": value for name, value in counts[entity].items()}
            )
    return result
