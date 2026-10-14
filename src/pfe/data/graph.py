"""Declared, partial business graphs with economic validity and knowledge history."""

import json
from dataclasses import dataclass
from datetime import date

import pandas as pd

from pfe.data.contracts import _instant, _provenance, normalize_observations
from pfe.data.documents import content_hash
from pfe.data.storage import _json_bytes

NODE_TYPES = ("company", "country", "industry", "commodity", "factor")


class EntityRegistry:
    """Explicit aliases only; no fuzzy matching, ticker guessing or global crosswalk.

    Records require alias/entity_id/node_type/available_at. Optional valid_from/to
    are economic ISO dates, [from,to). A relation binds at its explicit effective
    start, otherwise its knowledge date. Later identity mappings delay availability.
    An ambiguous earliest mapping is unresolved and is recorded in build issues.
    """

    def __init__(self, records):
        frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        required = {"alias", "entity_id", "node_type", "available_at"}
        if frame.empty and not len(frame.columns):
            frame = pd.DataFrame(columns=sorted(required))
        if not required.issubset(frame):
            raise ValueError(f"Entity aliases require {sorted(required)}")
        for name in ("alias", "entity_id"):
            if not frame[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError("Aliases and entity IDs must be nonempty strings")
        if not frame.node_type.isin(NODE_TYPES).all():
            raise ValueError("Invalid graph node_type")
        frame["available_at"] = frame.available_at.map(lambda v: _instant(v, "available_at"))
        _provenance(frame)
        for name in ("valid_from", "valid_to"):
            if name not in frame:
                frame[name] = None
            frame[name] = frame[name].where(frame[name].notna(), None)
            for value in frame[name].dropna():
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError("Identity validity requires ISO dates")
        if (
            (frame.valid_from.notna() & frame.valid_to.notna())
            & (frame.valid_from >= frame.valid_to)
        ).any():
            raise ValueError("Identity interval must be nonempty")
        for _, group in frame.groupby("entity_id"):
            if group.node_type.nunique() != 1:
                raise ValueError("An entity_id cannot change node_type")
        self.frame = frame

    def binding(self, alias, *, node_type="company", observation_date):
        if alias is None:
            return None
        found = self.frame.loc[
            self.frame.alias.str.casefold().eq(alias.casefold())
            & self.frame.node_type.eq(node_type)
            & (self.frame.valid_from.isna() | (self.frame.valid_from <= observation_date))
            & (self.frame.valid_to.isna() | (self.frame.valid_to > observation_date))
        ]
        if found.empty:
            return None
        first = found.loc[found.available_at == found.available_at.min()]
        if first.entity_id.nunique() != 1:
            return None
        return first.iloc[0].to_dict()


def fact_history(facts):
    """Validate availability of accepted/reviewed facts before any downstream join."""
    rows = []
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("record"), dict):
            raise ValueError("Expected fact envelopes with record dictionaries")
        for key in (
            "fact_id",
            "task",
            "source_document_id",
            "text_sha256",
            "source_id",
            "revision_id",
        ):
            if not isinstance(fact.get(key), str) or not fact[key]:
                raise ValueError(f"Fact requires {key}")
        from pfe.extraction.schemas import ENVELOPES, extraction_schema
        from pfe.extraction.validation import _check

        _check(
            fact["record"],
            extraction_schema(fact["task"])["properties"][ENVELOPES[fact["task"]]]["items"],
        )
        expected_id = "fact:" + content_hash(
            _json_bytes({"task": fact["task"], "record": fact["record"]})
        )
        if fact["fact_id"] != expected_id:
            raise ValueError("Fact identity does not match its immutable record")
        if fact["record"].get("source_document_id") != fact["source_document_id"]:
            raise ValueError("Fact source_document_id mismatch")
        if fact.get("review_status") not in ("unreviewed", "approved", "rejected"):
            raise ValueError("Invalid fact review_status")
        at = _instant(fact["available_at"], "available_at")
        for name in ("document_available_at", "processed_at", "reviewed_at", "published_at"):
            if fact.get(name) is not None and _instant(fact[name], name) > at:
                raise ValueError("Fact availability precedes its evidence, processing or review")
        if fact["review_status"] in ("approved", "rejected"):
            if any(
                not fact.get(name)
                for name in ("review_id", "reviewer", "review_notes", "reviewed_at")
            ):
                raise ValueError("Reviewed facts require a named review and review_id")
        for name in ("effective_from", "effective_to"):
            if (
                fact.get(name) is not None
                and date.fromisoformat(fact[name]).isoformat() != fact[name]
            ):
                raise ValueError("Economic validity requires ISO dates")
        rows.append({**fact, "available_at": at.isoformat()})
    versions = {}
    for row in rows:
        key = (row["fact_id"], row["available_at"])
        serialized = _json_bytes(row)
        if key in versions and versions[key] != serialized:
            raise ValueError("Conflicting simultaneous fact versions")
        versions[key] = serialized
    return sorted(rows, key=lambda row: row["available_at"])


def facts_asof(facts, decision_at, *, require_review=True):
    decision = _instant(decision_at, "decision_at")
    latest = {}
    for fact in fact_history(facts):
        if _instant(fact["available_at"], "available_at") <= decision:
            latest[fact["fact_id"]] = fact
    return [
        fact
        for fact in latest.values()
        if fact["review_status"] == "approved"
        or not require_review
        and fact["review_status"] == "unreviewed"
    ]


@dataclass
class GraphHistory:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    issues: pd.DataFrame


EDGE_COLUMNS = [
    "edge_id",
    "fact_id",
    "source_entity_id",
    "target_entity_id",
    "relation_type",
    "source_role",
    "owner_entity_id",
    "target_node_type",
    "available_at",
    "effective_from",
    "effective_to",
    "active",
    "revenue_dependency_pct",
    "supply_dependency_pct",
    "exposure_percentage",
    "record_json",
    "source_document_id",
    "evidence_quote",
    "source_id",
    "revision_id",
    "published_at",
    "point_in_time",
    "review_status",
    "state",
    "weight_scope",
    "weight_period",
    "evidence_at",
    "state_date",
]
NODE_COLUMNS = ["entity_id", "node_type", "label", "available_at"]
ISSUE_COLUMNS = ["fact_id", "reason", "alias", "available_at"]


def issues_asof(issues, decision):
    return issues.loc[
        issues.available_at.map(lambda v: _instant(v, "available_at")) <= decision
    ].copy()


def knowledge_frame(frame, *, entity_column, record_column, schema_version, economic_column=None):
    """Seven shared fields for structured histories; unknown economic dates stay null.

    Variable A-D records are encoded inside a uniform structured value to keep
    Parquet types stable. Source revisions remain separate from derived versions.
    These histories have their own schemas and allow unknown observation_date;
    they are not scalar observations for normalize_observations.
    """
    out = frame.copy()
    out["entity_id"] = out[entity_column]
    out["observation_date"] = out[economic_column] if economic_column else None
    if "published_at" not in out:
        out["published_at"] = None
    if "source_id" not in out:
        out["source_id"] = "explicit_entity_registry"
    if "value" not in out:
        out["value"] = pd.Series(
            [
                {
                    "record_json": row.get(
                        "record_json",
                        _json_bytes(
                            {
                                key: row[key]
                                for key in ("entity_id", "node_type", "label")
                                if key in row
                            }
                        ).decode(),
                    )
                }
                for row in out.to_dict("records")
            ],
            index=out.index,
            dtype=object,
        )
    out["source_revision_id"] = out["revision_id"] if "revision_id" in out else None
    out["revision_id"] = [
        schema_version
        + ":"
        + content_hash(
            _json_bytes(
                {
                    "record": row[record_column],
                    "available_at": row["available_at"],
                    "fact_id": row.get("fact_id"),
                    "source_revision": row.get("source_revision_id"),
                }
            )
        )
        for row in out.to_dict("records")
    ]
    out["schema_version"] = schema_version
    return out


def _scalar(value):
    return value if type(value) in (int, float) else None


def build_graph_history(facts, entities, *, require_review=True):
    """Build oriented supply flows and country/material exposures with evidence.

    A supplier of B -> A->B; A customer of B -> B->A. Other relations preserve
    stated source->related direction. Reference memberships/factor exposures can
    be appended with merge_graph_declarations using qualified observations.
    Unknown entity IDs stay in issues; dependency ranges remain record_json.
    """
    registry = entities if isinstance(entities, EntityRegistry) else EntityRegistry(entities)
    edges, issues = [], []
    nodes = [
        {
            "entity_id": row.entity_id,
            "node_type": row.node_type,
            "label": row.alias,
            "available_at": row.available_at.isoformat(),
        }
        for row in registry.frame.itertuples()
    ]
    for fact in fact_history(facts):
        task, record = fact["task"], fact["record"]
        if task not in ("A", "B"):
            continue
        observed = fact.get("effective_from") or fact["available_at"][:10]
        source_label = record.get("source_company") if task == "A" else record.get("company")
        source = registry.binding(source_label, observation_date=observed)
        if source is None:
            issues.append(
                {
                    "fact_id": fact["fact_id"],
                    "reason": "unresolved source entity",
                    "alias": source_label,
                    "available_at": fact["available_at"],
                }
            )
            continue
        active = (
            fact["review_status"] == "approved"
            or not require_review
            and fact["review_status"] == "unreviewed"
        )
        targets = []
        if task == "A":
            related = registry.binding(record.get("related_entity"), observation_date=observed)
            if related is None:
                issues.append(
                    {
                        "fact_id": fact["fact_id"],
                        "reason": "unresolved related entity",
                        "alias": record.get("related_entity"),
                        "available_at": fact["available_at"],
                    }
                )
                continue
            active &= (
                record["certainty"] == "confirmed" and record["effective_status"] == "existing"
            )
            targets.append((related, record["relation_type"], record["relation_type"]))
        else:
            for label, kind in (
                (record["country_or_region"], "country"),
                (
                    record["product_or_material"]
                    if record["exposure_type"] == "raw_material"
                    else None,
                    "commodity",
                ),
            ):
                if label is not None:
                    target = registry.binding(label, node_type=kind, observation_date=observed)
                    if target is None:
                        target = {
                            "entity_id": f"{kind}:" + content_hash(label.encode()),
                            "node_type": kind,
                            "alias": label,
                            "available_at": fact["available_at"],
                            "point_in_time": fact["point_in_time"],
                        }
                    targets.append((target, "exposure." + record["exposure_type"], "exposure"))
        for target, relation, role in targets:
            at = max(
                _instant(fact["available_at"], "available_at"),
                _instant(source["available_at"], "available_at"),
                _instant(target["available_at"], "available_at"),
            ).isoformat()
            start, end = source["entity_id"], target["entity_id"]
            if task == "A" and role == "customer":
                start, end = end, start
            edge_key = {
                "source": start,
                "target": end,
                "relation": relation,
                "owner": source["entity_id"],
                "target_node_type": target["node_type"],
                "product_scope": record.get("products_or_services")
                if task == "A"
                else record.get("product_or_material"),
            }
            edges.append(
                {
                    "edge_id": "edge:" + content_hash(_json_bytes(edge_key)),
                    "fact_id": fact["fact_id"],
                    "source_entity_id": start,
                    "target_entity_id": end,
                    "relation_type": relation,
                    "source_role": role,
                    "owner_entity_id": source["entity_id"],
                    "target_node_type": target["node_type"],
                    "available_at": at,
                    "evidence_at": _instant(
                        fact["published_at"] or fact["document_available_at"], "evidence_at"
                    ).isoformat(),
                    "state_date": fact.get("effective_from")
                    or _instant(
                        fact["published_at"] or fact["document_available_at"], "evidence_at"
                    )
                    .date()
                    .isoformat(),
                    "effective_from": fact.get("effective_from"),
                    "state": record["effective_status"] if task == "A" else "existing",
                    "weight_scope": edge_key["product_scope"],
                    "weight_period": record.get("effective_period")
                    if task == "A"
                    else record.get("fiscal_period"),
                    "effective_to": fact.get("effective_to"),
                    "active": bool(active),
                    "revenue_dependency_pct": _scalar(record.get("revenue_dependency_pct")),
                    "supply_dependency_pct": _scalar(record.get("supply_dependency_pct")),
                    "exposure_percentage": _scalar(record.get("exposure_percentage")),
                    "record_json": _json_bytes(record).decode(),
                    **{
                        key: fact.get(key)
                        for key in (
                            "source_document_id",
                            "source_id",
                            "revision_id",
                            "published_at",
                            "point_in_time",
                            "review_status",
                        )
                    },
                    "evidence_quote": record["evidence_quote"],
                }
            )
            edges[-1]["point_in_time"] = bool(
                fact["point_in_time"] and source["point_in_time"] and target["point_in_time"]
            )
            for node in (source, target):
                nodes.append(
                    {
                        "entity_id": node["entity_id"],
                        "node_type": node["node_type"],
                        "label": node["alias"],
                        "available_at": at,
                    }
                )
    node_frame = pd.DataFrame(nodes, columns=NODE_COLUMNS)
    if not node_frame.empty:
        node_frame = node_frame.sort_values("available_at", kind="stable").drop_duplicates(
            "entity_id"
        )
    return GraphHistory(
        knowledge_frame(
            node_frame,
            entity_column="entity_id",
            record_column="entity_id",
            schema_version="graph-nodes-1",
        ),
        knowledge_frame(
            pd.DataFrame(edges, columns=EDGE_COLUMNS),
            entity_column="owner_entity_id",
            record_column="edge_id",
            schema_version="graph-edges-1",
            economic_column="effective_from",
        ),
        pd.DataFrame(issues, columns=ISSUE_COLUMNS),
    )


def merge_graph_declarations(history, declarations):
    """Append deterministic reference memberships/exposures in Lot 0 contracts.

    Required value: target_entity_id, relation_type, active, weight_pct (nullable).
    Relations: industry_member/factor_exposure/country_exposure/commodity_exposure.
    metric must be ``graph.<relation_type>.<target_entity_id>`` to partition edges.
    unit is ``relationship``; source_document_id and evidence_quote trace the source
    table row. Caller qualifies timestamps/rights; no LLM approval is fabricated.
    observation_date is an explicit economic start, not an inferred filing date.
    """
    out = normalize_observations(declarations)
    rows = []
    for point in out.to_dict("records"):
        value = point["value"]
        required = {"target_entity_id", "relation_type", "active", "weight_pct"}
        if (
            not isinstance(value, dict)
            or set(value) != required
            or type(value["active"]) is not bool
        ):
            raise ValueError("Invalid declared graph relationship value")
        relation, target = value["relation_type"], value["target_entity_id"]
        kinds = {
            "industry_member": "industry",
            "factor_exposure": "factor",
            "country_exposure": "country",
            "commodity_exposure": "commodity",
        }
        if (
            relation not in kinds
            or point["metric"] != f"graph.{relation}.{target}"
            or point["unit"] != "relationship"
        ):
            raise ValueError("Declared graph metric/unit/relationship mismatch")
        weight = value["weight_pct"]
        if weight is not None and (type(weight) not in (int, float) or not 0 <= weight <= 100):
            raise ValueError("Declared weights require percentages in [0,100] or null")
        if (
            not point["source_document_id"]
            or not isinstance(point.get("evidence_quote"), str)
            or not point["evidence_quote"].strip()
        ):
            raise ValueError("Declared edges require source_document_id and evidence_quote")
        source_node = history.nodes.loc[history.nodes.entity_id.eq(point["entity_id"])]
        target_node = history.nodes.loc[history.nodes.entity_id.eq(target)]
        if (
            len(source_node) != 1
            or len(target_node) != 1
            or target_node.iloc[0].node_type != kinds[relation]
        ):
            raise ValueError("Declared graph nodes must exist with the expected target type")
        at = max(
            point["available_at"],
            _instant(source_node.iloc[0].available_at, "available_at"),
            _instant(target_node.iloc[0].available_at, "available_at"),
        ).isoformat()
        if len(point["observation_date"]) != 10:
            raise ValueError("Declared graph starts require date-only economic periods")
        end = point.get("effective_to")
        if end is not None and (
            date.fromisoformat(end).isoformat() != end or end <= point["observation_date"]
        ):
            raise ValueError("Declared graph interval must be nonempty ISO [from,to)")
        key = {
            "source": point["entity_id"],
            "target": target,
            "relation": relation,
            "owner": point["entity_id"],
            "target_node_type": kinds[relation],
            "product_scope": None,
        }
        record = {
            "value": value,
            "source_id": point["source_id"],
            "revision_id": point["revision_id"],
            "observation_date": point["observation_date"],
            "available_at": at,
        }
        rows.append(
            {
                "edge_id": "edge:" + content_hash(_json_bytes(key)),
                "fact_id": "declaration:" + content_hash(_json_bytes(record)),
                "source_entity_id": point["entity_id"],
                "target_entity_id": target,
                "relation_type": relation,
                "source_role": "declaration",
                "owner_entity_id": point["entity_id"],
                "target_node_type": kinds[relation],
                "available_at": at,
                "evidence_at": (
                    point["published_at"]
                    if pd.notna(point["published_at"])
                    else point["available_at"]
                ).isoformat(),
                "state_date": point["observation_date"],
                "effective_from": point["observation_date"],
                "effective_to": point.get("effective_to"),
                "active": value["active"],
                "state": "existing" if value["active"] else "terminated",
                "weight_scope": None,
                "weight_period": point["observation_date"],
                "revenue_dependency_pct": None,
                "supply_dependency_pct": None,
                "exposure_percentage": weight,
                "record_json": _json_bytes(record).decode(),
                "review_status": "source_qualified",
                "source_document_id": point["source_document_id"],
                "evidence_quote": point["evidence_quote"],
                "point_in_time": point["point_in_time"],
                "source_id": point["source_id"],
                "revision_id": point["revision_id"],
                "published_at": None
                if pd.isna(point["published_at"])
                else point["published_at"].isoformat(),
            }
        )
    added = knowledge_frame(
        pd.DataFrame(rows, columns=EDGE_COLUMNS),
        entity_column="owner_entity_id",
        record_column="edge_id",
        schema_version="graph-edges-1",
        economic_column="effective_from",
    )
    combined = pd.DataFrame(
        [*history.edges.to_dict("records"), *added.to_dict("records")],
        columns=history.edges.columns,
    )
    return GraphHistory(history.nodes.copy(), combined, history.issues.copy())


def graph_asof(history, decision_at, *, observation_date=None, max_age=None):
    """Snapshot latest known evidence, applying economic [from,to) and explicit TTL."""
    decision = _instant(decision_at, "decision_at")
    economic = observation_date or decision.date().isoformat()
    age = pd.Timedelta(max_age) if max_age is not None else None
    if age is not None and (pd.isna(age) or age < pd.Timedelta(0)):
        raise ValueError("max_age must be a finite nonnegative duration")
    if date.fromisoformat(economic).isoformat() != economic:
        raise ValueError("observation_date must be an ISO date")
    if economic > decision.date().isoformat():
        raise ValueError("Graph economic snapshot cannot be in the future")
    edges = history.edges.copy()
    issues = issues_asof(history.issues, decision)
    if edges.empty:
        nodes = history.nodes.loc[
            history.nodes.available_at.map(lambda v: _instant(v, "available_at")) <= decision
        ]
        return GraphHistory(nodes.copy(), edges, issues)
    available = edges.available_at.map(lambda value: _instant(value, "available_at"))
    edges = edges.loc[
        (available <= decision)
        & (edges.effective_from.isna() | (edges.effective_from <= economic))
        & (edges.effective_from.notna() | (edges.state_date <= economic))
    ]
    edges = edges.sort_values(["available_at", "fact_id"], kind="stable")
    edges = edges.drop_duplicates(["edge_id", "fact_id"], keep="last")
    # Rejected/planned/uncertain evidence cannot erase a different approved fact.
    edges = edges.loc[
        edges.active
        | (
            edges.state.eq("terminated")
            & edges.review_status.isin(("approved", "source_qualified"))
        )
    ]
    selected = []
    for _, group in edges.groupby("edge_id", sort=False):
        state = group.loc[group.state_date.eq(group.state_date.max())]
        latest = state.loc[state.evidence_at.eq(state.evidence_at.max())]
        row = latest.iloc[0].copy()
        row["supporting_fact_ids"] = sorted(set(latest.fact_id))
        row["supporting_document_ids"] = sorted(set(latest.source_document_id))
        row["ambiguous_weights"] = False
        for field in ("revenue_dependency_pct", "supply_dependency_pct", "exposure_percentage"):
            if latest[field].nunique(dropna=False) > 1:
                row[field] = None
                row["ambiguous_weights"] = True
        if latest.active.nunique() > 1:
            row["active"] = False
        row["point_in_time"] = bool(latest.point_in_time.all())
        selected.append(row)
    edges = pd.DataFrame(selected) if selected else edges.iloc[:0].copy()
    edges = edges.loc[edges.active & (edges.effective_to.isna() | (edges.effective_to > economic))]
    if max_age is not None:
        known = edges.evidence_at.map(lambda value: _instant(value, "evidence_at"))
        edges = edges.loc[decision - known <= age]
    nodes = history.nodes.copy()
    nodes = nodes.loc[nodes.available_at.map(lambda v: _instant(v, "available_at")) <= decision]
    return GraphHistory(nodes.reset_index(drop=True), edges.reset_index(drop=True), issues)


def statements_frame(facts, entities, *, require_review=True):
    """All C fields, ranges and classifications; no synthetic consensus or midpoint."""
    registry = entities if isinstance(entities, EntityRegistry) else EntityRegistry(entities)
    rows = []
    for fact in fact_history(facts):
        if fact["task"] != "C":
            continue
        if require_review and fact["review_status"] == "unreviewed":
            continue
        record = fact["record"]
        entity = registry.binding(record["company"], observation_date=fact["available_at"][:10])
        if entity is None:
            continue
        at = max(
            _instant(fact["available_at"], "available_at"),
            _instant(entity["available_at"], "available_at"),
        ).isoformat()
        rows.append(
            {
                **record,
                **{
                    key: fact.get(key)
                    for key in (
                        "fact_id",
                        "source_id",
                        "revision_id",
                        "published_at",
                        "review_status",
                        "point_in_time",
                    )
                },
                "entity_id": entity["entity_id"],
                "point_in_time": bool(fact["point_in_time"] and entity["point_in_time"]),
                "observation_date": record["fiscal_period_end"],
                "available_at": at,
                "statement_origin_at": _instant(
                    fact["published_at"] or fact["document_available_at"], "statement_origin_at"
                ).isoformat(),
                "value_json": json.dumps(record),
                "schema_version": "statements-1",
            }
        )
    return (
        knowledge_frame(
            pd.DataFrame(rows),
            entity_column="entity_id",
            record_column="fact_id",
            schema_version="statements-1",
            economic_column="fiscal_period_end",
        )
        if rows
        else pd.DataFrame(
            columns=[
                "entity_id",
                "available_at",
                "fact_id",
                "metric",
                "fiscal_period_start",
                "fiscal_period_label",
                "statement_type",
                "fiscal_period_end",
                "value",
                "lower_bound",
                "upper_bound",
                "currency",
                "unit_scale",
                "review_status",
                "point_in_time",
                "segment_name",
                "observation_date",
                "published_at",
                "source_id",
                "revision_id",
                "source_revision_id",
                "schema_version",
                "statement_origin_at",
            ]
        )
    )
