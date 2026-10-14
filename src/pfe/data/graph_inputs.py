"""Explicit security bindings and documented structured graph inputs, without LLMs."""

import hashlib
import json
from datetime import date

import pandas as pd

from pfe.data.contracts import _instant, normalize_observations
from pfe.data.graph import (
    EDGE_COLUMNS,
    ISSUE_COLUMNS,
    NODE_COLUMNS,
    EntityRegistry,
    GraphHistory,
    knowledge_frame,
)
from pfe.data.security_master import normalize_security_master, security_master_asof
from pfe.data.storage import _json_bytes, _reject_secrets
from pfe.extraction.schemas import RELATIONS


def entity_registry_from_security_master(master, decision_at, *, source_id=None, aliases=()):
    """Registry of known issuer IDs/tickers/instruments/CIKs at an explicit snapshot.

    No parent/subsidiary or ADR merger is inferred. Extra aliases require their
    own entity_id and availability. Same-day historical symbol transitions should
    use snapshot-specific registries: EntityRegistry economic bounds are dates.
    """
    known = security_master_asof(master, decision_at)
    if source_id is not None:
        known = known.loc[known.source_id.eq(source_id)]
    rows = []
    for point in known.itertuples():
        labels = {point.entity_id, point.instrument_id, point.ticker}
        if point.cik is not None:
            labels.add("cik:" + point.cik)
        for alias in sorted(labels):
            rows.append(
                dict(
                    alias=alias,
                    entity_id=point.entity_id,
                    node_type="company",
                    available_at=point.available_at.isoformat(),
                    point_in_time=bool(point.point_in_time),
                )
            )
    extra = aliases.to_dict("records") if isinstance(aliases, pd.DataFrame) else list(aliases)
    records = [
        dict(availability_basis="unknown", point_in_time=False) | record for record in rows + extra
    ]
    return EntityRegistry(records)


def bind_graph_observations(
    observations,
    master,
    decision_at,
    *,
    identifier_kind="instrument_id",
    identifier_column="entity_id",
    source_id=None,
    venue=None,
):
    """Map explicit instrument/ticker/issuer identifiers, retaining mapping latency.

    Only data known at decision are emitted. Ambiguous/missing bindings fail;
    multiple share classes stay separate instruments and aren't averaged.
    """
    if identifier_kind not in ("instrument_id", "ticker", "entity_id"):
        raise ValueError("Choose explicit graph identifier kind")
    master = normalize_security_master(master)
    decision = _instant(decision_at, "decision_at")
    data = normalize_observations(observations)
    if identifier_column not in data:
        raise ValueError("Missing graph identifier column")
    rows = []
    for point in data.loc[data.available_at <= decision].to_dict("records"):
        economic = pd.Timestamp(point["observation_date"])
        economic = (
            economic.tz_localize("UTC") if economic.tzinfo is None else economic.tz_convert("UTC")
        )
        if economic > decision:
            continue
        known = security_master_asof(master, decision, effective_at=economic)
        if source_id is not None:
            known = known.loc[known.source_id.eq(source_id)]
        found = known.loc[known[identifier_kind].eq(point[identifier_column])]
        if identifier_kind == "ticker":
            selected_venue = point.get("venue", venue)
            if not selected_venue:
                raise ValueError("Ticker binding requires an explicit venue")
            found = found.loc[found.venue.eq(selected_venue)]
        if len(found[["entity_id", "instrument_id", "currency"]].drop_duplicates()) != 1:
            raise ValueError("Missing or ambiguous graph security binding")
        binding = found.sort_values("available_at").iloc[-1]
        point["source_entity_id"] = point["entity_id"]
        point["entity_id"], point["instrument_id"] = binding.entity_id, binding.instrument_id
        point["available_at"] = max(point["available_at"], binding.available_at)
        point["source_revision_id"] = point["revision_id"]
        point["binding_revision_id"] = binding.revision_id
        point["revision_id"] = hashlib.sha256(
            _json_bytes(
                dict(
                    source=str(point["revision_id"]),
                    mapping=str(binding.revision_id),
                    instrument=binding.instrument_id,
                )
            )
        ).hexdigest()
        point["point_in_time"] = bool(point["point_in_time"] and binding.point_in_time)
        rows.append(point)
    return normalize_observations(pd.DataFrame(rows)) if rows else data.iloc[:0].copy()


def import_graph_relations(relations, entities, *, max_rows=20_000):
    """Import caller-qualified structured relations with stable IDs and evidence.

    Required: relationship_id, company_id, related_entity_id, relation_type,
    source_id, revision_id, available_at, source_document_id, evidence_quote,
    qualification_reference. Direction describes the role of company_id:
    supplier -> company supplies related; customer -> company buys from related.
    Percentages describe company_id's revenue/supply/exposure, never its peer's.
    Existing/terminated versions share relationship_id within source_id.
    """
    frame = relations.copy() if isinstance(relations, pd.DataFrame) else pd.DataFrame(relations)
    required = {
        "relationship_id",
        "company_id",
        "related_entity_id",
        "relation_type",
        "source_id",
        "revision_id",
        "available_at",
        "source_document_id",
        "evidence_quote",
        "qualification_reference",
    }
    if frame.empty and not len(frame.columns):
        frame = pd.DataFrame(columns=sorted(required))
    if (
        not required <= set(frame)
        or type(max_rows) is not int
        or max_rows < 1
        or len(frame) > max_rows
    ):
        raise ValueError("Structured graph input needs bounded qualified relation records")
    registry = entities if isinstance(entities, EntityRegistry) else EntityRegistry(entities)
    _reject_secrets(frame.to_dict("records"))
    rows, versions, revisions = [], {}, {}
    frame = frame.assign(
        available_at=frame.available_at.map(lambda v: _instant(v, "available_at"))
    ).sort_values("available_at", kind="stable")
    nodes = [
        dict(
            entity_id=r.entity_id,
            node_type=r.node_type,
            label=r.alias,
            available_at=r.available_at.isoformat(),
        )
        for r in registry.frame.itertuples()
    ]
    allowed = (
        set(RELATIONS)
        | {
            "exposure." + kind
            for kind in ("revenue", "production", "supplier", "operating_cost", "raw_material")
        }
        | {"industry_member", "factor_exposure", "country_exposure", "commodity_exposure"}
    )
    for original in frame.to_dict("records"):
        record = {
            key: None if not isinstance(value, (dict, list)) and pd.isna(value) else value
            for key, value in original.items()
        }
        if record.get("state") is None:
            record["state"] = "existing"
        if record.get("point_in_time") is None:
            record["point_in_time"] = False
        if any(
            not isinstance(record[key], str) or not record[key].strip()
            for key in required - {"available_at"}
        ):
            raise ValueError("Structured relation IDs and evidence must be nonempty strings")
        relation = record["relation_type"]
        if relation not in allowed or record.get("state", "existing") not in (
            "existing",
            "terminated",
        ):
            raise ValueError("Unknown structured graph relation/state")
        available = _instant(record["available_at"], "available_at")
        evidence = _instant(
            record.get("published_at") or record.get("evidence_at") or available, "evidence_at"
        )
        if evidence > available:
            raise ValueError("Graph relation precedes its evidence")
        for field in ("effective_from", "effective_to"):
            if (
                record.get(field) is not None
                and date.fromisoformat(record[field]).isoformat() != record[field]
            ):
                raise ValueError("Graph economic bounds must be ISO dates")
        if (
            record.get("effective_from")
            and record.get("effective_to")
            and record["effective_from"] >= record["effective_to"]
        ):
            raise ValueError("Invalid graph economic interval")
        bindings = []
        for entity in (record["company_id"], record["related_entity_id"]):
            found = registry.frame.loc[registry.frame.entity_id.eq(entity)]
            if found.empty:
                raise ValueError("Unresolved structured graph entity")
            first = found.sort_values("available_at").iloc[0]
            available = max(available, first.available_at)
            bindings.append(first)
            nodes.append(
                dict(
                    entity_id=entity,
                    node_type=first.node_type,
                    label=first.alias,
                    available_at=first.available_at.isoformat(),
                )
            )
        if (
            bindings[0].node_type != "company"
            or relation in RELATIONS
            and bindings[1].node_type != "company"
        ):
            raise ValueError("Business relations require company nodes")
        point_in_time = record.get("point_in_time", False)
        target_types = {
            "industry_member": "industry",
            "country_exposure": "country",
            "commodity_exposure": "commodity",
            "factor_exposure": "factor",
        }
        if relation in target_types and bindings[1].node_type != target_types[relation]:
            raise ValueError("Declared graph exposure has an incompatible target node type")
        if type(point_in_time) is not bool:
            raise ValueError("Graph point_in_time requires a boolean")
        percentages = {}
        for field in ("revenue_dependency_pct", "supply_dependency_pct", "exposure_percentage"):
            value = record.get(field)
            if value is not None and (type(value) not in (int, float) or not 0 <= value <= 100):
                raise ValueError("Graph percentages must be finite numbers in [0,100] or null")
            percentages[field] = value
        owner, peer = record["company_id"], record["related_entity_id"]
        if owner == peer:
            raise ValueError("Structured business relations cannot be self loops")
        start, end = (peer, owner) if relation == "customer" else (owner, peer)
        edge_id = (
            "edge:"
            + hashlib.sha256(
                _json_bytes([record["source_id"], record["relationship_id"]])
            ).hexdigest()
        )
        encoded = json.dumps(record, sort_keys=True, default=str, allow_nan=False)
        revision_key = (edge_id, record["revision_id"])
        semantic = json.dumps(
            {
                key: value
                for key, value in record.items()
                if key not in ("available_at", "ingested_at")
            },
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        if revision_key in revisions:
            if revisions[revision_key] != semantic:
                raise ValueError("Graph revision changed its immutable contents")
            continue
        revisions[revision_key] = semantic
        row = dict(
            edge_id=edge_id,
            fact_id="structured:" + edge_id.removeprefix("edge:"),
            source_entity_id=start,
            target_entity_id=end,
            relation_type=relation,
            source_role=relation if relation in RELATIONS else "exposure",
            owner_entity_id=owner,
            target_node_type=bindings[1].node_type,
            available_at=available.isoformat(),
            effective_from=record.get("effective_from"),
            effective_to=record.get("effective_to"),
            active=record.get("state", "existing") == "existing",
            state=record.get("state", "existing"),
            record_json=encoded,
            source_document_id=record["source_document_id"],
            evidence_quote=record["evidence_quote"],
            source_id=record["source_id"],
            revision_id=record["revision_id"],
            published_at=record.get("published_at"),
            point_in_time=bool(point_in_time and all(bool(b.point_in_time) for b in bindings)),
            review_status="source_qualified",
            weight_scope=record.get("weight_scope"),
            weight_period=record.get("weight_period"),
            evidence_at=evidence.isoformat(),
            state_date=record.get("effective_from") or evidence.date().isoformat(),
            **percentages,
        )
        key = (edge_id, available.isoformat())
        if key in versions and versions[key] != row:
            raise ValueError("Conflicting simultaneous graph versions")
        versions[key] = row
        rows.append(row)
    nodes = pd.DataFrame(nodes, columns=NODE_COLUMNS).drop_duplicates("entity_id")
    return GraphHistory(
        knowledge_frame(
            nodes,
            entity_column="entity_id",
            record_column="entity_id",
            schema_version="graph-nodes-1",
        ),
        knowledge_frame(
            pd.DataFrame(rows, columns=EDGE_COLUMNS).drop_duplicates(["edge_id", "available_at"]),
            entity_column="owner_entity_id",
            record_column="edge_id",
            schema_version="graph-edges-1",
            economic_column="effective_from",
        ),
        pd.DataFrame(columns=ISSUE_COLUMNS),
    )
