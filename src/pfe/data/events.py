"""Distinct sourced events, explicit cross-document identities and entity links."""

from dataclasses import dataclass

import pandas as pd

from pfe.data.contracts import _instant
from pfe.data.documents import content_hash
from pfe.data.graph import ISSUE_COLUMNS, EntityRegistry, fact_history, issues_asof, knowledge_frame
from pfe.data.storage import _json_bytes

EVENT_COLUMNS = [
    "event_id",
    "event_version_id",
    "fact_id",
    "event_type",
    "event_status",
    "available_at",
    "first_known_publication_time",
    "source_document_id",
    "evidence_quote",
    "magnitude",
    "magnitude_unit",
    "record_json",
    "review_status",
    "active",
    "point_in_time",
    "source_id",
    "revision_id",
    "text_sha256",
    "published_at",
]
LINK_COLUMNS = [
    "event_id",
    "event_version_id",
    "entity_id",
    "available_at",
    "source_document_id",
    "point_in_time",
    "published_at",
    "source_id",
    "revision_id",
    "record_json",
]


@dataclass
class EventHistory:
    events: pd.DataFrame
    links: pd.DataFrame
    issues: pd.DataFrame


def build_event_history(facts, entities, *, require_review=True, event_ids=None):
    """Deduplicate exact extracted events, including their evidence excerpt.

    Cross-document equivalence requires caller's explicit fact_id->event_id map;
    Distinct excerpts remain distinct unless explicitly linked: missing event
    details must not collapse two launches in one article. All versions
    and evidence remain, while as-of counts use one latest known event identity.
    """
    registry = entities if isinstance(entities, EntityRegistry) else EntityRegistry(entities)
    mapping = event_ids or {}
    if any(not isinstance(value, str) or not value.strip() for value in mapping.values()):
        raise ValueError("Explicit event IDs must be nonempty strings")
    events, links, issues = [], [], []
    for fact in fact_history(facts):
        if fact["task"] != "D":
            continue
        record = fact["record"]
        payload = {
            key: sorted(set(value)) if isinstance(value, list) else value
            for key, value in record.items()
        }
        event_id = mapping.get(fact["fact_id"], "event:" + content_hash(_json_bytes(payload)))
        version_id = "event-version:" + content_hash(
            _json_bytes({"fact": fact["fact_id"], "available_at": fact["available_at"]})
        )
        active = (
            fact["review_status"] == "approved"
            or not require_review
            and fact["review_status"] == "unreviewed"
        )
        events.append(
            {
                "event_id": event_id,
                "event_version_id": version_id,
                "fact_id": fact["fact_id"],
                "event_type": record["event_type"],
                "event_status": record["event_status"],
                "available_at": fact["available_at"],
                "first_known_publication_time": record["first_known_publication_time"],
                "source_document_id": fact["source_document_id"],
                "evidence_quote": record["evidence_quote"],
                "magnitude": record["event_magnitude"],
                "magnitude_unit": record["magnitude_unit"],
                "record_json": _json_bytes(record).decode(),
                "review_status": fact["review_status"],
                "active": bool(active),
                "point_in_time": fact["point_in_time"],
                "source_id": fact["source_id"],
                "revision_id": fact["revision_id"],
                "text_sha256": fact["text_sha256"],
                "published_at": fact["published_at"],
            }
        )
        for alias in record["entities_involved"] or []:
            entity = registry.binding(alias, observation_date=fact["available_at"][:10])
            if entity is None:
                issues.append(
                    {
                        "fact_id": fact["fact_id"],
                        "alias": alias,
                        "reason": "unresolved event entity",
                        "available_at": fact["available_at"],
                    }
                )
                continue
            at = max(
                _instant(fact["available_at"], "available_at"),
                _instant(entity["available_at"], "available_at"),
            ).isoformat()
            links.append(
                {
                    "event_id": event_id,
                    "event_version_id": version_id,
                    "entity_id": entity["entity_id"],
                    "available_at": at,
                    "source_document_id": fact["source_document_id"],
                    "point_in_time": bool(fact["point_in_time"] and entity["point_in_time"]),
                    "published_at": fact["published_at"],
                    "source_id": fact["source_id"],
                    "revision_id": fact["revision_id"],
                    "record_json": _json_bytes(
                        {"event_id": event_id, "event_version_id": version_id}
                    ).decode(),
                }
            )
    return EventHistory(
        knowledge_frame(
            pd.DataFrame(events, columns=EVENT_COLUMNS),
            entity_column="event_id",
            record_column="event_version_id",
            schema_version="events-1",
        ),
        knowledge_frame(
            pd.DataFrame(links, columns=LINK_COLUMNS).drop_duplicates(),
            entity_column="entity_id",
            record_column="event_version_id",
            schema_version="event-links-1",
        ),
        pd.DataFrame(issues, columns=ISSUE_COLUMNS),
    )


def events_asof(history, decision_at):
    decision = _instant(decision_at, "decision_at")
    events, links = history.events.copy(), history.links.copy()
    issues = issues_asof(history.issues, decision)
    if events.empty:
        return EventHistory(events, links.iloc[:0].copy(), issues)
    events = events.loc[events.available_at.map(lambda v: _instant(v, "available_at")) <= decision]
    versions = events.sort_values("available_at", kind="stable")
    events = versions.drop_duplicates("fact_id", keep="last")
    events = events.loc[events.active]
    # A new review updates the proof's status, not when the event first became
    # known. Retain earlier active versions only for proofs still accepted now.
    accepted_versions = versions.loc[versions.fact_id.isin(events.fact_id) & versions.active]
    first = accepted_versions.groupby("event_id").available_at.min()
    publication = {}
    for event in accepted_versions.itertuples():
        if event.first_known_publication_time is not None:
            at = _instant(event.first_known_publication_time, "first_known_publication_time")
            prior = publication.get(event.event_id)
            if prior is None or at < prior[0]:
                publication[event.event_id] = (at, event.source_document_id)
    valid_links = links.loc[
        links.event_version_id.isin(accepted_versions.event_version_id)
        & (links.available_at.map(lambda v: _instant(v, "available_at")) <= decision)
    ]
    first_links = valid_links.groupby(["event_id", "entity_id"]).available_at.min()
    events = events.drop_duplicates("event_id", keep="last")
    events["first_available_at"] = events.event_id.map(first)
    events["first_known_publication_time"] = [
        publication[row.event_id][0].isoformat() if row.event_id in publication else None
        for row in events.itertuples()
    ]
    events["publication_source_document_id"] = [
        publication[row.event_id][1] if row.event_id in publication else None
        for row in events.itertuples()
    ]
    links = links.loc[
        links.event_version_id.isin(events.event_version_id)
        & (links.available_at.map(lambda v: _instant(v, "available_at")) <= decision)
    ]
    links["first_available_at"] = [
        first_links.loc[(row.event_id, row.entity_id)] for row in links.itertuples()
    ]
    return EventHistory(events.reset_index(drop=True), links.reset_index(drop=True), issues)
