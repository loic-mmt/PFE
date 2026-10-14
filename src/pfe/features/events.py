"""Availability-based event counts, age, compatible magnitudes and optional outputs."""

import numpy as np
import pandas as pd

from pfe.data.contracts import _instant
from pfe.data.events import events_asof
from pfe.extraction.schemas import EVENT_TYPES
from pfe.features.graph import decision_grid

EVENT_FEATURE_GROUPS = ("counts", "age", "magnitude", "sentiment", "embedding", "quality")


def compute_event_features(
    decisions,
    history,
    *,
    entity_ids,
    windows=("7D", "30D"),
    magnitude_units=None,
    enrichments=(),
    model_revision=None,
    embedding_dimensions=None,
):
    """Counts mean known events in the supplied corpus, never all market news.

    Window clock is first event/entity availability, preserved across explicitly
    linked re-publications. Rumors/plans/confirmed stay separate. Magnitudes are
    aggregated only for requested event_type->original unit pairs. Model outputs
    require an explicit compatible digest, and embedding dimensions are fixed.
    """
    index = decision_grid(decisions, entity_ids)
    durations = [(str(window), pd.Timedelta(window)) for window in windows]
    if (
        not durations
        or len({label for label, _ in durations}) != len(durations)
        or any(pd.isna(value) or value <= pd.Timedelta(0) for _, value in durations)
    ):
        raise ValueError("Event windows must be positive durations")
    units = magnitude_units or {}
    if any(
        key not in EVENT_TYPES or not isinstance(value, str) or not value
        for key, value in units.items()
    ):
        raise ValueError("magnitude_units must map event types to explicit units")
    if enrichments and (not isinstance(model_revision, str) or not model_revision):
        raise ValueError("Optional model outputs require explicit model_revision")
    if embedding_dimensions is not None and (
        type(embedding_dimensions) is not int or not 1 <= embedding_dimensions <= 2048
    ):
        raise ValueError("embedding_dimensions must be an integer in [1,2048]")
    rows = []
    for decision in index:
        snapshot = events_asof(history, decision)
        available_outputs = [
            record
            for record in enrichments
            if record["status"] == "accepted"
            and record.get("is_model_output") is True
            and record["model"].get("model_revision") == model_revision
            and _instant(record["available_at"], "available_at") <= decision
        ]
        for entity in entity_ids:
            links = snapshot.links.loc[snapshot.links.entity_id.eq(entity)]
            events = snapshot.events.loc[snapshot.events.event_id.isin(links.event_id)].copy()
            first = {
                row.event_id: max(
                    _instant(row.first_available_at, "first_available_at"),
                    _instant(
                        links.loc[links.event_id.eq(row.event_id), "first_available_at"].min(),
                        "link_available_at",
                    ),
                )
                for row in events.itertuples()
            }
            row = {
                "decision_at": decision,
                "entity_id": entity,
                "events.mask.observed": float(bool(len(events))),
                "events.quality.complete_news_universe": 0.0,
                "events.quality.known_count": float(len(events)),
                "events.quality.publication_coverage": (
                    events.first_known_publication_time.notna().mean() if len(events) else np.nan
                ),
                "events.quality.point_in_time": float(
                    events.point_in_time.all() and links.point_in_time.all()
                )
                if len(events)
                else np.nan,
                "events.age_seconds": min((decision - at).total_seconds() for at in first.values())
                if first
                else np.nan,
            }
            public_times = [
                value for value in events.first_known_publication_time if value is not None
            ]
            row["events.publication_age_seconds"] = (
                min(
                    (decision - _instant(value, "first_known_publication_time")).total_seconds()
                    for value in public_times
                )
                if public_times
                else np.nan
            )
            for label, duration in durations:
                ids = {event_id for event_id, at in first.items() if decision - at <= duration}
                current = events.loc[events.event_id.isin(ids)]
                for kind in EVENT_TYPES:
                    for status in ("confirmed", "announced_plan", "rumor"):
                        subset = current.loc[
                            current.event_type.eq(kind) & current.event_status.eq(status)
                        ]
                        row[f"events.count.{kind}.{status}.{label}"] = float(len(subset))
                for kind, unit in units.items():
                    for status in ("confirmed", "announced_plan", "rumor"):
                        magnitudes = current.loc[
                            current.event_type.eq(kind)
                            & current.magnitude_unit.eq(unit)
                            & current.event_status.eq(status),
                            "magnitude",
                        ].dropna()
                        row[f"events.magnitude.{kind}.{status}.{label}.mean"] = (
                            float(magnitudes.mean()) if len(magnitudes) else np.nan
                        )
                        row[f"events.mask.magnitude.{kind}.{status}.{label}"] = float(
                            bool(len(magnitudes))
                        )
                documents = {
                    record.source_document_id: record.text_sha256 for record in current.itertuples()
                }
                outputs = [
                    output
                    for output in available_outputs
                    if documents.get(output["source_document_id"]) == output["text_sha256"]
                ]
                # One latest output per document/kind; duplicates never overweight articles.
                unique = {}
                for output in sorted(outputs, key=lambda item: item["available_at"]):
                    unique[(output["source_document_id"], output["kind"])] = output
                sentiments = [
                    output["value"]["score"]
                    for output in unique.values()
                    if output["kind"] == "sentiment"
                ]
                if any(
                    type(value) not in (float, int)
                    or not np.isfinite(value)
                    or not -1 <= value <= 1
                    for value in sentiments
                ):
                    raise ValueError("Invalid qualified sentiment output")
                row[f"events.sentiment.{label}.mean"] = (
                    float(np.mean(sentiments)) if sentiments else np.nan
                )
                row[f"events.mask.sentiment.{label}"] = float(bool(sentiments))
                if embedding_dimensions is not None:
                    vectors = [
                        output["value"]["embedding"]
                        for output in unique.values()
                        if output["kind"] == "embedding"
                    ]
                    if any(
                        len(vector) != embedding_dimensions
                        or any(type(v) not in (int, float) or not np.isfinite(v) for v in vector)
                        for vector in vectors
                    ):
                        raise ValueError(
                            "Embedding dimension/finite values differ from configured model output"
                        )
                    mean = (
                        np.mean(vectors, axis=0)
                        if vectors
                        else np.full(embedding_dimensions, np.nan)
                    )
                    for number, value in enumerate(mean):
                        row[f"events.embedding.{label}.{number}"] = float(value)
                    row[f"events.mask.embedding.{label}"] = float(bool(vectors))
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
        model_outputs_qualified=True,
        window_clock="first event/entity availability in supplied corpus",
    )
    return out
