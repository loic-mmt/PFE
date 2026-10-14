"""Bitemporal security identities with explicit source revisions.

Each mapping_id identifies one economic interval whose corrections share that
ID. Separate listing/symbol/status intervals require separate IDs. No current
snapshot is backdated into a historical universe.
"""

import re

import pandas as pd

from pfe.data.contracts import (
    _frame,
    _identifiers,
    _instant,
    _provenance,
    _timestamps,
    _vintages,
)

SECURITY_MASTER_COLUMNS = (
    "mapping_id",
    "entity_id",
    "instrument_id",
    "ticker",
    "venue",
    "currency",
    "status",
    "valid_from",
    "valid_to",
    "available_at",
    "source_id",
    "revision_id",
    "schema_version",
)
SECURITY_STATUSES = ("active", "suspended", "delisted", "inactive")


def _intervals(frame, groups, *, identity=None):
    """Reject overlaps; identical cross-source identities can share a ticker."""
    for _, group in frame.groupby(groups, dropna=False, sort=False):
        greatest_end, incumbent = None, None
        for row in group.sort_values("valid_from", kind="stable").itertuples(index=False):
            start = row.valid_from.value
            end = pd.Timestamp.max.value if pd.isna(row.valid_to) else row.valid_to.value
            current = None if identity is None else tuple(getattr(row, name) for name in identity)
            if greatest_end is not None and start < greatest_end:
                if identity is None or current != incumbent:
                    raise ValueError("Overlapping or ambiguous security mappings")
            if greatest_end is None or end > greatest_end:
                greatest_end, incumbent = end, current


def _mapping_intervals(frame):
    _intervals(frame, ["source_id", "instrument_id"])
    _intervals(frame, ["ticker", "venue"], identity=("instrument_id", "entity_id"))
    _intervals(frame, ["instrument_id"], identity=("entity_id",))


def _validate_history(frame):
    # Batch simultaneous source changes before checking their resulting state:
    # closing the old symbol and adding the new one must be atomic at receipt.
    state = {}
    columns = list(frame.columns)
    for _, batch in frame.groupby("available_at", sort=True):
        for row in batch.itertuples(index=False, name=None):
            record = dict(zip(columns, row, strict=True))
            state[(record["source_id"], record["mapping_id"])] = record
        _mapping_intervals(pd.DataFrame(state.values(), columns=columns))


def normalize_security_master(frame):
    """Validate historical identity intervals and immutable mapping revisions.

    Required columns are ``SECURITY_MASTER_COLUMNS``; revision_id is nullable for
    a single unversioned mapping. mapping_id is stable within source_id across
    revisions, including corrections of the interval, entity or instrument.
    It is NOT the instrument ID: ticker changes and a delisted status interval
    have different mapping IDs. To end an open interval, append a corrected
    vintage under its existing mapping_id with valid_to; retain the old row.

    Economic validity is half-open [valid_from, valid_to); valid_to is nullable
    for an open interval. Both economic bounds and available_at require aware
    timestamps. Optional published_at/ingested_at and provenance follow the
    observation contract; point_in_time defaults False. cik and share_class are
    nullable (CIK must be a 1–10 digit string and is zero-padded to 10 digits).
    Ticker, venue, currency and IDs are strings, not normalized by guessing a
    provider convention. Status is active/suspended/delisted/inactive.

    Each availability batch is applied atomically. At EVERY historical state,
    one source cannot give overlapping intervals for an instrument; a ticker
    on a venue cannot overlap different instruments/entities, including across
    sources; an instrument cannot overlap different entities. Agreement between
    sources is retained as independent evidence. Conflicts must be resolved
    explicitly before calling this API. Extra metadata columns are preserved.
    """
    out = _frame(frame, SECURITY_MASTER_COLUMNS, "Security master")
    for column in ("cik", "share_class"):
        if column not in out:
            out[column] = None
    _identifiers(
        out,
        (
            "mapping_id",
            "entity_id",
            "instrument_id",
            "ticker",
            "venue",
            "currency",
            "source_id",
            "revision_id",
            "schema_version",
            "cik",
            "share_class",
        ),
        nullable=("revision_id", "cik", "share_class"),
    )
    if not out.status.isin(SECURITY_STATUSES).all():
        raise ValueError(f"status must be one of {SECURITY_STATUSES}")
    cik = out.cik.dropna()
    if not cik.map(lambda value: re.fullmatch(r"\d{1,10}", value) is not None).all():
        raise ValueError("cik requires a string containing 1 to 10 digits")
    out["cik"] = out.cik.map(lambda value: None if value is None else value.zfill(10))
    out["valid_from"] = _timestamps(out.valid_from, "valid_from")
    out["valid_to"] = _timestamps(out.valid_to, "valid_to", nullable=True)
    if (out.valid_to <= out.valid_from).any():
        raise ValueError("valid_to must be later than valid_from")
    _provenance(out)
    _vintages(out, ["source_id", "mapping_id"])
    out = out.sort_values("available_at", kind="stable", ignore_index=True)
    _validate_history(out)
    return out


def security_master_asof(frame, decision_at, effective_at=None):
    """Return mappings known at decision, valid at the requested economic time.

    decision_at and effective_at require aware timestamps; effective_at defaults
    to decision_at. First resolve the latest known vintage of each source/mapping
    ID, THEN filter its economic interval. Filtering intervals before resolving
    corrections would resurrect a superseded record. Historical decisions cannot
    see corrections or current catalogs delivered later. An explicitly requested
    past effective_at can query present knowledge about a historical interval;
    that result is not evidence of a universe known at the earlier time.

    Returns all valid statuses including delisted; callers must explicitly choose
    their eligible universe. No mapping is inferred outside its stated bounds,
    no current status is extended into the past, and sources remain separate.
    """
    decision = _instant(decision_at, "decision_at")
    effective = decision if effective_at is None else _instant(effective_at, "effective_at")
    out = normalize_security_master(frame)
    visible = out.loc[out.available_at <= decision]
    latest = visible.drop_duplicates(["source_id", "mapping_id"], keep="last")
    valid = (latest.valid_from <= effective) & (
        latest.valid_to.isna() | (effective < latest.valid_to)
    )
    return latest.loc[valid].reset_index(drop=True)
