"""Shared observation contracts and availability-based vintage selection.

Economic periods never establish when information was known. Date-only periods
remain ISO dates; observation instants become UTC ISO timestamps. This preserves
their precision while allowing mixed periods in one Parquet-compatible column.
"""

import json
import re
from datetime import date, datetime
from numbers import Integral, Real

import numpy as np
import pandas as pd

OBSERVATION_COLUMNS = (
    "entity_id",
    "observation_date",
    "published_at",
    "available_at",
    "value",
    "source_id",
    "revision_id",
    "metric",
    "unit",
    "schema_version",
)
AVAILABILITY_BASES = ("unknown", "receipt", "publication", "reconstructed", "event_time")


def _missing(value):
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or isinstance(value, (float, np.floating))
        and np.isnan(value)
    )


def _frame(frame, required, name):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f"{name} requires a DataFrame with unique columns")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} requires columns {sorted(missing)}")
    return frame.copy().reset_index(drop=True)


def _identifiers(frame, columns, *, nullable=()):
    for column in columns:
        values = []
        for value in frame[column]:
            if column in nullable and _missing(value):
                values.append(None)
            elif isinstance(value, str) and value and value == value.strip():
                values.append(value)
            else:
                raise ValueError(f"{column} requires non-empty strings without outer whitespace")
        frame[column] = pd.Series(values, dtype=object)


def _instant(value, name):
    if (
        isinstance(value, (bool, Real))
        or isinstance(value, date)
        and not isinstance(value, datetime)
    ):
        raise ValueError(f"{name} requires a timezone-aware timestamp, not a date or epoch")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} requires a timezone-aware timestamp") from error
    if pd.isna(stamp) or stamp.tz is None:
        raise ValueError(f"{name} requires a non-missing timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _timestamps(values, name, *, nullable=False):
    result = [pd.NaT if nullable and _missing(value) else _instant(value, name) for value in values]
    return pd.Series(result, dtype="datetime64[ns, UTC]")


def _provenance(frame):
    """Normalize optional metadata without asserting unverified PIT provenance."""
    for column in ("published_at", "ingested_at"):
        if column not in frame:
            frame[column] = pd.NaT
        frame[column] = _timestamps(frame[column], column, nullable=True)
    frame["available_at"] = _timestamps(frame.available_at, "available_at")
    if (frame.available_at < frame.published_at).any():
        raise ValueError("available_at cannot precede published_at")
    if "point_in_time" not in frame:
        frame["point_in_time"] = False
    if not frame.point_in_time.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("point_in_time requires explicit booleans")
    frame["point_in_time"] = frame.point_in_time.astype(bool)
    if "availability_basis" not in frame:
        frame["availability_basis"] = "unknown"
    if not frame.availability_basis.isin(AVAILABILITY_BASES).all():
        raise ValueError(f"availability_basis must be one of {AVAILABILITY_BASES}")
    receipt = frame.availability_basis.eq("receipt")
    if frame.loc[receipt, "ingested_at"].isna().any():
        raise ValueError("receipt availability requires ingested_at")
    if (frame.loc[receipt, "available_at"] < frame.loc[receipt, "ingested_at"]).any():
        raise ValueError("receipt available_at cannot precede ingested_at")
    publication = frame.availability_basis.eq("publication")
    if frame.loc[publication, "published_at"].isna().any():
        raise ValueError("publication availability requires published_at")


def _economic_date(value):
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as error:
            raise ValueError("observation_date requires a valid date or aware timestamp") from error
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    return _instant(value, "observation_date").isoformat()


def _structured(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Real) and np.isfinite(value):
        return float(value)
    if isinstance(value, list):
        return [_structured(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _structured(item) for key, item in value.items()}
    raise ValueError("value requires finite numbers or JSON-compatible structures")


def _value(value):
    if _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, dict, list)):
        raise ValueError("value requires finite numbers, JSON objects/arrays, or explicit missing")
    # Validate JSON encoding and detach nested structures from the caller's input.
    try:
        result = _structured(value)
        return json.loads(json.dumps(result, allow_nan=False))
    except RecursionError as error:
        raise ValueError("value requires acyclic JSON-compatible structures") from error
    except (TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ValueError) and str(error).startswith("value requires finite"):
            raise
        raise ValueError("value requires JSON-compatible structures") from error


def _vintages(frame, keys):
    if frame.duplicated([*keys, "available_at"]).any():
        raise ValueError("Duplicate or conflicting simultaneous vintage")
    if frame.duplicated([*keys, "revision_id"]).any():
        raise ValueError("Duplicate revision_id for the same record; revisions are immutable")


def normalize_observations(frame):
    """Validate numerical/structured observations without modifying input.

    Required columns are ``OBSERVATION_COLUMNS``. ``revision_id`` may be null
    for one unversioned record; subsequent vintages require unique non-null IDs.
    IDs, metric, unit and schema_version are non-empty strings. Units must remain
    consistent across vintages of one entity/metric/period/source/instrument.

    ``published_at`` is nullable; known publication, availability and optional
    ingestion require timezone-aware timestamps. A future observation period
    (for example guidance) is permitted. ``observation_date`` becomes an ISO
    date or UTC ISO timestamp; no time is invented for a date-only period.

    Optional columns: instrument_id, ingested_at, availability_basis,
    point_in_time, source_document_id. Provenance defaults to unknown/False,
    never inferred from timestamp presence. In receipt mode, ingestion is
    required and availability cannot precede it. Backfills may retain historical
    availability earlier than ingestion with a documented non-receipt basis.
    Extra columns are preserved. Top-level null/NaN/pd.NA is explicit missing;
    non-finite values inside JSON, infinity, numeric strings and booleans as a
    top-level numerical value are rejected.
    """
    out = _frame(frame, OBSERVATION_COLUMNS, "Observations")
    for column in ("instrument_id", "source_document_id"):
        if column not in out:
            out[column] = None
    _identifiers(
        out,
        (
            "entity_id",
            "metric",
            "unit",
            "source_id",
            "revision_id",
            "schema_version",
            "instrument_id",
            "source_document_id",
        ),
        nullable=("revision_id", "instrument_id", "source_document_id"),
    )
    if "currency" in out:
        _identifiers(out, ("currency",), nullable=("currency",))
    out["observation_date"] = pd.Series(
        [_economic_date(value) for value in out.observation_date], dtype=object
    )
    out["value"] = pd.Series([_value(value) for value in out.value], dtype=object)
    _provenance(out)
    keys = ["entity_id", "metric", "observation_date", "source_id", "instrument_id"]
    _vintages(out, keys)
    if out.groupby(keys, dropna=False).unit.nunique().gt(1).any():
        raise ValueError("unit cannot change between vintages of the same observation")
    return out.sort_values("available_at", kind="stable", ignore_index=True)


def observations_asof(frame, decision_at):
    """Return latest known vintage for every distinct economic period at decision.

    Versions are partitioned by entity_id/metric/observation_date/source_id and
    optional instrument_id. Different periods received simultaneously all remain
    visible; a late revision of an old period cannot replace a newer period.
    Sources remain independent rather than being silently reconciled. Equal-time
    versions of the same record are ambiguous and rejected by the normalizer.
    Availability is inclusive; decision_at must be timezone-aware. The entire
    input is validated, and future records are excluded rather than backfilled.
    """
    decision = _instant(decision_at, "decision_at")
    out = normalize_observations(frame)
    visible = out.loc[out.available_at <= decision]
    keys = ["entity_id", "metric", "observation_date", "source_id", "instrument_id"]
    return visible.drop_duplicates(keys, keep="last").reset_index(drop=True)
