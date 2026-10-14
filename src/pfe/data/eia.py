"""EIA API v2 observations, conservative receipt timing, bounded pagination.

Official contract: https://www.eia.gov/opendata/documentation.php . API values
are numeric strings; period is an economic label, not a release timestamp.
Latest API values do not establish historical vintages. API-key request echoes
are removed before raw storage. No economic date is used as availability.
"""

import hashlib
import io
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType

import numpy as np
import pandas as pd

from pfe.data.contracts import OBSERVATION_COLUMNS, _instant, normalize_observations
from pfe.data.qualification import sanitize_metadata
from pfe.data.sources import default_source_registry

EIA_DOCUMENTATION_URL = "https://www.eia.gov/opendata/documentation.php"
EIA_BASE_URL = "https://api.eia.gov/v2/"


class ContextError(ValueError):
    """Invalid source schema, without credentials or provider bodies."""


class ContextLimitError(ContextError):
    """An explicit source ingestion budget was reached."""


@dataclass(frozen=True)
class ContextLimits:
    max_rows: int = 25_000
    max_bytes: int = 1_048_576
    max_requests: int = 8
    page_size: int = 5000

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ContextError("Context limits must be positive integers")
        if self.page_size > 5000:
            raise ContextError("EIA JSON page_size cannot exceed 5000")


@dataclass(frozen=True)
class ContextCollection:
    collection_id: str
    observations: pd.DataFrame
    complete: bool
    issues: tuple[str, ...]
    metadata: dict


@dataclass(frozen=True)
class EIAQuery:
    route: str
    entity_id: str
    value_field: str = "value"
    frequency: str = "daily"
    facets: Mapping = field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    unit: str | None = None
    calendar: str = "provider_period_labels"
    published_column: str | None = None
    availability_basis: str = "receipt"

    def __post_init__(self):
        if not isinstance(self.route, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", self.route
        ):
            raise ContextError("EIA route must be a relative v2 dataset path")
        for name in ("entity_id", "value_field", "calendar"):
            _identifier(getattr(self, name), name)
        if self.frequency not in ("daily", "weekly", "monthly", "quarterly", "annual"):
            raise ContextError("EIA frequency must explicitly describe supported economic periods")
        if self.availability_basis not in ("receipt", "publication"):
            raise ContextError("EIA availability_basis must be receipt or publication")
        if not isinstance(self.facets, Mapping):
            raise ContextError("EIA facets must map names to explicit string values")
        selected_facets = {}
        for name, values in self.facets.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ContextError("Invalid EIA facet name")
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, (tuple, list)) or not values:
                raise ContextError("EIA facet values require a nonempty list")
            if len(values) != 1:
                raise ContextError("EIA factors require exactly one explicit code per facet")
            for value in values:
                _identifier(value, "facet value")
            selected_facets[name] = tuple(values)
        object.__setattr__(self, "facets", MappingProxyType(selected_facets))
        for name in ("unit", "published_column"):
            if getattr(self, name) is not None:
                _identifier(getattr(self, name), name)
        for value in (self.start, self.end):
            if value is not None:
                _period(value, self.frequency)
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ContextError("EIA start must not follow end")

    def to_dict(self):
        result = {
            item.name: getattr(self, item.name) for item in fields(self) if item.name != "facets"
        }
        result["facets"] = {name: list(values) for name, values in self.facets.items()}
        return result

    def params(self, *, offset, length):
        result = {
            "frequency": self.frequency,
            "data[0]": self.value_field,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": length,
        }
        for name, value in (("start", self.start), ("end", self.end)):
            if value is not None:
                result[name] = value
        for name, values in self.facets.items():
            for index, value in enumerate([values] if isinstance(values, str) else values):
                result[f"facets[{name}][{index}]"] = value
        return result


def _identifier(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContextError(f"{name} requires a nonempty explicit string")


def _json_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ContextError("Source payload must be finite JSON-compatible data") from None


def _number(value):
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str) and value.strip() in ("", "NA", "N/A", "--", "..", "..."):
        return None
    if isinstance(value, (bool, np.bool_, complex)):
        raise ContextError("Source values require finite numbers or explicit missing values")
    if isinstance(value, str) and not re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip()
    ):
        raise ContextError("Unsupported source numeric string; no unit/decimal conversion inferred")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ContextError("Source values require finite numbers") from None
    if not np.isfinite(result):
        raise ContextError("Source values require finite numbers")
    return result


def _period(value, frequency):
    if not isinstance(value, str):
        raise ContextError("Economic periods require explicit date strings")
    try:
        if frequency in ("daily", "weekly") and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            period = pd.Timestamp(value).date().isoformat()
            return period, period, "provider_date_label"
        if frequency == "monthly" and re.fullmatch(r"\d{4}-\d{2}", value):
            period = pd.Period(value, freq="M")
        elif frequency == "quarterly" and re.fullmatch(r"\d{4}-Q[1-4]", value):
            period = pd.Period(value, freq="Q")
        elif frequency == "annual" and re.fullmatch(r"\d{4}", value):
            period = pd.Period(value, freq="Y")
        else:
            raise ValueError
        return (
            period.start_time.date().isoformat(),
            period.end_time.date().isoformat(),
            "economic_period_start",
        )
    except (ValueError, OverflowError):
        raise ContextError("Economic period does not match the declared frequency") from None


def _publication(value):
    if value is None or value is pd.NaT or value is pd.NA:
        return None, None
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return None, pd.Timestamp(value).date().isoformat()
        except ValueError:
            raise ContextError("Invalid publication date") from None
    return _instant(value, "published_at"), None


def _observation(
    *,
    entity_id,
    metric,
    period_start,
    period_end,
    period_label,
    value,
    unit,
    source_id,
    revision_id,
    received_at,
    published_at=None,
    availability_basis="receipt",
    frequency,
    calendar,
    currency=None,
):
    receipt = _instant(received_at, "received_at")
    publication, publication_date = _publication(published_at)
    if availability_basis == "publication":
        if publication is None:
            raise ContextError("Publication policy requires an explicit aware release timestamp")
        available = publication
    else:
        available = max(receipt, publication) if publication is not None else receipt
    return dict(
        entity_id=entity_id,
        metric=metric,
        observation_date=period_start,
        period_end=period_end,
        observation_period=period_label,
        published_at=publication,
        publication_date=publication_date,
        available_at=available,
        value=_number(value),
        unit=unit,
        source_id=source_id,
        revision_id=revision_id,
        schema_version="1",
        source_document_id=f"{source_id}:{revision_id}",
        ingested_at=receipt,
        availability_basis=availability_basis,
        point_in_time=False,
        frequency=frequency,
        calendar=calendar,
        currency=currency,
        **{
            "quality.value_known": _number(value) is not None,
            "quality.unit_known": unit != "unknown",
        },
    )


def _observations(records):
    columns = list(OBSERVATION_COLUMNS) + [
        "period_end",
        "observation_period",
        "publication_date",
        "ingested_at",
        "availability_basis",
        "point_in_time",
        "frequency",
        "calendar",
        "currency",
        "quality.value_known",
        "quality.unit_known",
        "source_document_id",
    ]
    return normalize_observations(pd.DataFrame(records, columns=columns))


def _eia_payload(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
        raise ContextError("EIA response requires an explicit response object")
    response = payload["response"]
    rows = response.get("data")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ContextError("EIA response.data must be an array of objects")
    total = response.get("total")
    if type(total) is int and total >= 0:
        pass
    elif isinstance(total, str) and re.fullmatch(r"\d+", total):
        total = int(total)
    else:
        raise ContextError("EIA response requires a non-negative provider total")
    return response, rows, total


def _eia_identity(rows, query):
    """One factor must have one dimensional combination, including across pages."""
    timing = {
        "period",
        "released",
        "release_date",
        "releaseDate",
        "published_at",
        "publication_date",
        "updated",
        "updated_at",
        "last_updated",
        query.value_field,
        query.value_field + "-units",
        query.published_column,
    }
    signatures, units = set(), set()
    for row in rows:
        for name, values in query.facets.items():
            selected = values if isinstance(values, str) else values[0]
            if name in row and row[name] != selected:
                raise ContextError("EIA returned facet metadata disagrees with the exact selection")
        signatures.add(_json_bytes({key: value for key, value in row.items() if key not in timing}))
        unit = row.get(query.value_field + "-units")
        if unit is not None:
            _identifier(unit, "provider unit")
            units.add(unit)
    if len(signatures) > 1:
        raise ContextError("Ambiguous EIA multiple dimensional series; select one factor per query")
    if len(units) > 1:
        raise ContextError("EIA units change within a factor; select a consistent unit")
    return next(iter(signatures)) if signatures else None, next(iter(units)) if units else None


def parse_eia_page(payload, *, query: EIAQuery, received_at, revision_id=None, max_rows=5000):
    """Parse exact selected field/units; date-only release remains unknown time."""
    response, rows, _ = _eia_payload(payload)
    if type(max_rows) is not int or max_rows < 1 or len(rows) > max_rows:
        raise ContextLimitError("EIA page exceeds max_rows")
    if response.get("frequency", query.frequency) != query.frequency:
        raise ContextError("EIA response frequency disagrees with the explicit query")
    _eia_identity(rows, query)
    evidence = sanitize_metadata(payload)
    evidence.pop("request", None)
    revision_id = revision_id or hashlib.sha256(_json_bytes(evidence)).hexdigest()
    records = []
    for row in rows:
        if query.value_field not in row or "period" not in row:
            raise ContextError("EIA row misses the explicitly selected period/value field")
        first, last, _ = _period(row["period"], query.frequency)
        if (query.start is not None and row["period"] < query.start) or (
            query.end is not None and row["period"] > query.end
        ):
            raise ContextError("EIA economic period falls outside the inclusive query bounds")
        provider_unit = row.get(query.value_field + "-units")
        if provider_unit is not None:
            _identifier(provider_unit, "provider unit")
            if query.unit is not None and query.unit != provider_unit:
                raise ContextError("Declared EIA unit disagrees with provider units")
        unit = provider_unit or query.unit or "unknown"
        records.append(
            _observation(
                entity_id=query.entity_id,
                metric=query.value_field,
                period_start=first,
                period_end=last,
                period_label=row["period"],
                value=row[query.value_field],
                unit=unit,
                source_id="eia",
                revision_id=revision_id,
                received_at=received_at,
                published_at=row.get(query.published_column) if query.published_column else None,
                availability_basis=query.availability_basis,
                frequency=query.frequency,
                calendar=query.calendar,
            )
        )
    return _observations(records)


def _cached(store, collection_id):
    manifest = store.recover(collection_id)
    if any(page["page_id"] == "normalized" for page in manifest["pages"]):
        summary = json.loads(store.read_page(collection_id, "summary"))
        return ContextCollection(
            collection_id,
            normalize_observations(
                pd.read_parquet(io.BytesIO(store.read_page(collection_id, "normalized")))
            ),
            summary["complete"],
            tuple(summary["issues"]),
            summary,
        )
    return None


def _result(store, collection_id, frame, complete, issues, metadata):
    summary = {
        **metadata,
        "complete": complete,
        "issues": list(dict.fromkeys(issues)),
        "rows": len(frame),
        "completion_scope": "queried_response_or_selected_file",
        "historical_point_in_time": "unqualified",
        "universe_coverage": "unqualified",
    }
    if complete:
        stream = io.BytesIO()
        frame.to_parquet(stream)
        store.write_page(collection_id, "summary", summary)
        store.write_page(
            collection_id,
            "normalized",
            stream.getvalue(),
            complete=True,
            row_count=len(frame),
            metadata={"completion_scope": summary["completion_scope"]},
        )
    return ContextCollection(collection_id, frame, complete, tuple(summary["issues"]), summary)


def collect_eia(
    transport,
    store,
    *,
    query: EIAQuery,
    snapshot_id,
    api_key=None,
    limits=None,
    source_metadata=None,
    offline_pages=None,
):
    """Persist receipts per paginated query; caps yield an incomplete collection.

    Provider totals must remain stable. Completion refers only to the explicit
    query, never exhaustive markets, series availability or historical vintages.
    offline_pages contains dictionaries with payload and explicit received_at.
    """
    _identifier(snapshot_id, "snapshot_id")
    limits = limits or ContextLimits()
    request = {
        "query": query.to_dict(),
        "snapshot_id": snapshot_id,
        "limits": asdict(limits),
        "mode": "offline" if offline_pages is not None else "online",
    }
    if offline_pages is not None:
        request["offline_sha256"] = hashlib.sha256(
            _json_bytes(sanitize_metadata(offline_pages, secrets=(api_key or "",)))
        ).hexdigest()
    request = sanitize_metadata(request, secrets=(api_key or "",))
    source = source_metadata or default_source_registry().get("eia").to_dict()
    collection_id = store.begin("eia", "api_v2_observations", request, source_metadata=source)
    cached = _cached(store, collection_id)
    if cached:
        return cached
    pages = {page["page_id"] for page in store.manifest(collection_id)["pages"]}
    offset, total, used_bytes, requests, page_index = 0, None, 0, 0, 0
    frames, issues = [], []
    identity, known_unit = None, None
    while total is None or offset < total:
        if offset >= limits.max_rows:
            issues.append("row_cap_reached")
            break
        length = min(limits.page_size, limits.max_rows - offset)
        page_id = f"page-{offset:010d}"
        if page_id in pages:
            page = json.loads(store.read_page(collection_id, page_id))
        else:
            if requests >= limits.max_requests:
                issues.append("request_cap_reached")
                break
            remaining = limits.max_bytes - used_bytes
            if remaining < 1:
                issues.append("byte_cap_reached")
                break
            if offline_pages is not None:
                if page_index >= len(offline_pages):
                    issues.append("offline_pages_incomplete")
                    break
                provided = offline_pages[page_index]
                payload, received = provided["payload"], provided["received_at"]
                raw_size = len(_json_bytes(payload))
            else:
                if not isinstance(api_key, str) or not api_key.strip():
                    raise ContextError("EIA_API_KEY is required for an uncached online query")
                params = {**query.params(offset=offset, length=length), "api_key": api_key}
                response = transport.get(
                    EIA_BASE_URL + query.route + "/data/", params=params, max_bytes=remaining
                )
                try:
                    payload = json.loads(response.body)
                except (ValueError, UnicodeError, RecursionError):
                    raise ContextError("Invalid EIA JSON") from None
                received, raw_size = response.received_at, len(response.body)
            if raw_size > remaining:
                raise ContextLimitError("EIA response exceeds aggregate max_bytes")
            payload = sanitize_metadata(payload, secrets=(api_key or "",))
            payload.pop("request", None)  # Provider echo is credential-bearing, not a fact.
            _, rows, page_total = _eia_payload(payload)
            if len(rows) > length:
                raise ContextLimitError("EIA response exceeds requested page length")
            _instant(received, "received_at")
            page = {
                "payload": payload,
                "received_at": received,
                "response_bytes": raw_size,
                "offset": offset,
                "length": length,
                "total": page_total,
            }
            store.write_page(
                collection_id,
                page_id,
                page,
                row_count=len(rows),
                next_cursor=str(offset + len(rows)),
                metadata={"received_at": received},
            )
            requests += 1
        used_bytes += page["response_bytes"]
        if used_bytes > limits.max_bytes:
            raise ContextLimitError("Cached EIA responses exceed aggregate max_bytes")
        response, rows, page_total = _eia_payload(page["payload"])
        page_identity, page_unit = _eia_identity(rows, query)
        if page_identity is not None:
            if identity is not None and page_identity != identity:
                raise ContextError("EIA dimensional series changes between pages")
            identity = page_identity
        if page_unit is not None:
            if known_unit is not None and page_unit != known_unit:
                raise ContextError("EIA factor units change between pages")
            known_unit = page_unit
        if total is not None and total != page_total:
            issues.append("provider_total_changed")
        total = page_total if total is None else total
        if page["payload"].get("warning") or response.get("warning"):
            issues.append("provider_warning")
        frames.append(
            parse_eia_page(
                page["payload"],
                query=query,
                received_at=page["received_at"],
                max_rows=page["length"],
            )
        )
        offset += len(rows)
        page_index += 1
        if offset > total:
            issues.append("provider_count_mismatch")
            break
        if not rows and offset < total:
            issues.append("empty_provider_page_before_total")
            break
        if total == 0:
            break
    frame = (
        normalize_observations(pd.concat(frames, ignore_index=True))
        if frames
        else _observations([])
    )
    complete = total is not None and offset == total and not issues
    return _result(
        store,
        collection_id,
        frame,
        complete,
        issues,
        {
            "source_id": "eia",
            "provider_total": total,
            "received_rows": offset,
            "source_bytes": used_bytes,
            "request": request,
            "documentation": EIA_DOCUMENTATION_URL,
        },
    )


def _collect_file(
    transport,
    store,
    *,
    source_id,
    dataset,
    url,
    snapshot_id,
    options,
    parser,
    limits=None,
    source_metadata=None,
    offline=None,
):
    limits = limits or ContextLimits()
    _identifier(snapshot_id, "snapshot_id")
    request = {
        "snapshot_id": snapshot_id,
        "url": url,
        "options": json.loads(_json_bytes(options)),
        "limits": asdict(limits),
        "mode": "offline" if offline is not None else "online",
    }
    if offline is not None:
        request["offline_sha256"] = hashlib.sha256(offline[0]).hexdigest()
        request["received_at"] = _instant(offline[1], "received_at").isoformat()
    source = source_metadata or default_source_registry().get(source_id).to_dict()
    collection_id = store.begin(source_id, dataset, request, source_metadata=source)
    cached = _cached(store, collection_id)
    if cached:
        return cached
    pages = {page["page_id"]: page for page in store.manifest(collection_id)["pages"]}
    if "raw-file" in pages:
        body = store.read_page(collection_id, "raw-file")
        received = pages["raw-file"]["metadata"]["received_at"]
    else:
        if offline is not None:
            body, received = offline
        else:
            response = transport.get(url, max_bytes=limits.max_bytes)
            body, received = response.body, response.received_at
        if not isinstance(body, bytes) or len(body) > limits.max_bytes:
            raise ContextLimitError("Context file exceeds max_bytes")
        _instant(received, "received_at")
        store.write_page(
            collection_id,
            "raw-file",
            body,
            metadata={"received_at": received, "sha256": hashlib.sha256(body).hexdigest()},
        )
    frame = parser(body, received_at=received, max_rows=limits.max_rows, **options)
    return _result(
        store,
        collection_id,
        frame,
        True,
        [],
        {
            "source_id": source_id,
            "request": request,
            "source_bytes": len(body),
            "received_at": received,
            "source_version": hashlib.sha256(body).hexdigest(),
        },
    )
