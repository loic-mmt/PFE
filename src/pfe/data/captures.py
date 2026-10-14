"""Small raw/normalized captures shared by optional Lot 5 sources."""

import hashlib
import io
import json
from dataclasses import asdict, dataclass

import pandas as pd

from pfe.data.contracts import _instant, normalize_observations
from pfe.data.http import HTTPResponse
from pfe.data.storage import _reject_secrets


@dataclass(frozen=True)
class CaptureLimits:
    max_rows: int = 1000
    max_response_bytes: int = 524_288
    max_archive_bytes: int = 4_194_304

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Capture budgets must be positive integers")
        if self.max_response_bytes > self.max_archive_bytes:
            raise ValueError("Response must fit archive budget")


@dataclass(frozen=True)
class ObservationCapture:
    collection_id: str
    observations: pd.DataFrame
    complete: bool
    issues: tuple[str, ...]
    metadata: dict


def capture_observations(
    *,
    store,
    source,
    dataset,
    snapshot_id,
    query,
    parser,
    declaration,
    fetch=None,
    offline=None,
    limits=None,
    validate_body=None,
):
    """At most one fetch; parser returns (canonical observations, issues).

    Fetch accepts max_bytes and returns HTTPResponse with credential-free bytes.
    A caller's authenticated client must enforce its own network byte cap.
    Parser/query/version form the immutable capture identity. No retries/bulk.
    """
    limits = limits or CaptureLimits()
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("Capture requires explicit snapshot_id")
    request = dict(snapshot_id=snapshot_id, query=query, limits=asdict(limits), schema_version="1")
    if offline is not None:
        body, receipt = offline
        if not isinstance(body, bytes) or len(body) > limits.max_response_bytes:
            raise ValueError("Capture input exceeds response budget")
        request.update(
            offline_sha256=hashlib.sha256(body).hexdigest(),
            receipt=_instant(receipt, "received_at").isoformat(),
        )
    collection = store.begin(source, dataset, request, source_metadata=declaration)
    pages = {p["page_id"]: p for p in store.recover(collection)["pages"]}
    if "summary" in pages:
        metadata = json.loads(store.read_page(collection, "summary"))
        frame = normalize_observations(
            pd.read_parquet(io.BytesIO(store.read_page(collection, "normalized")))
        )
        return ObservationCapture(
            collection, frame, metadata["complete"], tuple(metadata["issues"]), metadata
        )
    if "response" in pages:
        body = store.read_page(collection, "response")
        receipt = pages["response"]["metadata"]["received_at"]
    else:
        if offline is None:
            if fetch is None:
                raise ValueError(
                    "Uncached capture requires offline bytes or a configured source client"
                )
            response = fetch(max_bytes=limits.max_response_bytes)
            if not isinstance(response, HTTPResponse):
                raise ValueError("Source client must return HTTPResponse")
            body, receipt = response.body, response.received_at
        if len(body) > limits.max_response_bytes:
            raise ValueError("Source response exceeds byte budget")
        # Optional clients must strip request/authentication echoes before return.
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            payload = None
        if payload is not None:
            _reject_secrets(payload)
        if validate_body is not None:
            validate_body(body)
        _instant(receipt, "received_at")
        store.write_page(collection, "response", body, metadata={"received_at": receipt})
    frame, issues = parser(body, received_at=receipt, max_rows=limits.max_rows)
    frame = normalize_observations(frame)
    if len(frame) > limits.max_rows:
        raise ValueError("Normalized capture exceeds row budget")
    stream = io.BytesIO()
    frame.to_parquet(stream)
    normalized = stream.getvalue()
    metadata = dict(
        source_id=source,
        source_sha256=hashlib.sha256(body).hexdigest(),
        received_at=receipt,
        rows=len(frame),
        complete=not issues,
        issues=list(issues),
        completion_scope="selected_response",
        historical_coverage="unqualified",
        request=request,
    )
    summary = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    if len(body) + len(normalized) + len(summary) > limits.max_archive_bytes:
        raise ValueError("Raw plus normalized capture exceeds archive budget")
    store.write_page(collection, "normalized", normalized, row_count=len(frame))
    store.write_page(collection, "summary", summary, complete=not issues)
    return ObservationCapture(collection, frame, not issues, tuple(issues), metadata)


def merge_observation_captures(*captures):
    """Combine overlapping COT/estimate vintages, preserving their first receipt."""
    frames = [c.observations if isinstance(c, ObservationCapture) else c for c in captures]
    if not frames:
        raise ValueError("Provide at least one observation capture")
    out = pd.concat([normalize_observations(frame) for frame in frames], ignore_index=True)
    keys = ["entity_id", "metric", "observation_date", "source_id", "instrument_id", "revision_id"]
    for _, group in out.groupby(keys, dropna=False):
        values = {json.dumps(v, sort_keys=True, allow_nan=False) for v in group.value}
        if len(values) != 1 or group.unit.nunique() != 1:
            raise ValueError("Conflicting immutable observation revision")
    out = out.sort_values("available_at", kind="stable").drop_duplicates(keys, keep="first")
    return normalize_observations(out)
