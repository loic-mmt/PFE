"""Immutable prospective option samples; existing calculation engine is reused."""

import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from pfe.data.captures import CaptureLimits
from pfe.data.contracts import _instant
from pfe.data.options import OptionData, normalize_option_chain, option_data_from_payload
from pfe.data.qualification import sanitize_metadata
from pfe.data.storage import _reject_secrets


@dataclass(frozen=True)
class OptionCapture:
    collection_id: str
    data: OptionData
    metadata: dict


def archive_option_sample(payload, store, *, snapshot_id, received_at, limits=None):
    """Archive a selected LSE chain/print/contract payload at receipt availability.

    Original snapshot_at is metadata; all rows are unavailable before receipt.
    Filters/caps never imply a consolidated market, OI or NBBO universe.
    """
    limits = limits or CaptureLimits()
    if not isinstance(payload, dict) or payload.get("provider") != "LSE":
        raise ValueError("Option archive requires LSE sample payload")
    _reject_secrets(payload)
    body = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    if (
        len(body) > limits.max_response_bytes
        or sum(len(payload.get(k, [])) for k in ("chain", "flow", "candles")) > limits.max_rows
    ):
        raise ValueError("Option sample exceeds byte/row budget")
    receipt = _instant(received_at, "received_at")
    original = _instant(payload["snapshot_at"], "snapshot_at")
    if original > receipt:
        raise ValueError("Option snapshot cannot follow receipt")
    source = {
        "source_id": "lse",
        "coverage_scope": "selected_sample",
        "rights": "qualify_upstream_license",
    }
    collection = store.begin(
        "lse",
        "option_history",
        dict(
            snapshot_id=snapshot_id,
            response_sha256=hashlib.sha256(body).hexdigest(),
            received_at=receipt.isoformat(),
        ),
        source_metadata=source,
    )
    pages = {p["page_id"]: p for p in store.recover(collection)["pages"]}
    if "summary" in pages:
        metadata = json.loads(store.read_page(collection, "summary"))
        frames = {
            k: pd.read_parquet(io.BytesIO(store.read_page(collection, k)))
            for k in ("chain", "flow", "candles")
        }
        for key, frame in frames.items():
            frame.attrs.update(
                limit_reached=metadata["limit_reached"].get(key, False),
                coverage_scope="selected_sample",
            )
        return OptionCapture(
            collection,
            OptionData(**frames, snapshot_at=original, limit_reached=metadata["limit_reached"]),
            metadata,
        )
    store.write_page(collection, "response", body, metadata={"received_at": receipt.isoformat()})
    canonical = dict(payload)
    for kind, field in (("chain", "last_trade_at"), ("flow", "ts"), ("candles", "minute")):
        canonical[kind] = [dict(row) for row in payload.get(kind, [])]
        for row in canonical[kind]:
            value = row.get(field)
            # Native vault contract, also normalized by lse-data 0.14 _isoify.
            if isinstance(value, str) and re.fullmatch(
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?", value
            ):
                row[field] = value.replace(" ", "T", 1) + "Z"
    data = option_data_from_payload(canonical)
    for kind in ("chain", "flow", "candles"):
        frame = getattr(data, kind)
        frame["available_at"] = frame.available_at.where(frame.available_at >= receipt, receipt)
    metadata = dict(
        source_id="lse",
        received_at=receipt.isoformat(),
        source_snapshot_at=original.isoformat(),
        source_sha256=hashlib.sha256(body).hexdigest(),
        coverage_scope="selected_sample",
        limit_reached=data.limit_reached,
        filters=payload.get("filters", {}),
        historical_point_in_time="unqualified",
        quote_coverage="unqualified",
        oi_coverage="unqualified",
    )
    outputs = {}
    for key in ("chain", "flow", "candles"):
        frame = getattr(data, key)
        frame.attrs.update(
            limit_reached=bool(data.limit_reached.get(key, False)), coverage_scope="selected_sample"
        )
        stream = io.BytesIO()
        frame.to_parquet(stream)
        outputs[key] = stream.getvalue()
    summary = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    if len(body) + sum(map(len, outputs.values())) + len(summary) > limits.max_archive_bytes:
        raise ValueError("Option archive exceeds cumulative byte budget")
    for key, content in outputs.items():
        store.write_page(collection, key, content)
    store.write_page(collection, "summary", summary, complete=True)
    return OptionCapture(collection, data, metadata)


def collect_option_chain(
    transport, store, *, underlying, snapshot_id, limit=20, filters=None, limits=None
):
    """One bounded native LSE chain GET; replay uses stored bytes without SDK."""
    import re

    limits = limits or CaptureLimits()
    filters = filters or {}
    if not isinstance(underlying, str) or not re.fullmatch(r"[A-Z][A-Z0-9.]{0,14}", underlying):
        raise ValueError("Option underlying must be an explicit ticker")
    if type(limit) is not int or not 1 <= limit <= min(5000, limits.max_rows):
        raise ValueError("Option limit must fit provider and capture budgets")
    if set(filters) - {"type", "expiry", "strike_min", "strike_max"}:
        raise ValueError("Unsupported selected chain filter")
    if "type" in filters and filters["type"] not in ("call", "put"):
        raise ValueError("Option type filter must be call or put")
    if (
        "expiry" in filters
        and pd.Timestamp(filters["expiry"]).strftime("%Y-%m-%d") != filters["expiry"]
    ):
        raise ValueError("Option expiry filter must be YYYY-MM-DD")
    for name in ("strike_min", "strike_max"):
        if name in filters and (
            type(filters[name]) not in (int, float) or not 0 < filters[name] < float("inf")
        ):
            raise ValueError("Option strike filter must be finite and positive")
    if filters.get("strike_min", 0) > filters.get("strike_max", float("inf")):
        raise ValueError("Option strike bounds reversed")
    request = {"underlying": underlying, "limit": limit, **filters}
    audit = store.begin(
        "lse",
        "option_chain_request",
        dict(snapshot_id=snapshot_id, params=request),
        source_metadata={"source_id": "lse", "coverage_scope": "selected_chain"},
    )
    pages = {p["page_id"]: p for p in store.recover(audit)["pages"]}
    if "response" in pages:
        sample = json.loads(store.read_page(audit, "response"))
        received = pages["response"]["metadata"]["received_at"]
    else:
        response = transport.get("/options/chain", request, max_bytes=limits.max_response_bytes)
        rows = response.payload
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError("Option response exceeds requested row count")
        if any(row.get("underlying") != underlying for row in rows):
            raise ValueError("Option chain returned a different underlying")
        for row in rows:
            if any(
                row.get(field) != filters[key]
                for field, key in (("contract_type", "type"), ("expiry", "expiry"))
                if key in filters
            ):
                raise ValueError("Option chain response differs from selected contract filters")
            if (
                not filters.get("strike_min", 0)
                <= row.get("strike", -1)
                <= filters.get("strike_max", float("inf"))
            ):
                raise ValueError("Option strike outside selected bounds")
        received = datetime.now(UTC).isoformat()
        sample = dict(
            provider="LSE",
            snapshot_at=received,
            underlying=underlying,
            limit=limit,
            filters=filters,
            chain=sanitize_metadata(rows),
            flow=[],
            candles=[],
            limit_reached={"chain": len(rows) == limit, "flow": False, "candles": False},
        )
        body = json.dumps(sample, allow_nan=False).encode()
        if len(body) > limits.max_response_bytes:
            raise ValueError("Option envelope exceeds response budget")
        store.write_page(audit, "response", body, complete=True, metadata={"received_at": received})
    return archive_option_sample(
        sample, store, snapshot_id=snapshot_id, received_at=received, limits=limits
    )


def merge_option_chains(*captures):
    frames = [c.data.chain if isinstance(c, OptionCapture) else c for c in captures]
    if not frames:
        raise ValueError("Provide captured option chains")
    out = pd.concat(frames, ignore_index=True)
    keys = ["underlying", "snapshot_at", "ticker"]
    for _, group in out.groupby(keys):
        if len(group.drop_duplicates()) > 1:
            raise ValueError("Conflicting option snapshot; preserve raw revisions separately")
    out = normalize_option_chain(out.drop_duplicates(keys))
    out.attrs.update(
        limit_reached=any(f.attrs.get("limit_reached", False) for f in frames),
        coverage_scope="selected_samples",
    )
    return out
