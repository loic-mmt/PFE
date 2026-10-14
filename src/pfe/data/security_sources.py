"""Bounded SEC/Nasdaq/listing snapshots and conservative master updates."""

import csv
import hashlib
import io
import json
from dataclasses import dataclass

import pandas as pd

from pfe.data.contracts import _instant
from pfe.data.http import BoundedHTTPTransport
from pfe.data.security_master import (
    SECURITY_MASTER_COLUMNS,
    normalize_security_master,
    security_master_asof,
)
from pfe.data.storage import _reject_secrets

SECURITIES_URLS = {
    "sec": "https://www.sec.gov/files/company_tickers.json",
    "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "nasdaq_other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    "alpha_vantage": "https://www.alphavantage.co/query",
}


@dataclass(frozen=True)
class SecurityCollection:
    collection_id: str
    snapshot: pd.DataFrame
    history: pd.DataFrame
    complete: bool
    issues: tuple[str, ...]


def parse_security_snapshot(body, *, source, identities, received_at, max_rows=20_000):
    """Parse explicit ticker -> stable identity mappings; never derive permanent IDs.

    Identity values require entity_id, instrument_id, venue, currency. Source
    listing/delisting dates retain date precision in extra columns; validity starts
    at receipt, never at an old economic date from a current export. Missing names
    do not imply delisting. Nasdaq footer completeness is mandatory.
    """
    receipt = _instant(received_at, "received_at")
    if source not in SECURITIES_URLS or not isinstance(body, bytes):
        raise ValueError("Choose a supported security source and bytes")
    if (
        type(max_rows) is not int
        or max_rows < 1
        or not isinstance(identities, dict)
        or not identities
    ):
        raise ValueError("Security snapshot requires identities and a positive row cap")
    _reject_secrets(identities)
    text = body.decode("utf-8-sig")
    if source == "sec":
        payload = json.loads(text)
        _reject_secrets(payload)
        if not isinstance(payload, dict):
            raise ValueError("SEC ticker mapping requires an object")
        rows = list(payload.values())
    else:
        reader = csv.DictReader(
            io.StringIO(text), delimiter="|" if source.startswith("nasdaq") else ","
        )
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Security CSV requires unique exact headers")
        rows = list(reader)
        if source.startswith("nasdaq"):
            footer = rows[-1] if rows else {}
            if not str(next(iter(footer.values()), "")).startswith("File Creation Time:"):
                raise ValueError("Nasdaq directory missing completeness footer")
            rows = rows[:-1]
        _reject_secrets(rows)
    if len(rows) > max_rows:
        raise ValueError("Security snapshot exceeds max_rows")
    records, seen = [], set()
    raw_hash = hashlib.sha256(body).hexdigest()
    for row in rows:
        if not isinstance(row, dict) or None in row:
            raise ValueError("Malformed security source row")
        ticker = (
            row.get("ticker")
            if source == "sec"
            else row.get("symbol")
            if source == "alpha_vantage"
            else row.get("Symbol")
            if source == "nasdaq"
            else row.get("ACT Symbol")
        )
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("Security source misses exact ticker column")
        if ticker not in identities:
            continue
        if ticker in seen:
            raise ValueError("Duplicate selected security ticker")
        seen.add(ticker)
        identity = identities[ticker]
        if (
            not isinstance(identity, dict)
            or not {"entity_id", "instrument_id", "venue", "currency"} <= identity.keys()
        ):
            raise ValueError(
                "Each selected ticker requires explicit permanent identities/venue/currency"
            )
        if source.startswith("nasdaq") and row.get("Test Issue") != "N":
            continue
        status = "active"
        if source == "alpha_vantage":
            status = {"Active": "active", "Delisted": "delisted"}.get(row.get("status"))
            if status is None:
                raise ValueError("Unknown listing status")
        token = json.dumps([source, identity, ticker, receipt.isoformat()], sort_keys=True)
        mapping = hashlib.sha256(token.encode()).hexdigest()
        records.append(
            dict(
                **{k: identity[k] for k in ("entity_id", "instrument_id", "venue", "currency")},
                ticker=ticker,
                status=status,
                mapping_id=mapping,
                valid_from=receipt,
                valid_to=None,
                available_at=receipt,
                ingested_at=receipt,
                source_id=source,
                revision_id=raw_hash,
                schema_version="1",
                availability_basis="receipt",
                point_in_time=False,
                source_document_id=source + ":" + raw_hash,
                cik=str(row["cik_str"]) if source == "sec" else identity.get("cik"),
                share_class=identity.get("share_class"),
                listing_date=row.get("ipoDate"),
                delisting_date=row.get("delistingDate"),
                provider_exchange=row.get("exchange", row.get("Exchange")),
                identity_basis="caller_explicit_mapping",
            )
        )
    return normalize_security_master(
        pd.DataFrame(
            records,
            columns=[
                *SECURITY_MASTER_COLUMNS,
                *sorted(set().union(*(r.keys() for r in records)) - set(SECURITY_MASTER_COLUMNS)),
            ],
        )
    )


def update_security_master(history, snapshot):
    """Append status/ticker changes atomically; absence never closes a listing."""
    snapshot = normalize_security_master(snapshot)
    old = normalize_security_master(history) if history is not None else snapshot.iloc[:0].copy()
    keys = ["source_id", "mapping_id", "revision_id"]
    if snapshot.empty:
        return old
    if not old.empty:
        existing = set(map(tuple, old[keys].to_numpy()))
        if all(tuple(row) in existing for row in snapshot[keys].to_numpy()):
            return old
    if snapshot.available_at.nunique() != 1:
        raise ValueError("Apply one receipt batch at a time")
    receipt = snapshot.available_at.iloc[0]
    if not old.empty and receipt <= old.available_at.max():
        raise ValueError("Apply new security snapshots in receipt order")
    known = security_master_asof(old, receipt)
    additions = []
    fields = ["entity_id", "ticker", "venue", "currency", "status", "cik", "share_class"]
    for _, row in snapshot.iterrows():
        prior = known.loc[
            known.source_id.eq(row.source_id) & known.instrument_id.eq(row.instrument_id)
        ]
        if len(prior) > 1:
            raise ValueError("Ambiguous security identity before snapshot")
        if len(prior):
            previous = prior.iloc[0]
            if all(
                (pd.isna(row[k]) and pd.isna(previous[k])) or row[k] == previous[k] for k in fields
            ):
                continue
            closed = previous.to_dict()
            closed.update(
                valid_to=receipt,
                available_at=receipt,
                ingested_at=receipt,
                revision_id=row.revision_id + ":close:" + row.mapping_id,
                source_document_id=row.source_document_id,
                availability_basis="receipt",
                point_in_time=False,
            )
            additions.append(closed)
        additions.append(row.to_dict())
    return (
        normalize_security_master(pd.concat([old, pd.DataFrame(additions)], ignore_index=True))
        if additions
        else old
    )


def security_universe_asof(history, decision_at, *, statuses=("active",), source_id=None):
    """Explicit eligible mappings known AND economically valid at the decision."""
    if not statuses or set(statuses) - {"active", "suspended", "delisted", "inactive"}:
        raise ValueError("Choose explicit universe statuses")
    known = security_master_asof(history, decision_at)
    if source_id is not None:
        known = known.loc[known.source_id.eq(source_id)]
    for _, group in known.groupby("instrument_id", sort=False):
        if any(
            group[name].nunique(dropna=False) > 1
            for name in ("entity_id", "ticker", "venue", "currency", "status")
        ):
            raise ValueError("Conflicting security sources; select an explicit source_id")
    known = known.loc[known.status.isin(statuses)]
    return known.drop_duplicates(["instrument_id", "entity_id", "ticker", "venue"]).reset_index(
        drop=True
    )


def collect_security_master(
    store,
    *,
    source,
    identities,
    snapshot_id,
    previous=None,
    offline=None,
    transport=None,
    api_key=None,
    user_agent=None,
    max_bytes=2_097_152,
    max_rows=20_000,
    date=None,
    state="active",
):
    """One bounded public directory/export; cache before parse, no bulk/history scan."""
    if source not in SECURITIES_URLS or not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Choose source and snapshot_id")
    if type(max_bytes) is not int or max_bytes < 1 or type(max_rows) is not int or max_rows < 1:
        raise ValueError("Security budgets must be positive integers")
    request = dict(
        snapshot_id=snapshot_id,
        identities=identities,
        date=date,
        state=state,
        max_rows=max_rows,
        max_bytes=max_bytes,
    )
    if offline is not None:
        request["offline_sha256"] = hashlib.sha256(offline[0]).hexdigest()
        request["receipt"] = _instant(offline[1], "received_at").isoformat()
    collection = store.begin(
        source,
        "security_snapshot",
        request,
        source_metadata={
            "historical_pit": "unqualified",
            "identity_basis": "caller_mapping",
            "source_url": SECURITIES_URLS[source],
        },
    )
    pages = {p["page_id"]: p for p in store.recover(collection)["pages"]}
    if "response" in pages:
        body = store.read_page(collection, "response")
        receipt = pages["response"]["metadata"]["received_at"]
    else:
        if offline is not None:
            body, receipt = offline
        else:
            params = {}
            headers = {}
            if source == "sec":
                from pfe.data.sec import SECClient

                SECClient(user_agent, store=store)  # Validate contact without issuing a call.
                headers["User-Agent"] = user_agent
            if source == "alpha_vantage":
                if not api_key or state not in ("active", "delisted"):
                    raise ValueError("Alpha Vantage needs a key and explicit listing state")
                params.update(function="LISTING_STATUS", apikey=api_key, state=state)
                if date is not None:
                    from datetime import date as date_type

                    params["date"] = date_type.fromisoformat(date).isoformat()
            from urllib.parse import urlsplit

            transport = transport or BoundedHTTPTransport(
                allowed_hosts=(urlsplit(SECURITIES_URLS[source]).hostname,),
                headers=headers,
                min_interval=1,
            )
            response = transport.get(
                SECURITIES_URLS[source], params=params, headers=headers, max_bytes=max_bytes
            )
            body, receipt = response.body, response.received_at
        if not isinstance(body, bytes) or len(body) > max_bytes:
            raise ValueError("Security response exceeds byte budget")
        if api_key and api_key.encode() in body:
            raise ValueError("Source response contains credentials")
        # Parse before persistence also rejects credential-bearing CSV/JSON.
        parse_security_snapshot(
            body, source=source, identities=identities, received_at=receipt, max_rows=max_rows
        )
        store.write_page(collection, "response", body, metadata={"received_at": str(receipt)})
    snapshot = parse_security_snapshot(
        body, source=source, identities=identities, received_at=receipt, max_rows=max_rows
    )
    history = update_security_master(previous, snapshot)
    issues = tuple(
        "selected_ticker_missing:" + ticker
        for ticker in identities
        if ticker not in set(snapshot.ticker)
    )
    if "normalized" not in pages:
        stream = io.BytesIO()
        snapshot.to_parquet(stream)
        store.write_page(
            collection,
            "normalized",
            stream.getvalue(),
            row_count=len(snapshot),
            complete=not issues,
        )
    return SecurityCollection(collection, snapshot, history, not issues, issues)
