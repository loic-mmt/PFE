"""Bounded SEC JSON ingestion with distinct acceptance and publication clocks.

Official API: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
Fair access: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
Company Facts covers standard-taxonomy entity facts, not all filing contexts.
"""

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from pfe.data.contracts import (
    OBSERVATION_COLUMNS,
    _identifiers,
    _instant,
    _missing,
    _provenance,
    _timestamps,
    normalize_observations,
)
from pfe.data.http import BoundedHTTPTransport
from pfe.data.sessions import _session_date
from pfe.data.sources import SourceRecord
from pfe.data.storage import RawStore

_ACCESSION = re.compile(r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z")
SEC_CONCEPT_METRICS = {
    "Revenues": "revenue",
    "SalesRevenueNet": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    "EarningsPerShareBasic": "eps",
    "EarningsPerShareDiluted": "eps",
    "NetIncomeLoss": "net_income",
    "OperatingIncomeLoss": "operating_income",
    "GrossProfit": "gross_profit",
    "Assets": "assets",
    "Liabilities": "liabilities",
    "StockholdersEquity": "equity",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "LongTermDebt": "long_term_debt",
    "LongTermDebtCurrent": "long_term_debt_current",
    "LongTermDebtNoncurrent": "long_term_debt_noncurrent",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
}
_STOCK_METRICS = {
    "assets",
    "liabilities",
    "equity",
    "cash",
    "long_term_debt",
    "long_term_debt_current",
    "long_term_debt_noncurrent",
}
_FACT_EXTRAS = (
    "financial_metric",
    "taxonomy",
    "concept",
    "period_start",
    "period_end",
    "period_kind",
    "basis",
    "eps_basis",
    "entity_scope",
    "currency",
    "accession",
    "accepted_at",
    "form",
    "filed_date",
    "filing_fiscal_year",
    "filing_fiscal_period",
    "frame",
    "context_json",
    "coverage_complete",
    "quality.missing_publication",
    "quality.period_inferred",
    "quality.unmapped_metric",
    "statement_type",
    "cik",
)


def _cik(value):
    if isinstance(value, bool):
        raise ValueError("CIK requires 1 to 10 digits")
    text = str(value)
    if not re.fullmatch(r"[0-9]{1,10}", text) or int(text) == 0:
        raise ValueError("CIK requires 1 to 10 digits and must be positive")
    return text.zfill(10)


def _delay(value):
    result = pd.Timedelta(value)
    if pd.isna(result) or result < pd.Timedelta(0):
        raise ValueError("latency requires a non-negative fixed duration")
    return result


def _availability(published, receipt, mode, latency, *, accepted=pd.NaT):
    if mode not in ("receipt", "publication"):
        raise ValueError("availability must be receipt or publication")
    if mode == "publication":
        anchor = published if pd.notna(published) else accepted
        if pd.notna(anchor):
            return anchor + latency, "reconstructed"
    return max(receipt, published + latency) if pd.notna(published) else receipt, "receipt"


def normalize_sec_filings(
    payload,
    *,
    received_at,
    cik=None,
    entity_id=None,
    availability="receipt",
    latency="0s",
    publication_times=None,
):
    """Normalize submissions recent/history column arrays; no filed-date clock.

    An aware acceptanceDateTime becomes accepted_at, never published_at.
    publication_times optionally maps accessions to independently verified aware
    publication timestamps. Without it published_at remains null: acceptance can
    precede public dissemination. Missing/naive acceptance stays raw metadata.
    A filingDate/reportDate remains a date. Legacy availability="publication"
    reconstructs availability from acceptance+latency when publication is unknown;
    it keeps PIT False and does not invent a next-business-day publication clock.
    Current tickers/exchanges are deliberately not turned into historic mappings.
    """
    if not isinstance(payload, dict):
        raise ValueError("SEC submissions require an object")
    identifier = _cik(cik if cik is not None else payload.get("cik"))
    if "cik" in payload and _cik(payload["cik"]) != identifier:
        raise ValueError("SEC response CIK differs from requested filer")
    receipt, delay = _instant(received_at, "received_at"), _delay(latency)
    _availability(pd.NaT, receipt, availability, delay)
    if publication_times is None:
        publication_times = {}
    if not isinstance(publication_times, dict):
        raise ValueError("publication_times requires an accession to verified timestamp mapping")
    publications = {}
    for accession, timestamp in publication_times.items():
        if not isinstance(accession, str) or not _ACCESSION.fullmatch(accession):
            raise ValueError("Invalid SEC publication accession")
        publications[accession] = (
            pd.NaT if _missing(timestamp) else _instant(timestamp, "published_at")
        )
    columns = payload.get("filings", {}).get("recent", payload)
    accessions = columns.get("accessionNumber", [])
    if not isinstance(accessions, list):
        raise ValueError("SEC accessionNumber must be an array")
    count = len(accessions)
    fields = (
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "isXBRL",
        "isInlineXBRL",
    )
    for field in fields:
        if field in columns and (
            not isinstance(columns[field], list) or len(columns[field]) != count
        ):
            raise ValueError("SEC submissions arrays must have equal lengths")
    records = []
    for index, accession in enumerate(accessions):
        if not isinstance(accession, str) or not _ACCESSION.fullmatch(accession):
            raise ValueError("Invalid SEC accession number")
        row = {field: columns.get(field, [None] * count)[index] for field in fields}
        raw_acceptance = row.pop("acceptanceDateTime")
        try:
            accepted = _instant(raw_acceptance, "acceptanceDateTime")
        except ValueError:
            accepted = pd.NaT
        published = publications.get(accession, pd.NaT)
        available, basis = _availability(published, receipt, availability, delay, accepted=accepted)
        for field in ("filingDate", "reportDate"):
            row[field] = (
                None if _missing(row[field]) or row[field] == "" else _session_date(row[field])
            )
        records.append(
            dict(
                **row,
                accession=accession,
                cik=identifier,
                entity_id=entity_id if entity_id is not None else f"sec:cik:{identifier}",
                source_id="sec",
                source_document_id=accession,
                accepted_at=accepted,
                published_at=published,
                available_at=available,
                ingested_at=receipt,
                availability_basis=basis,
                point_in_time=False,
                acceptance_datetime_raw=raw_acceptance,
            )
        )
    out = pd.DataFrame(
        records,
        columns=[
            *fields[:2],
            "form",
            "primaryDocument",
            "isXBRL",
            "isInlineXBRL",
            "accession",
            "cik",
            "entity_id",
            "source_id",
            "source_document_id",
            "accepted_at",
            "published_at",
            "available_at",
            "ingested_at",
            "availability_basis",
            "point_in_time",
            "acceptance_datetime_raw",
        ],
    )
    if out.accession.duplicated().any():
        raise ValueError("Duplicate SEC accession in one submissions page")
    _identifiers(out, ("entity_id", "accession", "source_id"))
    _provenance(out)
    out["accepted_at"] = _timestamps(out.accepted_at, "accepted_at", nullable=True)
    return out.sort_values("available_at", kind="stable", ignore_index=True)


def _period(start, end, role):
    if start is None:
        return "instant" if role is None or role in _STOCK_METRICS else "unknown"
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    if days <= 0:
        raise ValueError("SEC fact period end must not precede start")
    if 330 <= days <= 400:
        return "annual"
    if 60 <= days <= 120:
        return "quarter"
    if 120 < days < 330:
        return "ytd"
    return "unknown"


def normalize_company_facts(
    payload, *, received_at, filings=None, entity_id=None, availability="receipt", latency="0s"
):
    """Map XBRL concepts/units/periods into versioned observations.

    accession -> submissions accepted_at preserves acceptance, not publication.
    Only explicitly verified filings published_at establishes publication. Legacy
    availability="publication" reconstructs from acceptance when publication is
    unknown, keeping PIT False. Filed dates never establish a timestamp.
    Numeric values/unit scales are unchanged.
    Metric identity includes taxonomy/concept/unit/start/basis/EPS/scope, avoiding
    collisions between annual, quarterly and YTD facts with the same end date.
    Revision ID includes accession and factual content; duplicates with identical
    context/value are collapsed, conflicting same-accession values are rejected.

    fy/fp describe the filing context and are named filing_fiscal_*: comparative
    facts do not necessarily belong to that fiscal year/quarter. period_kind is
    duration-based, explicitly flagged as inferred; exact dates are retained.
    Segmented/custom/non-GAAP facts remain distinguishable, never claimed as a
    complete entity-level GAAP dataset. Current API post-acceptance corrections
    and historic delivery cannot be verified; PIT remains False in both modes.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), dict):
        raise ValueError("SEC companyfacts requires a facts object")
    identifier = _cik(payload.get("cik"))
    receipt, delay = _instant(received_at, "received_at"), _delay(latency)
    _availability(pd.NaT, receipt, availability, delay)
    publications, acceptances = {}, {}
    if filings is not None:
        if not isinstance(filings, pd.DataFrame) or not {"accession", "published_at"}.issubset(
            filings
        ):
            raise ValueError("filings require accession and published_at")
        if filings.accession.duplicated().any():
            raise ValueError("Resolve duplicate filings before mapping Company Facts")
        publications = dict(
            zip(
                filings.accession,
                _timestamps(filings.published_at, "published_at", nullable=True),
                strict=True,
            )
        )
        if "accepted_at" in filings:
            acceptances = dict(
                zip(
                    filings.accession,
                    _timestamps(filings.accepted_at, "accepted_at", nullable=True),
                    strict=True,
                )
            )
    records, seen = [], {}
    for taxonomy, concepts in payload["facts"].items():
        if not isinstance(concepts, dict):
            raise ValueError("SEC taxonomy must contain concepts")
        for concept, metadata in concepts.items():
            if not isinstance(metadata, dict) or not isinstance(metadata.get("units"), dict):
                raise ValueError("SEC concept requires units")
            role = SEC_CONCEPT_METRICS.get(concept) if taxonomy == "us-gaap" else None
            for unit, rows in metadata["units"].items():
                if not isinstance(rows, list):
                    raise ValueError("SEC fact units require arrays")
                for fact in rows:
                    if not isinstance(fact, dict) or not {"end", "val", "accn"}.issubset(fact):
                        raise ValueError("SEC facts require end, val and accn")
                    if (
                        isinstance(fact["val"], (bool, np.bool_))
                        or not isinstance(fact["val"], Real)
                        or not np.isfinite(fact["val"])
                    ):
                        raise ValueError("SEC fact val requires a finite number")
                    accession = fact["accn"]
                    if not isinstance(accession, str) or not _ACCESSION.fullmatch(accession):
                        raise ValueError("Invalid SEC fact accession")
                    end = _session_date(fact["end"])
                    start = None if _missing(fact.get("start")) else _session_date(fact["start"])
                    kind = _period(start, end, role)
                    basis = fact.get(
                        "basis",
                        metadata.get(
                            "basis",
                            {"us-gaap": "us_gaap", "ifrs-full": "ifrs"}.get(taxonomy, "unknown"),
                        ),
                    )
                    if basis not in ("us_gaap", "ifrs", "non_gaap", "unknown"):
                        raise ValueError("Unknown financial accounting basis")
                    context = {key: fact[key] for key in ("segment", "dimensions") if key in fact}
                    scope = (
                        "segment"
                        if context
                        else "entity"
                        if taxonomy in ("us-gaap", "ifrs-full", "dei", "srt")
                        else "unknown"
                    )
                    eps = (
                        "basic"
                        if concept == "EarningsPerShareBasic"
                        else "diluted"
                        if concept == "EarningsPerShareDiluted"
                        else None
                    )
                    currency = (
                        unit[:3]
                        if isinstance(unit, str) and re.fullmatch(r"[A-Z]{3}(?:/shares)?", unit)
                        else None
                    )
                    context_json = json.dumps(context, sort_keys=True, allow_nan=False)
                    identity = (
                        taxonomy,
                        concept,
                        unit,
                        start,
                        end,
                        basis,
                        eps,
                        scope,
                        context_json,
                        accession,
                    )
                    if identity in seen:
                        if seen[identity] != fact["val"]:
                            raise ValueError(
                                "Conflicting SEC fact values for one accession/context"
                            )
                        continue
                    seen[identity] = fact["val"]
                    published = publications.get(accession, pd.NaT)
                    accepted = acceptances.get(accession, pd.NaT)
                    available, provenance = _availability(
                        published, receipt, availability, delay, accepted=accepted
                    )
                    content = json.dumps([*identity, fact["val"]], allow_nan=False)
                    revision = f"{accession}:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
                    context_key = hashlib.sha256(context_json.encode()).hexdigest()[:12]
                    metric = (
                        f"sec:{taxonomy}:{concept}:{unit}:{start or 'instant'}:"
                        f"{basis}:{eps or 'na'}:{scope}:{context_key}"
                    )
                    records.append(
                        dict(
                            entity_id=entity_id
                            if entity_id is not None
                            else f"sec:cik:{identifier}",
                            cik=identifier,
                            observation_date=end,
                            metric=metric,
                            unit=unit,
                            value=fact["val"],
                            source_id="sec",
                            revision_id=revision,
                            schema_version="1",
                            source_document_id=accession,
                            accepted_at=accepted,
                            published_at=published,
                            available_at=available,
                            ingested_at=receipt,
                            availability_basis=provenance,
                            point_in_time=False,
                            financial_metric=role,
                            taxonomy=taxonomy,
                            concept=concept,
                            period_start=start,
                            period_end=end,
                            period_kind=kind,
                            basis=basis,
                            eps_basis=eps,
                            entity_scope=scope,
                            currency=currency,
                            accession=accession,
                            form=fact.get("form"),
                            filed_date=None
                            if _missing(fact.get("filed"))
                            else _session_date(fact["filed"]),
                            filing_fiscal_year=fact.get("fy"),
                            filing_fiscal_period=fact.get("fp"),
                            frame=fact.get("frame"),
                            context_json=context_json,
                            coverage_complete=False,
                            statement_type="reported_actual",
                            **{
                                "quality.missing_publication": pd.isna(published),
                                "quality.period_inferred": start is not None,
                                "quality.unmapped_metric": role is None,
                            },
                        )
                    )
    out = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=[*OBSERVATION_COLUMNS, *_FACT_EXTRAS])
    )
    # Several historical declarations are learned at the same receipt. Keep the
    # latest justified declaration in that batch, not invented delivery offsets.
    if not out.empty:
        keep = []
        keys = ["entity_id", "metric", "observation_date", "source_id", "available_at"]
        for _, batch in out.groupby(keys, dropna=False, sort=False):
            if len(batch) == 1:
                keep.append(batch.index[0])
                continue
            order = "accepted_at" if batch.accepted_at.notna().all() else "filed_date"
            if batch[order].isna().any():
                latest = batch
            else:
                latest = batch.loc[batch[order] == batch[order].max()]
            if latest.value.nunique(dropna=False) > 1:
                raise ValueError("Ambiguous simultaneous SEC fact vintages")
            keep.append(latest.index[-1])
        out = out.loc[keep].reset_index(drop=True)
    result = normalize_observations(out)
    result["accepted_at"] = _timestamps(result.accepted_at, "accepted_at", nullable=True)
    result.attrs["availability_mode"] = availability
    return result


def merge_company_facts(*frames):
    """Accumulate SEC audits without rewriting already-known factual vintages.

    Identical revision IDs/context/value retain their earliest availability and
    its original metadata. Refreshed receipts are not new historical vintages.
    Changed facts need a different revision ID and keep their new receipt time.
    Receipt and legacy publication-reconstruction policies cannot be merged. A
    publication-mode snapshot may contain receipt fallbacks for unknown clocks;
    its availability_mode attr retains that declared collection policy.
    Within one revision, changed value/context/publication or availability_basis
    is rejected. Different simultaneous revisions remain ambiguous and rejected.
    """
    if not frames:
        return normalize_observations(pd.DataFrame(columns=[*OBSERVATION_COLUMNS, *_FACT_EXTRAS]))
    normalized, modes = [], set()
    for frame in frames:
        data = normalize_observations(frame)
        if not set(_FACT_EXTRAS).issubset(data):
            raise ValueError("merge_company_facts requires normalized SEC fact frames")
        data["accepted_at"] = _timestamps(data.accepted_at, "accepted_at", nullable=True)
        mode = frame.attrs.get("availability_mode")
        if mode is None and not data.empty:
            bases = set(data.availability_basis)
            if len(bases) != 1 or not bases <= {"receipt", "reconstructed"}:
                raise ValueError("Specify a single SEC collection availability policy")
            mode = "publication" if bases == {"reconstructed"} else "receipt"
        if mode is not None:
            modes.add(mode)
        normalized.append(data)
    if len(modes) > 1:
        raise ValueError("Cannot mix receipt and reconstruction availability policies")
    merged = pd.concat(normalized, ignore_index=True)
    keys = ["entity_id", "metric", "observation_date", "source_id", "instrument_id", "revision_id"]
    context = (
        "value",
        "unit",
        "source_document_id",
        "schema_version",
        "financial_metric",
        "taxonomy",
        "concept",
        "period_start",
        "period_end",
        "period_kind",
        "basis",
        "eps_basis",
        "entity_scope",
        "currency",
        "accession",
        "accepted_at",
        "context_json",
        "statement_type",
        "cik",
    )
    keep = []
    for _, group in merged.groupby(keys, dropna=False, sort=False):
        original = group.sort_values("available_at", kind="stable").iloc[0]
        for _, row in group.iterrows():
            for field in context:
                a, b = original[field], row[field]
                a_missing, b_missing = _missing(a), _missing(b)
                if a_missing != b_missing or (not a_missing and a != b):
                    raise ValueError("SEC revision_id conflicts with immutable value/context")
        if group.availability_basis.nunique() != 1 or group.published_at.dropna().nunique() > 1:
            raise ValueError("SEC revision_id conflicts with availability/publication policy")
        keep.append(original.name)
    result = normalize_observations(merged.loc[keep].reset_index(drop=True))
    if modes:
        result.attrs["availability_mode"] = next(iter(modes))
    result.attrs["raw_collection_ids"] = list(
        dict.fromkeys(
            frame.attrs["raw_collection_id"]
            for frame in frames
            if "raw_collection_id" in frame.attrs
        )
    )
    return result


@dataclass(frozen=True)
class SECData:
    filings: pd.DataFrame
    facts: pd.DataFrame
    collection_id: str
    complete: bool


class SECClient:
    """Serial requests strictly below SEC's 10/s limit; immutable bounded cache.

    User-Agent must identify an application/contact email. Cached pages preserve
    receipt times and do not issue HTTP requests on replay. max_pages covers the
    recent submissions page, Company Facts and optional history pages; caps are
    reported, never silently labelled complete. One client should be shared by
    parallel callers to enforce its request budget; transport remains injectable.
    """

    def __init__(
        self,
        user_agent,
        *,
        transport=None,
        store=None,
        max_requests_per_second=8,
        max_bytes=4194304,
    ):
        contact = re.search(r"\S+@\S+\.\S+", user_agent) if isinstance(user_agent, str) else None
        if (
            not isinstance(user_agent, str)
            or contact is None
            or not user_agent[: contact.start()].strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in user_agent)
        ):
            raise ValueError("SEC User-Agent must identify an application and contact email")
        if (
            isinstance(max_requests_per_second, bool)
            or not isinstance(max_requests_per_second, Real)
            or not np.isfinite(max_requests_per_second)
            or not 0 < max_requests_per_second < 10
        ):
            raise ValueError("SEC request rate must be positive and strictly below 10/s")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes requires a positive integer")
        self.transport = transport or BoundedHTTPTransport(allowed_hosts=("data.sec.gov",))
        self.store = (
            store
            if isinstance(store, RawStore)
            else RawStore(
                Path("data/raw/sec") if store is None else store, max_page_bytes=max_bytes
            )
        )
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}
        self.max_bytes, self.interval = max_bytes, 1 / max_requests_per_second
        self._lock, self._last = threading.Lock(), None

    def _fetch(self, collection, page_id, url):
        manifest = self.store.manifest(collection)
        saved = next((page for page in manifest["pages"] if page["page_id"] == page_id), None)
        if saved:
            raw, received = (
                self.store.read_page(collection, page_id),
                saved["metadata"]["received_at"],
            )
        else:
            with self._lock:
                now = time.monotonic()
                if self._last is not None and now - self._last < self.interval:
                    time.sleep(self.interval - (now - self._last))
                self._last = time.monotonic()
                response = self.transport.get(url, max_bytes=self.max_bytes, headers=self.headers)
            raw, received = response.body, _instant(response.received_at, "received_at").isoformat()
            if not isinstance(raw, bytes) or len(raw) > self.max_bytes:
                raise ValueError("SEC response exceeds max_bytes or is not bytes")
            self.store.write_page(
                collection,
                page_id,
                raw,
                next_cursor=page_id,
                complete=False,
                metadata={"url": url, "received_at": received},
            )
        try:
            payload = json.loads(
                raw,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError("Non-finite SEC JSON")
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid SEC JSON response") from error
        if not isinstance(payload, dict):
            raise ValueError("SEC response must be a JSON object")
        return payload, received

    def download(
        self,
        cik,
        *,
        audit_id,
        include_history=True,
        max_pages=20,
        max_filings=1000,
        forms=None,
        start=None,
        end=None,
        availability="receipt",
        latency="0s",
    ):
        """Collect recent/history submission metadata and Company Facts, no bulk.

        start inclusive/end exclusive filter filing DATE, not publication time.
        History caps retain unknown-publication facts at receipt time. Completion
        describes requested HTTP pages and selected filing rows, not SEC economic
        completeness, all custom concepts, segments or point-in-time delivery.
        A new audit_id is required for a refreshed provider snapshot.
        """
        identifier = _cik(cik)
        if availability not in ("receipt", "publication"):
            raise ValueError("availability must be receipt or publication")
        if not isinstance(audit_id, str) or not audit_id.strip() or audit_id != audit_id.strip():
            raise ValueError("audit_id requires a non-empty trimmed string")
        if (
            any(isinstance(n, bool) or not isinstance(n, int) for n in (max_pages, max_filings))
            or max_pages < 2
            or max_filings < 1
        ):
            raise ValueError("Require max_pages >= 2 and max_filings >= 1")
        if not isinstance(include_history, bool):
            raise ValueError("include_history requires a boolean")
        if isinstance(forms, str):
            raise ValueError("forms requires a list of form names, not one string")
        forms = None if forms is None else list(forms)
        if forms is not None and (
            not forms or any(not isinstance(form, str) or not form for form in forms)
        ):
            raise ValueError("forms require a non-empty list of form names")
        first, last = (
            None if start is None else _session_date(start),
            None if end is None else _session_date(end),
        )
        if first is not None and last is not None and first >= last:
            raise ValueError("start must precede end")
        source = SourceRecord(
            "sec",
            "edgar-api-v1",
            "SEC EDGAR",
            "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
            access="free",
            availability="receipt",
            license_notes="Public API; no complete-context or historical-delivery guarantee.",
        )
        request = dict(
            cik=identifier,
            audit_id=audit_id,
            include_history=include_history,
            max_pages=max_pages,
            max_filings=max_filings,
            forms=forms,
            start=first,
            end=last,
            availability=availability,
            latency=str(_delay(latency)),
        )
        collection = self.store.begin(
            "sec", "submissions_companyfacts", request, source_metadata=source.to_dict()
        )
        self.store.recover(collection)
        recent, recent_receipt = self._fetch(
            collection, "submissions", f"https://data.sec.gov/submissions/CIK{identifier}.json"
        )
        options = dict(cik=identifier, availability=availability, latency=latency)
        tables = [normalize_sec_filings(recent, received_at=recent_receipt, **options)]
        files = recent.get("filings", {}).get("files", [])
        if not isinstance(files, list):
            raise ValueError("SEC submissions files must be an array")
        selected_files = files[: max_pages - 2] if include_history else []
        for file in selected_files:
            name = file.get("name") if isinstance(file, dict) else None
            if not isinstance(name, str) or not re.fullmatch(
                rf"CIK{identifier}-submissions-[0-9]+\.json", name
            ):
                raise ValueError("Invalid SEC historical submissions filename")
            history, receipt = self._fetch(
                collection, name, f"https://data.sec.gov/submissions/{name}"
            )
            tables.append(normalize_sec_filings(history, received_at=receipt, **options))
        filings = pd.concat(tables, ignore_index=True)
        for _, duplicate in filings.groupby("accession", sort=False):
            if duplicate.accepted_at.dropna().nunique() > 1:
                raise ValueError("Conflicting SEC acceptance timestamps")
        filings = (
            filings.sort_values(
                ["accepted_at", "filingDate", "available_at"], na_position="first", kind="stable"
            )
            .drop_duplicates("accession", keep="last")
            .reset_index(drop=True)
        )
        facts_payload, facts_receipt = self._fetch(
            collection,
            "companyfacts",
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{identifier}.json",
        )
        if _cik(facts_payload.get("cik")) != identifier:
            raise ValueError("SEC Company Facts CIK differs from requested filer")
        facts = normalize_company_facts(
            facts_payload,
            received_at=facts_receipt,
            filings=filings,
            availability=availability,
            latency=latency,
        )
        selected = filings
        if forms is not None:
            selected = selected.loc[selected.form.isin(forms)]
        if first is not None:
            selected = selected.loc[selected.filingDate >= first]
        if last is not None:
            selected = selected.loc[selected.filingDate < last]
        capped = len(selected) > max_filings
        selected = selected.iloc[:max_filings].reset_index(drop=True)
        if forms is not None or first is not None or last is not None:
            facts = facts.loc[facts.accession.isin(selected.accession)].reset_index(drop=True)
        complete = not capped and (not include_history or len(selected_files) == len(files))
        summary = dict(
            complete=complete,
            filings=len(selected),
            facts=len(facts),
            history_pages=len(selected_files),
            capped=capped,
            history_capped=include_history and len(selected_files) < len(files),
        )
        if not any(
            page["page_id"] == "summary" for page in self.store.manifest(collection)["pages"]
        ):
            self.store.write_page(
                collection,
                "summary",
                json.dumps(summary, sort_keys=True).encode(),
                row_count=len(selected),
                complete=complete,
                metadata={"partial": not complete},
            )
        selected.attrs["raw_collection_id"] = facts.attrs["raw_collection_id"] = collection
        return SECData(selected, facts, collection, complete)
