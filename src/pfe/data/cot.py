"""Targeted CFTC Socrata COT reports; report date never means publication."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from pfe.data.captures import CaptureLimits, capture_observations
from pfe.data.contracts import _instant, normalize_observations
from pfe.data.eia import _observations, _period
from pfe.data.http import BoundedHTTPTransport
from pfe.data.international_sources import _json, _number, _point
from pfe.data.sources import SourceRecord

COT_DATASETS = {
    "legacy_futures": "6dca-aqww",
    "legacy_combined": "jun7-fc8e",
    "disaggregated_futures": "72hh-3qpy",
    "disaggregated_combined": "kh3c-gbw2",
    "tff_futures": "gpe5-46if",
    "tff_combined": "yw9f-hn96",
}


@dataclass(frozen=True)
class COTQuery:
    report: str
    market_code: str
    entity_id: str
    start: str
    end: str
    columns: dict = field(default_factory=lambda: {"open_interest": "open_interest_all"})
    publication_times: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.report not in COT_DATASETS or not re.fullmatch(r"[A-Za-z0-9]{6}", self.market_code):
            raise ValueError("Select COT report and exact six-character market code")
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("COT entity_id is required")
        for value in (self.start, self.end):
            _period(value, "daily")
        if self.start > self.end:
            raise ValueError("COT range must be ordered")
        if not self.columns or any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", k) or not re.fullmatch(r"[a-z][a-z0-9_]*", v)
            for k, v in self.columns.items()
        ):
            raise ValueError("COT columns must map exact metrics to provider field names")
        for day, stamp in self.publication_times.items():
            _period(day, "daily")
            if _instant(stamp, "published_at").date().isoformat() < day:
                raise ValueError("COT release cannot precede economic observation")


def cot_source_record():
    return SourceRecord(
        "cftc",
        "lot5-2026-10-14",
        "CFTC COT",
        "https://publicreporting.cftc.gov/stories/s/r4w3-av2u",
        access="free",
        availability="receipt",
        license_notes=(
            "Public CFTC reports; selected dataset and release history must be qualified."
        ),
    )


def parse_cot(body, *, query, received_at, max_rows=1000):
    query = COTQuery(**query) if isinstance(query, dict) else query
    rows = _json(body)
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ValueError("COT response must be a bounded array")
    records = []
    for row in rows:
        if not isinstance(row, dict) or row.get("cftc_contract_market_code") != query.market_code:
            raise ValueError("COT market differs from explicit selection")
        label = row.get("report_date_as_yyyy_mm_dd", "")
        if not isinstance(label, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}(?:T00:00:00(?:\.000)?)?", label
        ):
            raise ValueError("COT report date must be a date-only economic label")
        day = label[:10]
        if not query.start <= day <= query.end:
            raise ValueError("COT observation outside selected range")
        for metric, column in query.columns.items():
            if column not in row:
                raise ValueError("COT response misses an explicitly selected position column")
            value = _number(row[column])
            if value is not None and (value < 0 or value % 1):
                raise ValueError("COT positions must be nonnegative integer contracts")
            record = _point(
                body,
                received_at=received_at,
                source="cftc",
                entity_id=query.entity_id,
                metric="cot." + query.report + "." + metric,
                unit="contracts",
                period=day,
                frequency="daily",
                value=value,
                country="US",
                published_at=query.publication_times.get(day),
                evidence=json.dumps(row, sort_keys=True),
                report=query.report,
                market_code=query.market_code,
                provider_column=column,
            )
            record.update(
                frequency="weekly",
                revision_id=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
            )
            records.append(record)
    return normalize_observations(pd.DataFrame(records)) if records else _observations([])


def collect_cot(query, store, *, snapshot_id, transport=None, offline=None, limits=None):
    query = COTQuery(**query) if isinstance(query, dict) else query
    limits = limits or CaptureLimits()
    # rows limit includes selected scalar columns, not only provider records.
    row_limit = max(1, limits.max_rows // len(query.columns))
    params = {
        "$where": (
            f"cftc_contract_market_code='{query.market_code}' AND "
            f"report_date_as_yyyy_mm_dd between '{query.start}T00:00:00' "
            f"and '{query.end}T00:00:00'"
        ),
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": row_limit,
    }

    def fetch(*, max_bytes):
        client = transport or BoundedHTTPTransport({"publicreporting.cftc.gov"}, min_interval=1)
        return client.get(
            "https://publicreporting.cftc.gov/resource/" + COT_DATASETS[query.report] + ".json",
            params=params,
            max_bytes=max_bytes,
        )

    def parser(body, *, received_at, max_rows):
        frame = parse_cot(body, query=query, received_at=received_at, max_rows=max_rows)
        issues = ("provider_row_cap_coverage_unknown",) if len(_json(body)) >= row_limit else ()
        return frame, issues

    return capture_observations(
        store=store,
        source="cftc",
        dataset="cot_selected",
        snapshot_id=snapshot_id,
        query=asdict(query),
        parser=parser,
        declaration=cot_source_record().to_dict(),
        fetch=fetch,
        offline=offline,
        limits=limits,
    )
