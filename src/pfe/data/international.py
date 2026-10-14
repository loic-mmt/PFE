"""Small international captures, immutable replay and explicit source clocks.

One request is one selected response, never a claim of full historical coverage.
Source-specific parsers live in ``international_sources``. No optional SDK,
translation model, ZIP archive or full-market crawler is imported here.
"""

import hashlib
import inspect
import io
import json
import re
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

import pandas as pd

from pfe.data.contracts import _instant, normalize_observations
from pfe.data.eia import ContextError, ContextLimitError
from pfe.data.http import BoundedHTTPTransport
from pfe.data.qualification import sanitize_metadata
from pfe.data.sources import SourceRecord

INTERNATIONAL_SOURCES = {
    "tsmc": ("https://investor.tsmc.com/english/monthly-revenue", {"investor.tsmc.com"}),
    "mops": ("https://mops.twse.com.tw/", {"openapi.twse.com.tw", "mopsfin.twse.com.tw"}),
    "twse": ("https://www.twse.com.tw/en/trading/historical/stock-day.html", {"www.twse.com.tw"}),
    "opendart": ("https://engopendart.fss.or.kr/", {"engopendart.fss.or.kr"}),
    "eurostat": ("https://ec.europa.eu/eurostat/", {"ec.europa.eu"}),
    "boj": ("https://www.stat-search.boj.or.jp/index_en.html", {"www.stat-search.boj.or.jp"}),
    "estat": ("https://www.e-stat.go.jp/api/", {"api.e-stat.go.jp"}),
    "nbs": ("https://www.stats.gov.cn/english/", {"www.stats.gov.cn"}),
    "usgs": (
        "https://www.usgs.gov/centers/national-minerals-information-center",
        {"www.usgs.gov", "pubs.usgs.gov", "www.sciencebase.gov", "sciencebase.gov"},
    ),
    "gdelt": ("https://gdeltproject.org/data.html", {"api.gdeltproject.org"}),
    "gdelt_events": ("https://gdeltproject.org/data.html", {"data.gdeltproject.org"}),
}


@dataclass(frozen=True)
class InternationalLimits:
    max_rows: int = 5000
    max_response_bytes: int = 524_288
    max_archive_bytes: int = 4_194_304
    max_requests: int = 8

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ContextError("International limits must be positive integers")
        if self.max_response_bytes > self.max_archive_bytes:
            raise ContextError("Response budget must fit archive budget")


@dataclass(frozen=True)
class InternationalCollection:
    collection_id: str
    observations: pd.DataFrame
    complete: bool
    issues: tuple[str, ...]
    metadata: dict


def international_source_record(source):
    if source not in INTERNATIONAL_SOURCES:
        raise ContextError("Unknown international source")
    url, _ = INTERNATIONAL_SOURCES[source]
    notes = (
        "Public access; qualify selected data rights, attribution and reuse. "
        "Latest values do not establish historical vintages or publication times."
    )
    if source == "gdelt":
        notes += (
            " Article links do not grant rights to publisher text; detection is not publication."
        )
    if source == "usgs":
        notes += " MCS 2026 data release declares CC0; check the selected file's metadata."
    return SourceRecord(
        source,
        "lot4-2026-10-12",
        source.upper(),
        url,
        access="free",
        availability="receipt",
        license_notes=notes,
    )


def _code(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_@.\-]+", value):
        raise ContextError(f"{label} requires an explicit provider code")
    return value


def _bounded_url(source, url):
    try:
        parts = urlsplit(url)
        valid = (
            parts.scheme == "https"
            and parts.hostname in INTERNATIONAL_SOURCES[source][1]
            and parts.port in (None, 443)
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ContextError(
            "Source URL requires an official HTTPS host without query or credentials"
        )
    return url


def _request(source, config, limits):
    """Build narrow provider GETs. Parsing config is part of immutable identity."""
    options = dict(config)
    if source == "tsmc":
        year = options["year"]
        if type(year) is not int or not 1990 <= year <= 2100:
            raise ContextError("TSMC year must be explicit")
        return f"https://investor.tsmc.com/english/monthly-revenue/{year}", {}, options
    if source == "mops":
        return "https://openapi.twse.com.tw/v1/opendata/t187ap05_L", {}, options
    if source == "twse":
        month = options["month"]
        if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
            raise ContextError("TWSE requires one YYYY-MM month")
        pd.Period(month, freq="M")
        stock = _code(options["stock_no"], "TWSE stock_no")
        return (
            "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY",
            {"date": month.replace("-", "") + "01", "stockNo": stock, "response": "json"},
            options,
        )
    if source == "opendart":
        corp = options["corp_code"]
        if not isinstance(corp, str) or not re.fullmatch(r"\d{8}", corp):
            raise ContextError("OpenDART corp_code requires exactly eight digits")
        year, report = options["year"], options["report_code"]
        if (
            type(year) is not int
            or year < 2015
            or report not in ("11011", "11012", "11013", "11014")
        ):
            raise ContextError("OpenDART requires a fiscal year and report code")
        return (
            "https://engopendart.fss.or.kr/engapi/fnlttSinglAcnt.json",
            {"corp_code": corp, "bsns_year": year, "reprt_code": report},
            options,
        )
    if source == "eurostat":
        dataset = _code(options["dataset"], "Eurostat dataset")
        dimensions = options["dimensions"]
        if not isinstance(dimensions, dict) or not dimensions:
            raise ContextError("Eurostat requires explicit singleton dimensions")
        params = {"lang": "EN", "format": "JSON"}
        for name, value in dimensions.items():
            params[_code(name, "dimension")] = _code(value, "dimension value")
        for name in ("start", "end"):
            if name not in options:
                raise ContextError("Eurostat requires a bounded economic period range")
        params.update(sinceTimePeriod=options["start"], untilTimePeriod=options["end"])
        return (
            "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/" + dataset,
            params,
            options,
        )
    if source == "boj":
        code, db = _code(options["code"], "BoJ series"), _code(options["db"], "BoJ db")
        start, end = options["start"], options["end"]
        if (
            not all(
                isinstance(v, str) and re.fullmatch(r"\d{4}(?:\d{2})?", v) for v in (start, end)
            )
            or start > end
        ):
            raise ContextError("BoJ requires ordered YYYY or YYYYMM/QQ bounds")
        return (
            "https://www.stat-search.boj.or.jp/api/v1/getDataCode",
            {
                "format": "json",
                "lang": "en",
                "db": db,
                "code": code,
                "startDate": start,
                "endDate": end,
            },
            options,
        )
    if source == "estat":
        table = _code(options["table_id"], "e-Stat table")
        dimensions = options["dimensions"]
        if not isinstance(dimensions, dict) or not dimensions:
            raise ContextError("e-Stat requires exact dimensional selections")
        params = {
            "statsDataId": table,
            "lang": "E",
            "limit": limits.max_rows,
            "cdTimeFrom": options["start_code"],
            "cdTimeTo": options["end_code"],
        }
        for name, value in dimensions.items():
            if name not in ("area", "tab", *(f"cat{i:02d}" for i in range(1, 16))):
                raise ContextError("Unsupported e-Stat dimensional selector")
            params["cd" + name[0].upper() + name[1:]] = _code(value, "e-Stat dimension")
        return "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData", params, options
    if source in ("nbs", "usgs"):
        url = options.pop("url", None)
        filename = options.pop("file_name", None)
        params = {}
        if filename is not None:
            if (
                source != "usgs"
                or not isinstance(filename, str)
                or not re.fullmatch(r"[A-Za-z0-9_.\-]+\.csv", filename)
            ):
                raise ContextError("USGS file_name must select one plain CSV")
            params["name"] = filename
        return _bounded_url(source, url), params, options
    if source == "gdelt":
        from pfe.data.international_sources import gdelt_query_params

        return (
            "https://api.gdeltproject.org/api/v2/doc/doc",
            gdelt_query_params(options, limits),
            options,
        )
    if source == "gdelt_events":
        return (
            "https://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf",
            {},
            options,
        )
    raise ContextError("Unknown international source")


def collect_international(
    source,
    config,
    store,
    *,
    snapshot_id,
    transport=None,
    limits=None,
    offline=None,
    credential=None,
    source_metadata=None,
):
    """Capture one targeted source response or replay supplied (bytes, receipt).

    Keys are passed separately, excluded from capture identity, redacted before
    persistence. A cached capture needs no key and retains its first receipt.
    A response with a provider continuation or hit cap is explicitly partial;
    narrow the query or capture a selected next page separately.
    """
    from pfe.data.international_sources import PARSERS, response_issues

    if source not in PARSERS or not isinstance(config, dict):
        raise ContextError("Choose a supported international source and config object")
    if source == "gdelt_events" and offline is None:
        raise ContextError("GDELT Events requires a small local plain TSV; no ZIP/bulk download")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ContextError("snapshot_id must be explicit")
    limits = limits or InternationalLimits()
    # Strict JSON also rejects non-serializable config and credentials in RawStore.
    config = json.loads(json.dumps(config, allow_nan=False))
    try:
        url, params, options = _request(source, config, limits)
    except KeyError:
        raise ContextError("Missing required source query configuration") from None
    try:
        inspect.signature(PARSERS[source]).bind(
            b"", received_at="2000-01-01T00:00:00Z", max_rows=limits.max_rows, **options
        )
    except TypeError:
        raise ContextError("Missing or unsupported source parser configuration") from None
    request = {
        "snapshot_id": snapshot_id,
        "url": url,
        "params": params,
        "config": config,
        "limits": asdict(limits),
        "parser_version": "1",
    }
    if offline is not None:
        body, receipt = offline
        if not isinstance(body, bytes) or len(body) > limits.max_response_bytes:
            raise ContextLimitError("Offline response exceeds max_response_bytes")
        request.update(
            offline_sha256=hashlib.sha256(body).hexdigest(),
            offline_received_at=_instant(receipt, "received_at").isoformat(),
        )
    declaration = source_metadata or international_source_record(source).to_dict()
    collection = store.begin(source, "international_selected", request, source_metadata=declaration)
    manifest = store.recover(collection)
    pages = {p["page_id"]: p for p in manifest["pages"]}
    if "normalized" in pages and "summary" in pages:
        metadata = json.loads(store.read_page(collection, "summary"))
        return InternationalCollection(
            collection,
            normalize_observations(
                pd.read_parquet(io.BytesIO(store.read_page(collection, "normalized")))
            ),
            metadata["complete"],
            tuple(metadata["issues"]),
            metadata,
        )
    if "response" in pages:
        body = store.read_page(collection, "response")
        receipt = pages["response"]["metadata"]["received_at"]
    else:
        if offline is None:
            actual = params.copy()
            key_name = {"opendart": "crtfc_key", "estat": "appId"}.get(source)
            if key_name:
                if not isinstance(credential, str) or not credential.strip():
                    raise ContextError(f"{source} credential required for uncached collection")
                actual[key_name] = credential
            transport = transport or BoundedHTTPTransport(
                allowed_hosts=INTERNATIONAL_SOURCES[source][1],
                min_interval=1,
            )
            response = transport.get(url, params=actual, max_bytes=limits.max_response_bytes)
            body, receipt = response.body, response.received_at
        if len(body) > limits.max_response_bytes:
            raise ContextLimitError("Response exceeds max_response_bytes")
        if source in ("opendart", "estat"):
            try:
                payload = sanitize_metadata(json.loads(body), secrets=(credential or "",))
                if isinstance(payload, dict):
                    payload.pop("request", None)
                body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            except (UnicodeError, ValueError):
                raise ContextError("Invalid credential-bearing provider JSON") from None
            if credential and credential.encode() in body:
                raise ContextError(
                    "Provider response contains credential outside redacted metadata"
                )
        _instant(receipt, "received_at")
        if len(body) > min(limits.max_response_bytes, limits.max_archive_bytes):
            raise ContextLimitError("Sanitized response exceeds archive budget")
        store.write_page(collection, "response", body, metadata={"received_at": receipt})
    frame = PARSERS[source](body, received_at=receipt, max_rows=limits.max_rows, **options)
    issues = response_issues(source, body, config)
    metadata = {
        "source_id": source,
        "rows": len(frame),
        "complete": not issues,
        "issues": issues,
        "completion_scope": "selected_response_only",
        "universe_coverage": "unqualified",
        "historical_point_in_time": "unqualified",
        "received_at": receipt,
        "source_version": hashlib.sha256(body).hexdigest(),
        "source_bytes": len(body),
        "request": request,
    }
    stream = io.BytesIO()
    frame.to_parquet(stream)
    normalized = stream.getvalue()
    summary = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode()
    if len(body) + len(normalized) + len(summary) > limits.max_archive_bytes:
        raise ContextLimitError("Raw plus normalized payload exceeds max_archive_bytes")
    if max(len(normalized), len(summary)) > store.max_page_bytes:
        raise ContextLimitError("Normalized payload exceeds RawStore page budget")
    store.write_page(collection, "normalized", normalized, row_count=len(frame))
    store.write_page(collection, "summary", summary, complete=not issues)
    return InternationalCollection(collection, frame, not issues, tuple(issues), metadata)


def merge_international_observations(*captures):
    """Accumulate vintages without replacing earlier receipts on identical replay."""
    frames = [c.observations if isinstance(c, InternationalCollection) else c for c in captures]
    if not frames:
        raise ContextError("Provide at least one international capture")
    out = pd.concat([normalize_observations(frame) for frame in frames], ignore_index=True)
    keys = ["entity_id", "metric", "unit", "observation_date", "source_id", "revision_id"]
    out = out.sort_values("available_at", kind="stable")
    for _, group in out.groupby(keys, dropna=False):
        if len(group) > 1:
            values = [json.dumps(v, sort_keys=True, allow_nan=False) for v in group.value]
            if len(set(values)) > 1:
                raise ContextError("International revision identity has conflicting values")
    return normalize_observations(out.drop_duplicates(keys, keep="first"))
