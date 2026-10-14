"""Bounded LSE metadata discovery, with conservative coverage qualification.

This module never downloads prices or infers an account's data entitlements.
``available`` means listed in the supplied metadata, not validated market data.
Offline catalogues must explicitly assert completeness and authority before a
missing exact symbol can be labelled ``absent``. Units, publication timing and
historical PIT are independently unqualified until independently established.
"""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

VAULT_URL = "https://api.londonstrategicedge.com/vault"
DISCOVERY_ENDPOINTS = ("/meta", "/catalog", "/reference", "/usage")
SDK_CLIENT_URL = "https://github.com/londonstrategicedge/lse-data/blob/main/lse/client.py"
SDK_VAULT_URL = "https://github.com/londonstrategicedge/lse-data/blob/main/lse/vault.py"
SDK_CHANGELOG_URL = "https://github.com/londonstrategicedge/lse-data/blob/main/CHANGELOG.md"
IMPLEMENTATION_BASELINE = {
    "revision_id": "lot2-2026-10-12",
    "verified_at": "2026-10-12T00:00:00+00:00",
    "scope": "static repository implementation, not provider availability or account access",
}
QualificationStatus = Literal["available", "absent", "unqualified", "local", "not-applicable"]
_CREDENTIAL_FIELDS = frozenset(
    (
        "api_key",
        "apikey",
        "x-api-key",
        "authorization",
        "access_token",
        "token",
        "password",
        "secret",
        "credentials",
        "headers",
        "key",
        "auth",
        "cookie",
        "set-cookie",
        "passwd",
        "access_key",
        "secret_key",
        "private_key",
        "session_key",
    )
)
_NORMALIZED_CREDENTIAL_FIELDS = frozenset(
    "".join(char for char in field if char.isalnum()) for field in _CREDENTIAL_FIELDS
)


class DiscoveryError(ValueError):
    """An invalid metadata response, without provider body or credential details."""


class DiscoveryLimitError(DiscoveryError):
    """A response or audit would exceed a caller's explicit resource budget."""


@dataclass(frozen=True)
class DiscoveryLimits:
    """Total raw response bytes, rows per catalogue, and requests per invocation."""

    max_rows: int = 25_000
    max_bytes: int = 1_048_576
    max_requests: int = 4

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DiscoveryResponse:
    payload: Any
    byte_count: int


class DiscoveryTransport(Protocol):
    def get(self, endpoint: str, *, max_bytes: int) -> DiscoveryResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # API credentials must never follow a redirect to another host.
        return None


def _validate_lse_http_body(data, *, status, headers):
    """Reject partial HTTP evidence before a collector can assert completion."""
    content_range = None if headers is None else headers.get("Content-Range")
    if content_range is not None or status is not None and status != 200:
        raise DiscoveryError("LSE GET did not return a complete HTTP 200 response")
    content_length = None if headers is None else headers.get("Content-Length")
    transfer_encoding = None if headers is None else headers.get("Transfer-Encoding")
    if content_length is not None and transfer_encoding is None:
        if not isinstance(content_length, str) or not content_length.strip().isdigit():
            raise DiscoveryError("LSE Content-Length is invalid")
        try:
            size = int(content_length)
        except ValueError:
            raise DiscoveryError("LSE Content-Length is invalid") from None
        if size != len(data):
            raise DiscoveryError("LSE response differs from Content-Length")


class BoundedLSETransport:
    """GET-only stdlib transport for four SDK metadata paths; no export access.

    SDK ``catalog`` has no byte or row limit, so using its unbounded response
    reader would defeat the audit's cap. The transport enforces the cap before
    JSON decoding, rejects redirects, and never surfaces provider error bodies.
    """

    def __init__(self, api_key: str, *, timeout: float = 30, opener=None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("LSE_API_KEY is required")
        if any(ord(char) < 32 or ord(char) == 127 for char in api_key):
            raise ValueError("LSE_API_KEY must not contain control characters")
        if timeout <= 0 or timeout > 60:
            raise ValueError("timeout must be between 0 and 60 seconds")
        self._api_key = api_key.strip()
        self._timeout = timeout
        self._opener = opener or urllib.request.build_opener(_NoRedirect())

    def get(self, endpoint: str, *, max_bytes: int) -> DiscoveryResponse:
        if endpoint not in DISCOVERY_ENDPOINTS:
            raise DiscoveryError("Only LSE discovery metadata endpoints are allowed")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise DiscoveryLimitError("No metadata byte budget remains")
        request = urllib.request.Request(
            VAULT_URL + endpoint,
            headers={"x-api-key": self._api_key, "User-Agent": "pfe-lse-qualification/1"},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                data = response.read(max_bytes + 1)
                status = getattr(response, "status", None)
                headers = getattr(response, "headers", None)
        except urllib.error.HTTPError as exc:
            exc.close()
            raise DiscoveryError(f"LSE metadata HTTP {exc.code}") from None
        except (OSError, ValueError, http.client.HTTPException):
            raise DiscoveryError("LSE metadata request failed") from None
        if len(data) > max_bytes:
            raise DiscoveryLimitError("LSE metadata response exceeds max_bytes")
        _validate_lse_http_body(data, status=status, headers=headers)
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError):
            raise DiscoveryError("Invalid LSE metadata JSON") from None
        return DiscoveryResponse(sanitize_metadata(payload, secrets=(self._api_key,)), len(data))


def sanitize_metadata(value: Any, *, secrets: Sequence[str] = ()) -> Any:
    """Remove credential fields and known secret values before storage or output."""
    if isinstance(value, Mapping):
        return {
            sanitize_metadata(str(key), secrets=secrets): sanitize_metadata(item, secrets=secrets)
            for key, item in value.items()
            if not _credential_field(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
    return value


def _credential_field(key: str) -> bool:
    normalized = "".join(char for char in key.lower() if char.isalnum())
    if normalized in ("pagetoken", "nextpagetoken", "paginationtoken", "continuationtoken"):
        return False
    return normalized in _NORMALIZED_CREDENTIAL_FIELDS or normalized.endswith(
        ("apikey", "password", "passwd", "secret", "token", "authorization")
    )


def _json_bytes(payload) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError):
        raise DiscoveryError("Metadata must be finite, JSON-compatible values") from None


def _audit_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise DiscoveryError("audited_at must be an explicit timezone-aware timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise DiscoveryError("audited_at must be an explicit timezone-aware timestamp") from None
    if result.tzinfo is None:
        raise DiscoveryError("audited_at must be an explicit timezone-aware timestamp")
    return result.astimezone(UTC).isoformat()


def _catalogue(payload, *, max_rows: int) -> tuple[list, dict]:
    """Understand documented list responses, detect but do not guess pagination."""
    flags = {"complete": False, "authoritative": False, "issues": []}
    if isinstance(payload, list):
        rows = payload
        # SDK datasets()/reference() promise the complete lists for these paths.
        flags.update(complete=True, authoritative=True)
    elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
        flags.update(
            complete=payload.get("complete") is True,
            authoritative=payload.get("authoritative") is True,
        )
        if any(
            payload.get(field)
            for field in (
                "next_cursor",
                "next",
                "has_more",
                "next_page_token",
                "continuation_token",
            )
        ):
            flags["complete"] = False
            flags["issues"].append("pagination_not_followed")
        for field in ("total", "total_count"):
            if field in payload and (
                type(payload[field]) is not int or payload[field] != len(rows)
            ):
                flags["complete"] = False
                flags["issues"].append("provider_count_mismatch")
        if payload.get("truncated") or payload.get("limit_reached"):
            flags["complete"] = False
            flags["issues"].append("provider_truncated")
    else:
        raise DiscoveryError("LSE catalogue must be a list or an explicit rows envelope")
    if len(rows) >= max_rows:
        flags["complete"] = False
        flags["issues"].append("row_cap_reached")
    if any(not isinstance(row, Mapping) for row in rows):
        flags["complete"] = False
        flags["issues"].append("invalid_catalogue_row")
    return list(rows[:max_rows]), flags


def collect_lse_discovery(
    transport: DiscoveryTransport,
    *,
    audit_id: str,
    audited_at: str,
    limits: DiscoveryLimits | None = None,
    cached_pages: Mapping[str, dict] | None = None,
    on_page: Callable[[str, dict], None] | None = None,
) -> dict:
    """Fetch metadata only; cache callbacks permit endpoint-by-endpoint resume.

    Byte budget covers the original response bytes, including cached pages;
    requests count new requests in this invocation. Oversized responses fail
    without using partial JSON. Truncated catalogue rows remain explicitly
    incomplete and can never establish absence.
    """
    if not isinstance(audit_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", audit_id
    ):
        raise ValueError("audit_id must be a safe identifier")
    audited_at = _audit_timestamp(audited_at)
    limits = limits or DiscoveryLimits()
    pages, used_bytes, requests = {}, 0, 0
    for endpoint in DISCOVERY_ENDPOINTS:
        page_id = endpoint[1:]
        page = (cached_pages or {}).get(page_id)
        if page is None:
            if requests >= limits.max_requests:
                raise DiscoveryLimitError("LSE discovery exceeds max_requests")
            remaining = limits.max_bytes - used_bytes
            if remaining < 1:
                raise DiscoveryLimitError("LSE discovery exceeds max_bytes")
            response = transport.get(endpoint, max_bytes=remaining)
            requests += 1
            page = {
                "endpoint": endpoint,
                "captured_at": datetime.now(UTC).isoformat(),
                "response_bytes": response.byte_count,
                "response": sanitize_metadata(response.payload),
            }
        if (
            not isinstance(page, Mapping)
            or page.get("endpoint") != endpoint
            or "response" not in page
            or isinstance(page.get("response_bytes"), bool)
            or not isinstance(page.get("response_bytes"), int)
            or page["response_bytes"] < 1
        ):
            raise DiscoveryError("Invalid cached LSE discovery page")
        _audit_timestamp(page.get("captured_at"))
        used_bytes += page["response_bytes"]
        if used_bytes > limits.max_bytes or len(_json_bytes(page["response"])) > limits.max_bytes:
            raise DiscoveryLimitError("LSE discovery exceeds max_bytes")
        if endpoint in ("/catalog", "/reference"):
            rows, coverage = _catalogue(page["response"], max_rows=limits.max_rows)
            original_coverage = page.get("coverage", coverage)
            if (
                not isinstance(original_coverage, Mapping)
                or type(original_coverage.get("complete")) is not bool
                or type(original_coverage.get("authoritative")) is not bool
                or not isinstance(original_coverage.get("issues"), list)
                or any(not isinstance(issue, str) for issue in original_coverage["issues"])
            ):
                raise DiscoveryError("Invalid cached LSE catalogue coverage")
            page = {**page, "response": rows, "coverage": dict(original_coverage)}
            # Retrying must not turn an originally truncated page into complete.
            if coverage["complete"] is False:
                page["coverage"] = {**page["coverage"], "complete": False}
        elif not isinstance(page["response"], Mapping):
            raise DiscoveryError("LSE meta and usage must be JSON objects")
        page = sanitize_metadata(page)
        if page_id not in (cached_pages or {}) and on_page is not None:
            on_page(page_id, page)
        pages[page_id] = page
    return {
        "schema_version": "1",
        "provider": "LSE",
        "audit_id": audit_id,
        "audited_at": audited_at,
        "mode": "metadata",
        "catalog": pages["catalog"]["response"],
        "catalog_complete": pages["catalog"]["coverage"]["complete"],
        "catalog_authoritative": pages["catalog"]["coverage"]["authoritative"],
        "catalog_issues": pages["catalog"]["coverage"]["issues"],
        "reference": pages["reference"]["response"],
        "reference_complete": pages["reference"]["coverage"]["complete"],
        "reference_authoritative": pages["reference"]["coverage"]["authoritative"],
        "reference_issues": pages["reference"]["coverage"]["issues"],
        "meta": pages["meta"]["response"],
        "usage": pages["usage"]["response"],
        "limits": asdict(limits),
        "response_bytes": used_bytes,
        "requests_this_run": requests,
        "endpoint_captured_at": {name: page["captured_at"] for name, page in pages.items()},
        "documentation": {"catalogue_contract": SDK_VAULT_URL},
        "entitlement": "unknown",
    }


@dataclass(frozen=True)
class SymbolRequest:
    """An exact provider code; dataset is required when that code is ambiguous."""

    symbol: str
    dataset: str | None = None

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol or len(self.symbol) > 128:
            raise ValueError("symbol must be a nonempty exact provider code")
        if any(ord(char) < 32 for char in self.symbol):
            raise ValueError("symbol must not contain control characters")
        if self.dataset is not None and (
            not isinstance(self.dataset, str)
            or not self.dataset
            or len(self.dataset) > 128
            or any(ord(char) < 32 for char in self.dataset)
        ):
            raise ValueError("dataset must be a nonempty exact provider code")


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    label: str
    country_or_region: str
    source_frequency: str
    models: tuple[str, ...]
    priority: str
    sdk_methods: tuple[str, ...] = ()
    reference_datasets: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    kind: str = "data"
    implementation: str = "not_implemented"
    scope: str = "unqualified"


# The 34 rows in the current roadmap, preserving priorities and model families.
# Generic dataset classes are not used to certify particular instruments/countries.
REQUIREMENTS = (
    Requirement(
        "us_ohlcv",
        "OHLCV actions US",
        "USA",
        "Historique de marché / daily",
        ("GRU", "TCN", "GNN"),
        "P0",
        ("candles", "history"),
        implementation="partial",
        scope="us_stocks",
    ),
    Requirement(
        "corporate_actions",
        "Ajustements splits et dividendes",
        "USA",
        "Historique des corporate actions",
        ("Prétraitement",),
        "P0",
        ("splits", "dividends"),
        ("stock_splits", "dividends"),
        implementation="partial",
        scope="references",
    ),
    Requirement(
        "security_master",
        "Tickers, identifiants, historique de cotation",
        "USA",
        "SEC, fournisseur de marché",
        ("Prétraitement",),
        "P0",
        ("company_profiles",),
        ("company_profiles",),
        implementation="partial",
    ),
    Requirement(
        "technical",
        "RSI, ATR, MACD, moyennes mobiles, momentum",
        "USA",
        "Calcul local / daily",
        ("GRU", "TCN"),
        "P0",
        kind="local",
        implementation="implemented",
    ),
    Requirement(
        "price_statistics",
        "Volatilité, volume relatif, drawdown",
        "USA",
        "Calcul local / daily",
        ("GRU", "TCN", "GNN"),
        "P0",
        kind="local",
        implementation="implemented",
    ),
    Requirement(
        "benchmark_returns",
        "Rendements SPY, QQQ, IWM",
        "USA",
        "Marchés / daily",
        ("Macro", "GNN"),
        "P1",
        ("candles", "history"),
        symbols=("SPY", "QQQ", "IWM"),
        implementation="partial",
        scope="symbols",
    ),
    Requirement(
        "sector_etfs",
        "ETF sectoriels, SOXX, XLF, XLE, etc.",
        "USA",
        "Marchés / daily",
        ("Macro", "GNN"),
        "P1",
        ("candles", "history"),
        symbols=("SOXX", "XLF", "XLE"),
        implementation="partial",
        scope="symbols",
    ),
    Requirement(
        "volatility_indices",
        "VIX et indices de volatilité",
        "USA",
        "Cboe / daily",
        ("Macro",),
        "P1",
        ("candles", "history"),
        symbols=("VIX", "VXN", "RVX"),
        implementation="partial",
        scope="symbols",
    ),
    Requirement(
        "yield_curve",
        "Taux 2 ans, 10 ans, courbe des taux",
        "USA",
        "FRED / daily",
        ("Macro",),
        "P1",
        ("series", "bond_yields"),
        ("bond_yields",),
        ("US2Y", "US10Y"),
        implementation="partial",
        scope="symbols",
    ),
    Requirement(
        "credit_spreads",
        "Spreads de crédit",
        "USA",
        "FRED / daily",
        ("Macro",),
        "P1",
        ("economics", "series"),
        implementation="partial",
    ),
    Requirement(
        "us_macro",
        "Fed Funds, CPI, PPI, chômage",
        "USA",
        "ALFRED / publications",
        ("Macro",),
        "P1",
        ("economics", "series"),
        symbols=("fdtr", "cpi_yoy"),
        implementation="partial",
    ),
    Requirement(
        "fx",
        "Dollar, USD/JPY, USD/CNY, USD/TWD, USD/KRW",
        "USA, Japon, Chine, Taïwan, Corée",
        "Marchés FX / daily",
        ("Macro", "GNN"),
        "P1",
        ("candles", "history"),
        symbols=("USD/JPY", "USD/CNY", "USD/TWD", "USD/KRW"),
        implementation="partial",
    ),
    Requirement(
        "commodities",
        "Pétrole, gaz, cuivre, aluminium, or",
        "Monde",
        "Marchés, EIA, Banque mondiale",
        ("Macro", "GNN"),
        "P1",
        ("candles", "history"),
        implementation="partial",
    ),
    Requirement(
        "financial_actuals",
        "CA, EPS, marges, dette, cash-flow réels",
        "USA",
        "SEC XBRL / trimestriel",
        ("Fondamentaux", "GNN"),
        "P1",
        ("financial_reports",),
        ("financial_reports",),
        implementation="partial",
        scope="references",
    ),
    Requirement(
        "guidance",
        "Guidance CA, EPS, marge, CapEx",
        "USA",
        "8-K, earnings calls / irrégulier",
        ("LLM → Fondamentaux",),
        "P1",
    ),
    Requirement(
        "relationships",
        "Fournisseurs et clients déclarés",
        "Monde",
        "Rapports annuels / irrégulier",
        ("LLM → GNN",),
        "P1",
    ),
    Requirement(
        "geography",
        "Exposition géographique des revenus",
        "Monde",
        "10-K, rapports / annuel",
        ("LLM → GNN",),
        "P1",
    ),
    Requirement(
        "input_exposures",
        "Exposition matières premières et production",
        "Monde",
        "Rapports / annuel",
        ("LLM → GNN",),
        "P1",
    ),
    Requirement(
        "competition",
        "Relations sectorielles et concurrentielles",
        "Monde",
        "SEC, référentiels, extraction",
        ("GNN",),
        "P1",
        ("company_profiles",),
        ("company_profiles",),
    ),
    Requirement(
        "foreign_prices",
        "Prix et rendements des principaux fournisseurs étrangers",
        "Taïwan, Corée, Japon, Europe",
        "Bourses locales / daily",
        ("GNN", "Macro"),
        "P2",
        ("candles", "history"),
    ),
    Requirement(
        "tsmc_monthly",
        "CA mensuel des grandes fonderies, notamment TSMC",
        "Taïwan",
        "TSMC IR / mensuel",
        ("GNN", "Fondamentaux"),
        "P2",
    ),
    Requirement(
        "foreign_actuals",
        "Résultats des entreprises étrangères stratégiques",
        "Taïwan, Corée, Europe",
        "MOPS, OpenDART, rapports",
        ("LLM", "GNN"),
        "P2",
        ("financial_reports",),
        ("financial_reports",),
    ),
    Requirement(
        "china_macro",
        "PMI, production industrielle, activité manufacturière",
        "Chine",
        "NBS / mensuel",
        ("Macro", "GNN"),
        "P2",
        ("economics", "series"),
    ),
    Requirement(
        "eu_macro",
        "Inflation, industrie, commerce extérieur",
        "UE",
        "Eurostat / publications",
        ("Macro",),
        "P2",
        ("economics", "series"),
    ),
    Requirement(
        "japan_macro",
        "Tankan, taux, inflation, production",
        "Japon",
        "BoJ / publications",
        ("Macro",),
        "P2",
        ("economics", "series"),
    ),
    Requirement(
        "mineral_production",
        "Production et dépendances minières",
        "Monde, par pays producteur",
        "USGS / annuel",
        ("GNN structurel",),
        "P2",
        kind="external",
    ),
    Requirement(
        "global_events",
        "Événements géopolitiques et économiques",
        "Monde",
        "GDELT / événements",
        ("LLM", "Event Encoder"),
        "P2",
        kind="external",
    ),
    Requirement(
        "company_news",
        "Nouvelles propres aux entreprises",
        "Monde",
        "SEC 8-K, communiqués, fournisseur news",
        ("LLM", "Event Encoder"),
        "P2",
    ),
    Requirement(
        "text_vectors",
        "Embeddings textuels et sentiment",
        "Monde",
        "Calcul local / à publication",
        ("Event Encoder",),
        "P2",
        kind="local",
    ),
    Requirement(
        "analyst_consensus",
        "Consensus historiques de CA et EPS",
        "Monde",
        "FactSet, I/B/E/S / snapshots",
        ("Fondamentaux", "Event Encoder"),
        "P3",
    ),
    Requirement(
        "analyst_revisions",
        "Révisions d'analystes, dispersion, surprises",
        "Monde",
        "Calcul sur historiques de consensus",
        ("Fondamentaux",),
        "P3",
        kind="local",
    ),
    Requirement(
        "industrial_graph",
        "Relations fournisseur-client industrielles complètes",
        "Monde",
        "FactSet Revere / historique",
        ("GNN",),
        "P3",
    ),
    Requirement(
        "cot",
        "Positionnement sur futures",
        "USA, marchés mondiaux",
        "CFTC COT / hebdomadaire",
        ("Macro",),
        "P3",
        ("cot",),
        ("cot",),
        scope="references",
    ),
    Requirement(
        "options",
        "Options, skew et volatilité implicite",
        "USA",
        "Cboe, fournisseurs spécialisés",
        ("Macro", "GRU"),
        "P3",
        ("options", "options_flow", "option_candles"),
        implementation="partial",
        scope="options",
    ),
)


def _references(rows: list) -> tuple[set[str], int, int]:
    names, invalid, duplicates = set(), 0, 0
    for row in rows:
        name = row.get("dataset", row.get("name")) if isinstance(row, Mapping) else None
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            invalid += 1
        elif name in names:
            duplicates += 1
        else:
            names.add(name)
    return names, invalid, duplicates


def _text_status(value) -> str:
    if value is None:
        return "unqualified"
    return "provider_reported" if isinstance(value, str) and value.strip() else "malformed"


def _history_time_status(value) -> str:
    if value is None:
        return "unqualified"
    if not isinstance(value, str):
        return "malformed"
    try:
        if len(value) == 10:
            date.fromisoformat(value)
            return "provider_reported_day"
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "malformed"
    return "provider_reported" if timestamp.tzinfo else "unqualified_timezone"


def _ticks_status(value) -> str:
    if value is None:
        return "unqualified"
    return "provider_reported" if type(value) is int and value >= 0 else "malformed"


def _currency_status(value) -> str:
    if value is None:
        return "unqualified"
    if isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value):
        return "provider_reported"
    return "malformed"


def _timezone_status(value) -> str:
    if value is None:
        return "unqualified"
    if not isinstance(value, str) or not value:
        return "malformed"
    try:
        ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError):
        return "malformed"
    return "provider_reported"


def _issues(snapshot, field) -> list[str]:
    value = snapshot.get(field, [])
    if not isinstance(value, list) or any(not isinstance(issue, str) for issue in value):
        raise DiscoveryError("Catalogue issues must be arrays of strings")
    return list(value)


def _symbol_result(
    request: SymbolRequest, index: tuple[dict, dict], *, can_assert_absence: bool
) -> dict:
    by_symbol, by_pair = index
    matches = (
        by_symbol.get(request.symbol, [])
        if request.dataset is None
        else by_pair.get((request.symbol, request.dataset), [])
    )
    if len(matches) == 1 and matches[0]["status"] == "available":
        status, reason = "available", "exact_catalogue_match"
    elif matches:
        status, reason = "unqualified", "ambiguous_catalogue_match"
    elif can_assert_absence:
        status, reason = "absent", "exact_code_not_listed_in_complete_dated_catalogue"
    else:
        status, reason = "unqualified", "catalogue_does_not_establish_absence"
    return {
        **asdict(request),
        "status": status,
        "reason": reason,
        "matches": [{"symbol": row["symbol"], "dataset": row["dataset"]} for row in matches],
        "entitlement": "unknown",
    }


def qualify_lse_snapshot(
    snapshot: Mapping,
    *,
    symbols: Sequence[str | SymbolRequest] = (),
    limits: DiscoveryLimits | None = None,
) -> dict:
    """Produce metadata matrix and every requirement without network or LSE imports.

    Absent exact codes do not mean a company, economic series or user requirement
    is unavailable under an unknown alias. Requirements without confirmed exact
    mappings stay unqualified even when a broad asset class exists.
    """
    if not isinstance(snapshot, Mapping) or snapshot.get("provider", "LSE") != "LSE":
        raise DiscoveryError("Expected an LSE discovery snapshot")
    limits = limits or DiscoveryLimits()
    if len(_json_bytes(snapshot)) > limits.max_bytes:
        raise DiscoveryLimitError("LSE snapshot exceeds max_bytes")
    snapshot = sanitize_metadata(snapshot)
    audited_at = _audit_timestamp(snapshot.get("audited_at"))
    catalog, reference = snapshot.get("catalog", []), snapshot.get("reference", [])
    if not isinstance(catalog, list) or not isinstance(reference, list):
        raise DiscoveryError("catalog and reference must be arrays")
    catalog_issues = _issues(snapshot, "catalog_issues")
    reference_issues = _issues(snapshot, "reference_issues")
    capped = len(catalog) >= limits.max_rows
    reference_capped = len(reference) >= limits.max_rows
    catalog = catalog[: limits.max_rows]
    reference = reference[: limits.max_rows]
    matrix, invalid_rows, identities = [], 0, {}
    for row in catalog:
        if not isinstance(row, Mapping) or not all(
            isinstance(row.get(name), str) and row[name] for name in ("symbol", "dataset")
        ):
            invalid_rows += 1
            continue
        identity = row["symbol"], row["dataset"]
        identities[identity] = identities.get(identity, 0) + 1
        matrix.append(
            {
                "symbol": row["symbol"],
                "dataset": row["dataset"],
                "name": row.get("name"),
                "country": row.get("country"),
                "first": row.get("first_tick", row.get("first")),
                "first_status": _history_time_status(row.get("first_tick", row.get("first"))),
                "last": row.get("last_tick", row.get("last")),
                "last_status": _history_time_status(row.get("last_tick", row.get("last"))),
                "ticks": row.get("ticks"),
                "ticks_status": _ticks_status(row.get("ticks")),
                "unit": row.get("unit"),
                "currency": row.get("currency"),
                "currency_status": _currency_status(row.get("currency")),
                "timezone": row.get("timezone"),
                "timezone_status": _timezone_status(row.get("timezone")),
                "status": "available",
                "entitlement": "unknown",
                "unit_status": _text_status(row.get("unit")),
                "adjustment": {
                    "candles": "split_adjusted"
                    if row["dataset"] in ("stocks", "etf")
                    else "unqualified",
                    "ticks": "unqualified",
                    "dividends": "unqualified",
                },
                "adjustment_evidence": SDK_CHANGELOG_URL
                if row["dataset"] in ("stocks", "etf")
                else None,
                "point_in_time": "unqualified",
                "publication_timing": "unqualified",
                "coverage": "listed_metadata_only",
                "evidence": "catalog",
            }
        )
    for row in matrix:
        if identities[row["symbol"], row["dataset"]] > 1:
            row.update(status="unqualified", identity_issue="duplicate_symbol_dataset")
    duplicate_rows = sum(count - 1 for count in identities.values())
    if duplicate_rows:
        catalog_issues.append("ambiguous_symbol_dataset")
    catalog_complete = (
        snapshot.get("catalog_complete") is True
        and invalid_rows == 0
        and not capped
        and not catalog_issues
    )
    catalog_authoritative = snapshot.get("catalog_authoritative") is True
    can_assert_absence = catalog_complete and catalog_authoritative
    requests = [SymbolRequest(item) if isinstance(item, str) else item for item in symbols]
    if any(not isinstance(request, SymbolRequest) for request in requests):
        raise ValueError("symbols must contain exact strings or SymbolRequest values")
    if len(requests) > limits.max_rows:
        raise DiscoveryLimitError("Requested symbols exceed max_rows")
    if len(set(requests)) != len(requests):
        raise ValueError("Requested exact symbol/dataset pairs must be unique")
    by_symbol, by_pair = {}, {}
    for row in matrix:
        by_symbol.setdefault(row["symbol"], []).append(row)
        by_pair.setdefault((row["symbol"], row["dataset"]), []).append(row)
    index = by_symbol, by_pair
    symbol_results = [
        _symbol_result(request, index, can_assert_absence=can_assert_absence)
        for request in requests
    ]
    reference_names, invalid_references, duplicate_references = _references(reference)
    if invalid_references:
        reference_issues.append("invalid_reference_entry")
    if duplicate_references:
        reference_issues.append("duplicate_reference_entry")
    if reference_capped:
        reference_issues.append("row_cap_reached")
    reference_coverage = {
        "complete": snapshot.get("reference_complete") is True and not reference_issues,
        "authoritative": snapshot.get("reference_authoritative") is True,
        "invalid_rows": invalid_references,
        "duplicate_rows": duplicate_references,
        "issues": reference_issues,
        "row_cap_reached": reference_capped,
        "rows": len(reference),
    }
    requirements = []
    for requirement in REQUIREMENTS:
        reference_matches = sorted(reference_names.intersection(requirement.reference_datasets))
        candidates = [
            _symbol_result(SymbolRequest(symbol), index, can_assert_absence=can_assert_absence)
            for symbol in requirement.symbols
        ]
        status: QualificationStatus = "unqualified"
        if requirement.kind == "local":
            status = "local"
        elif requirement.kind == "external":
            status = "not-applicable"
        elif requirement.scope == "references" and len(reference_matches) == len(
            requirement.reference_datasets
        ):
            status = "available"
        elif (
            requirement.scope == "symbols"
            and candidates
            and all(item["status"] == "available" for item in candidates)
        ):
            status = "available"
        elif requirement.scope == "us_stocks" and any(
            row["dataset"] == "stocks"
            and row["country"] in ("US", "USA", "United States")
            and row["status"] == "available"
            for row in matrix
        ):
            status = "available"
        elif requirement.scope == "options" and any(
            row["dataset"] == "options"
            and row["country"] in ("US", "USA", "United States")
            and row["status"] == "available"
            for row in matrix
        ):
            status = "available"
        requirements.append(
            {
                **json.loads(_json_bytes(asdict(requirement))),
                "status": status,
                "reference_matches": reference_matches,
                "symbol_candidates": candidates,
                "coverage": "listed_metadata_only" if status == "available" else "unqualified",
                "entitlement": "unknown",
                "point_in_time": "unqualified",
                "sdk_evidence": SDK_CLIENT_URL if requirement.sdk_methods else None,
                "notes": [
                    "Availability is metadata evidence only; no market-data request was made.",
                    "Exact examples do not certify an exhaustive universe or unknown aliases.",
                    "Historical publication, units, adjustments and account access "
                    "need qualification.",
                ],
            }
        )
    return {
        "schema_version": "1",
        "provider": "LSE",
        "audit_id": snapshot.get("audit_id"),
        "audited_at": audited_at,
        "mode": snapshot.get("mode", "offline"),
        "catalogue_coverage": {
            "complete": catalog_complete,
            "authoritative": catalog_authoritative,
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "issues": catalog_issues,
            "row_cap_reached": capped,
            "absent_scope": "exact_provider_code_only",
            "rows": len(catalog),
        },
        "reference_coverage": reference_coverage,
        "matrix": matrix,
        "requested_symbols": symbol_results,
        "requirements": requirements,
        "implementation_baseline": dict(IMPLEMENTATION_BASELINE),
        "usage": snapshot.get("usage", {}),
        "entitlement": "unknown",
        "limitations": [
            "Discovery and quota metadata do not establish account entitlements.",
            "Current catalogue membership does not prove historical listings or delistings.",
            "Unknown aliases, units, publication timestamps and PIT remain unqualified.",
        ],
    }
