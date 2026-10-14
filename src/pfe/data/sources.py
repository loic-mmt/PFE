"""Versioned source declarations, separate from account entitlements and data licenses."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

ACCESS_TYPES = frozenset(("free", "licensed", "local", "unknown"))
AVAILABILITY_TYPES = frozenset(("receipt", "historical_pit", "unknown"))
USE_TYPES = frozenset(("research", "trading", "model_training", "commercial"))


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _public_url(value, name):
    _identifier(value, name)
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be an HTTP(S) URL without credentials or query parameters")


def _verified_time(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("verified_at must be a timezone-aware timestamp") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("verified_at must be a timezone-aware timestamp")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class SourceRecord:
    """Audited declaration; unknown rights remain unknown, never become an entitlement.

    ``revision_id`` identifies an immutable declaration, not a value revision.
    ``verified_at`` is the declaration's audit time, not historical data availability.
    ``allowed_uses`` records explicit provider permissions only; an empty tuple means
    permissions were not established here. Account/API access must be checked separately.
    """

    source_id: str
    revision_id: str
    name: str
    url: str | None
    access: str = "unknown"
    license_url: str | None = None
    license_notes: str = ""
    allowed_uses: tuple[str, ...] = ()
    redistribution_allowed: bool | None = None
    availability: str = "unknown"
    verified_at: datetime | None = None
    schema_version: str = "1"

    def __post_init__(self):
        for name in ("source_id", "revision_id", "name", "schema_version"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.access, str) or self.access not in ACCESS_TYPES:
            raise ValueError(f"access must be one of {sorted(ACCESS_TYPES)}")
        if self.url is not None or self.access != "local":
            _public_url(self.url, "url")
        if self.license_url is not None:
            _public_url(self.license_url, "license_url")
        if not isinstance(self.availability, str) or self.availability not in AVAILABILITY_TYPES:
            raise ValueError(f"availability must be one of {sorted(AVAILABILITY_TYPES)}")
        if not isinstance(self.license_notes, str):
            raise ValueError("license_notes must be a string")
        if not isinstance(self.allowed_uses, (tuple, list)):
            raise ValueError("allowed_uses must be an array of explicit permissions")
        if any(not isinstance(use, str) or use not in USE_TYPES for use in self.allowed_uses):
            raise ValueError(f"allowed_uses must contain only {sorted(USE_TYPES)}")
        if len(set(self.allowed_uses)) != len(self.allowed_uses):
            raise ValueError("allowed_uses must be unique")
        if (
            self.redistribution_allowed is not None
            and type(self.redistribution_allowed) is not bool
        ):
            raise ValueError("redistribution_allowed must be bool or None")
        object.__setattr__(self, "allowed_uses", tuple(sorted(self.allowed_uses)))
        object.__setattr__(self, "verified_at", _verified_time(self.verified_at))

    def to_dict(self):
        """JSON-compatible metadata suitable for a raw collection manifest."""
        record = asdict(self)
        record["allowed_uses"] = list(self.allowed_uses)
        record["verified_at"] = self.verified_at.isoformat() if self.verified_at else None
        return record

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, dict):
            raise ValueError("Source declaration must be an object")
        try:
            return cls(**record)
        except TypeError as error:
            raise ValueError("Invalid source declaration fields") from error


class SourceRegistry:
    """Append-only declarations keyed by (source_id, revision_id).

    Re-registering identical content is idempotent. Altering an existing revision
    raises; a corrected declaration needs a new revision ID and audit timestamp.
    """

    def __init__(self, records=()):
        self._records = {}
        for record in records:
            self.register(record)

    def register(self, record):
        if not isinstance(record, SourceRecord):
            raise ValueError("Expected a SourceRecord")
        key = (record.source_id, record.revision_id)
        prior = self._records.get(key)
        if prior is not None and prior != record:
            raise ValueError("Source revision already exists with different content")
        self._records[key] = record
        return record

    def get(self, source_id, revision_id=None):
        _identifier(source_id, "source_id")
        if revision_id is not None:
            return self._records[(source_id, _identifier(revision_id, "revision_id"))]
        records = [record for (source, _), record in self._records.items() if source == source_id]
        if not records:
            raise KeyError(source_id)
        if len(records) == 1:
            return records[0]
        if any(record.verified_at is None for record in records):
            raise ValueError("Specify revision_id when source audit order is unknown")
        latest = max(record.verified_at for record in records)
        candidates = [record for record in records if record.verified_at == latest]
        if len(candidates) != 1:
            raise ValueError("Specify revision_id for simultaneous source declarations")
        return candidates[0]

    def to_dict(self):
        return {
            "schema_version": "1",
            "sources": [self._records[key].to_dict() for key in sorted(self._records)],
        }

    @classmethod
    def from_dict(cls, payload):
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "sources"}
            or payload["schema_version"] != "1"
            or not isinstance(payload["sources"], list)
        ):
            raise ValueError("Invalid source registry schema")
        return cls(SourceRecord.from_dict(record) for record in payload["sources"])

    @classmethod
    def load(cls, path, *, max_bytes=1048576):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        with Path(path).open("rb") as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("Source registry exceeds max_bytes")
        return cls.from_dict(json.loads(raw))


def default_source_registry():
    """Declarations from dated audits; no live calls or credentials.

    Only LSE permissions explicitly documented by its SDK are recorded as allowed.
    SEC/FRED data and issuer/third-party content require per-dataset qualification.
    New audits must register a new revision rather than editing these declarations.
    """
    audited = datetime(2026, 10, 11, tzinfo=UTC)
    revision = "audit-2026-10-11"
    registry = SourceRegistry(
        (
            SourceRecord(
                "lse",
                revision,
                "London Strategic Edge",
                "https://londonstrategicedge.com/data/",
                access="free",
                license_url="https://londonstrategicedge.com/terms/",
                license_notes=(
                    "SDK MIT license is not the data license. Provider SDK documents own "
                    "research, trading and model training including commercial use; no "
                    "redistribution or resale. Confirm current terms and account quotas."
                ),
                allowed_uses=("research", "trading", "model_training", "commercial"),
                redistribution_allowed=False,
                availability="receipt",
                verified_at=audited,
            ),
            SourceRecord(
                "sec",
                revision,
                "SEC EDGAR",
                "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
                access="free",
                license_notes=(
                    "Public APIs; identify automated clients and respect rate limits. "
                    "Issuer filings may contain third-party content; no blanket rights inferred."
                ),
                verified_at=audited,
            ),
            SourceRecord(
                "fred",
                revision,
                "FRED / ALFRED",
                "https://fred.stlouisfed.org/",
                access="free",
                license_notes=(
                    "Rights and depth depend on each series, including licensed ICE data. "
                    "Vintage dates do not establish intraday delivery timestamps."
                ),
                verified_at=audited,
            ),
            SourceRecord(
                "local",
                revision,
                "Local calculation / synthetic fixtures",
                None,
                access="local",
                license_notes=(
                    "Derived data retains upstream restrictions; synthetic tests separately."
                ),
                verified_at=audited,
            ),
            SourceRecord(
                "eia",
                "audit-2026-10-12",
                "US Energy Information Administration",
                "https://www.eia.gov/opendata/",
                access="free",
                license_url="https://www.eia.gov/about/copyrights_reuse.php",
                license_notes=(
                    "US government material is generally public domain; third-party content "
                    "can carry separate rights. API key, route units and coverage must be "
                    "qualified per dataset. Period labels are not publication timestamps."
                ),
                availability="receipt",
                verified_at=datetime(2026, 10, 12, tzinfo=UTC),
            ),
            SourceRecord(
                "cboe",
                "audit-2026-10-12",
                "Cboe VIX historical files",
                "https://www.cboe.com/tradable_products/vix/vix_historical_data",
                access="free",
                license_url="https://www.cboe.com/terms",
                license_notes=(
                    "Public file access does not grant general reuse rights. Website terms "
                    "limit copying to personal noncommercial use; other uses, including "
                    "model training or commercial distribution, require qualification. "
                    "Daily labels do not prove delivery or publication timestamps."
                ),
                availability="receipt",
                verified_at=datetime(2026, 10, 12, tzinfo=UTC),
            ),
            SourceRecord(
                "world_bank",
                "audit-2026-10-12",
                "World Bank commodity Pink Sheet",
                "https://www.worldbank.org/en/research/commodity-markets",
                access="free",
                license_url="https://data.worldbank.org/summary-terms-of-use",
                license_notes=(
                    "Data terms default to CC BY 4.0 subject to dataset metadata and "
                    "third-party exceptions. Qualify the selected workbook and attribution. "
                    "Monthly prices are averages; economic month is not availability."
                ),
                availability="receipt",
                verified_at=datetime(2026, 10, 12, tzinfo=UTC),
            ),
        )
    )

    from pfe.data.international import INTERNATIONAL_SOURCES, international_source_record

    for source_id in INTERNATIONAL_SOURCES:
        registry.register(international_source_record(source_id))
    from pfe.data.conflicts import ucdp_source_record
    from pfe.data.cot import cot_source_record

    registry.register(cot_source_record())
    registry.register(ucdp_source_record())
    return registry
