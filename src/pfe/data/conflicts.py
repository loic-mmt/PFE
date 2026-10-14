"""Bounded versioned UCDP GED/annual captures, never dated at the event's occurrence."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from numbers import Real

import pandas as pd

from pfe.data.captures import CaptureLimits, capture_observations
from pfe.data.contracts import _instant, normalize_observations
from pfe.data.eia import _observations
from pfe.data.http import BoundedHTTPTransport
from pfe.data.international_sources import _json, _point
from pfe.data.sessions import _session_date
from pfe.data.sources import SourceRecord


def _integer(value, label, *, minimum=0):
    integral_number = isinstance(value, Real) and float(value).is_integer()
    if isinstance(value, bool) or not (integral_number or re.fullmatch(r"\d+", str(value))):
        raise ValueError(f"{label} requires an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} is below its minimum")
    return result


@dataclass(frozen=True)
class UCDPQuery:
    resource: str
    version: str
    start: str
    end: str
    countries: tuple[int, ...] = ()
    page: int = 1

    def __post_init__(self):
        if self.resource not in ("gedevents", "ucdpprioconflict"):
            raise ValueError("UCDP resource must be gedevents or ucdpprioconflict")
        if not isinstance(self.version, str) or not re.fullmatch(
            r"\d{2}\.\d+(?:\.\d+){0,2}", self.version
        ):
            raise ValueError("UCDP requires an explicit dataset version")
        if self.start != _session_date(self.start) or self.end != _session_date(self.end):
            raise ValueError("UCDP bounds require ISO dates")
        if self.start > self.end:
            raise ValueError("UCDP bounds must be ordered")
        countries = tuple(self.countries)
        if len(set(countries)) != len(countries) or any(
            type(v) is not int or v < 1 for v in countries
        ):
            raise ValueError("UCDP countries require distinct positive GW codes")
        object.__setattr__(self, "countries", countries)
        if type(self.page) is not int or self.page < 1:
            raise ValueError("UCDP page must be a positive integer")
        if self.resource == "ucdpprioconflict" and (
            not self.start.endswith("-01-01")
            or not self.end.endswith("-12-31")
            or int(self.end[:4]) - int(self.start[:4]) > 100
        ):
            raise ValueError("Annual UCDP queries require complete bounded calendar years")


def normalize_conflict_observations(frame):
    """Canonical JSON observations, with event identity stable across date corrections.

    Event values require event_id/conflict_id, GW country_id, region, type_of_violence,
    date_start/date_end and best/low/high deaths. Annual values require conflict_id,
    year and intensity_level (1 minor, 2 war). active=False withdraws a known
    record only from that vintage onward. Multiple providers must be selected
    explicitly before computing aggregates.
    """
    out = normalize_observations(frame)
    for row in out.itertuples():
        value = row.value
        if not isinstance(value, dict) or type(value.get("active", True)) is not bool:
            raise ValueError("Conflict observations require structured values and boolean active")
        if row.unit != "conflict_record" or row.revision_id is None:
            raise ValueError("Conflict records require conflict_record units and revision IDs")
        kind = value.get("record_type")
        value["conflict_id"] = _integer(value.get("conflict_id"), "conflict_id", minimum=1)
        if kind == "event":
            identifier = value["event_id"] = _integer(value.get("event_id"), "event_id", minimum=1)
            value["country_id"] = _integer(value.get("country_id"), "country_id", minimum=1)
            if value.get("type_of_violence") not in (1, 2, 3) or isinstance(
                value.get("type_of_violence"), bool
            ):
                raise ValueError("Conflict type_of_violence must be 1, 2 or 3")
            start, end = (_session_date(value.get(k)) for k in ("date_start", "date_end"))
            if start > end or row.observation_date != end:
                raise ValueError("Event dates must be ordered and match observation_date")
            if not isinstance(value.get("region"), str) or not value["region"].strip():
                raise ValueError("Conflict region requires an explicit label")
            for field in ("best", "low", "high"):
                if field not in value:
                    raise ValueError("Conflict deaths require explicit best/low/high fields")
                if value[field] is not None:
                    value[field] = _integer(value[field], field)
            low, best, high = (value[k] for k in ("low", "best", "high"))
            if (
                (low is not None and high is not None and low > high)
                or (best is not None and low is not None and best < low)
                or (best is not None and high is not None and best > high)
            ):
                raise ValueError("Conflict death bounds must contain best estimate")
            expected = f"ucdp.event.{identifier}"
        elif kind == "annual":
            year = value["year"] = _integer(value.get("year"), "year", minimum=1946)
            if year > 2200 or row.observation_date != f"{year}-12-31":
                raise ValueError("Annual conflict observation_date must be the economic year end")
            value["intensity_level"] = _integer(
                value.get("intensity_level"), "intensity_level", minimum=1
            )
            if value["intensity_level"] not in (1, 2):
                raise ValueError("Annual intensity_level must be 1 (minor) or 2 (war)")
            expected = f"ucdp.annual.{year}.{value['conflict_id']}"
        else:
            raise ValueError("Conflict record_type must be event or annual")
        if row.metric != expected:
            raise ValueError("Conflict metric must identify the exact event or conflict-year")
    # Economic dates may be corrected: availability uniqueness is by record ID.
    if out.duplicated(["source_id", "metric", "available_at"]).any():
        raise ValueError("Conflicting simultaneous conflict vintage")
    if out.duplicated(["source_id", "metric", "revision_id"]).any():
        raise ValueError("Duplicate immutable conflict revision")
    return out.sort_values("available_at", kind="stable", ignore_index=True)


def parse_ucdp(body, *, query, received_at, max_rows=1000, published_at=None):
    query = UCDPQuery(**query) if isinstance(query, dict) else query
    payload = _json(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("Result"), list):
        raise ValueError("UCDP response requires a Result array")
    rows = payload["Result"]
    if len(rows) > max_rows:
        raise ValueError("UCDP response exceeds row budget")
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("UCDP rows must be objects")
        conflict = _integer(
            row.get("conflict_new_id") if query.resource == "gedevents" else row.get("conflict_id"),
            "conflict_id",
            minimum=1,
        )
        if query.resource == "gedevents":
            event = _integer(row.get("id"), "event_id", minimum=1)
            country = _integer(row.get("country_id"), "country_id", minimum=1)
            day = _session_date(row.get("date_end"))
            if (
                not query.start <= day <= query.end
                or query.countries
                and country not in query.countries
            ):
                raise ValueError("UCDP event outside selected bounds/countries")
            value = dict(
                record_type="event",
                event_id=event,
                conflict_id=conflict,
                country_id=country,
                region=row.get("region"),
                type_of_violence=_integer(
                    row.get("type_of_violence"), "type_of_violence", minimum=1
                ),
                date_start=_session_date(row.get("date_start")),
                date_end=day,
                **{
                    k: _integer(row[k], k) if row.get(k) is not None else None
                    for k in ("best", "low", "high")
                },
            )
            metric = f"ucdp.event.{event}"
        else:
            year = _integer(row.get("year"), "year", minimum=1946)
            day = f"{year}-12-31"
            if not query.start <= day <= query.end:
                raise ValueError("UCDP annual record outside selected years")
            if query.countries:
                locations = {
                    int(v.strip())
                    for v in str(row.get("gwno_loc", "")).split(",")
                    if v.strip().isdigit()
                }
                if not locations.intersection(query.countries):
                    raise ValueError("UCDP annual record outside selected countries")
            value = dict(
                record_type="annual",
                conflict_id=conflict,
                year=year,
                intensity_level=_integer(row.get("intensity_level"), "intensity_level", minimum=1),
            )
            metric = f"ucdp.annual.{year}.{conflict}"
        record = _point(
            body,
            received_at=received_at,
            source="ucdp",
            entity_id="factor:conflicts",
            metric=metric,
            unit="conflict_record",
            period=day if query.resource == "gedevents" else str(year),
            frequency="daily" if query.resource == "gedevents" else "annual",
            value=None,
            country=str(row.get("country", "global")),
            published_at=published_at,
            evidence=json.dumps(row, sort_keys=True),
        )
        record.update(
            value=value,
            observation_date=day,
            revision_id=query.version
            + ":"
            + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
            provider_version=query.version,
        )
        records.append(record)
    return normalize_conflict_observations(pd.DataFrame(records)) if records else _observations([])


def ucdp_source_record():
    return SourceRecord(
        "ucdp",
        "context-2026-10-14",
        "UCDP",
        "https://ucdp.uu.se/apidocs/index.html",
        access="free",
        availability="receipt",
        license_notes="Free token on request; retain dataset citation/version and license.",
    )


def collect_ucdp(
    query,
    store,
    *,
    snapshot_id,
    api_token=None,
    transport=None,
    offline=None,
    limits=None,
    published_at=None,
):
    """Capture one explicit page, no implicit paging/retry or bulk download.

    API token travels only in x-ucdp-access-token, never query/cache metadata.
    Annual/event dates are economic labels; available_at is the actual receipt.
    complete means this supplied query's pages were fully returned, not a known
    global universe or historically available dataset.
    """
    query = UCDPQuery(**query) if isinstance(query, dict) else query
    limits = limits or CaptureLimits()
    params = dict(pagesize=min(1000, limits.max_rows), page=query.page)
    if query.resource == "gedevents":
        params.update(StartDate=query.start, EndDate=query.end)
    else:
        params["Year"] = ",".join(
            str(y) for y in range(int(query.start[:4]), int(query.end[:4]) + 1)
        )
    if query.countries:
        params["Country"] = ",".join(map(str, query.countries))

    def fetch(*, max_bytes):
        if not isinstance(api_token, str) or not api_token.strip():
            raise ValueError("Live UCDP collection requires UCDP_API_TOKEN")
        client = transport or BoundedHTTPTransport({"ucdpapi.pcr.uu.se"}, min_interval=1)
        return client.get(
            f"https://ucdpapi.pcr.uu.se/api/{query.resource}/{query.version}",
            params=params,
            max_bytes=max_bytes,
            headers={"x-ucdp-access-token": api_token},
        )

    def parser(body, *, received_at, max_rows):
        frame = parse_ucdp(
            body, query=query, received_at=received_at, max_rows=max_rows, published_at=published_at
        )
        payload = _json(body)
        total = _integer(payload.get("TotalCount"), "TotalCount")
        pages = _integer(payload.get("TotalPages"), "TotalPages")
        if total < len(frame) or pages == 0 and total != 0:
            raise ValueError("UCDP pagination metadata is inconsistent")
        complete = query.page == 1 and pages <= 1 and len(frame) == total
        return frame, () if complete else ("ucdp_query_pages_incomplete",)

    declaration = ucdp_source_record()
    return capture_observations(
        store=store,
        source="ucdp",
        dataset=query.resource,
        snapshot_id=snapshot_id,
        query=dict(
            asdict(query),
            countries=list(query.countries),
            published_at=_instant(published_at, "published_at").isoformat()
            if published_at is not None
            else None,
        ),
        parser=parser,
        declaration=declaration.to_dict(),
        fetch=fetch,
        offline=offline,
        limits=limits,
    )
