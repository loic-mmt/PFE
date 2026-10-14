"""Bounded FRED/ALFRED snapshots with explicit date-level vintage uncertainty.

API v1 output_type=1 is supported. Real-time/vintage dates never become an
intraday publication timestamp. Source units remain unchanged; missing '.' is
null, never zero. Metadata is a separately received snapshot, not proof that its
labels or schedules were known at a reconstructed historical decision.
"""

import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pfe.data.contracts import OBSERVATION_COLUMNS, _instant, normalize_observations
from pfe.data.macro import SERIES_COLUMNS, normalize_macro_series
from pfe.data.qualification import sanitize_metadata
from pfe.data.sources import default_source_registry

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_DOCUMENTATION = {
    "observations": "https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
    "vintages": "https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html",
    "realtime": "https://fred.stlouisfed.org/docs/api/fred/realtime_period.html",
    "series": "https://fred.stlouisfed.org/docs/api/fred/series.html",
    "release": "https://fred.stlouisfed.org/docs/api/fred/series_release.html",
    "release_dates": "https://fred.stlouisfed.org/docs/api/fred/release_dates.html",
}
FRED_COLUMNS = (
    *OBSERVATION_COLUMNS,
    "ingested_at",
    "availability_basis",
    "point_in_time",
    "series_id",
    "realtime_start",
    "realtime_end",
    "frequency",
    "seasonal_adjustment",
    "title",
    "units_short",
    "last_updated",
    "source_missing",
    "vintage_timezone",
    "availability_convention",
    "metadata_available_at",
    "metadata_point_in_time",
)


class FREDError(ValueError):
    """Invalid, unqualified or inconsistent FRED data."""


def _date(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise FREDError(f"{name} requires an explicit ISO date")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise FREDError(f"{name} requires a valid ISO date") from error
    return value


def _zone(value):
    try:
        return ZoneInfo(value)
    except (KeyError, ValueError, TypeError) as error:
        raise FREDError("Timezone requires a valid explicit IANA name") from error


def _series_metadata(metadata):
    if not isinstance(metadata, dict):
        raise FREDError("FRED series metadata requires one object")
    for name in ("id", "units", "title", "frequency", "seasonal_adjustment"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise FREDError(f"FRED series metadata requires {name}")
    return sanitize_metadata(metadata)


def normalize_fred_observations(
    rows,
    series_metadata,
    *,
    received_at,
    availability="receipt",
    vintage_timezone="America/New_York",
):
    """Adapt API output_type=1 rows into the common observation contract.

    Default receipt assigns current receipt availability and point_in_time=False.
    Use a single real-time snapshot with receipt mode: multiple same-period
    versions at the same receipt are ambiguous and rejected. Explicit
    vintage_next_day uses midnight NEXT calendar day in vintage_timezone,
    basis reconstructed and point_in_time=False. This conservative convention is
    not a provider publication/delivery guarantee. published_at always remains null.

    Date-only realtime bounds remain strings, including 9999-12-31. Units,
    frequency, title, seasonal adjustment and last_updated are source metadata;
    last_updated never establishes publication of each observation. '.' becomes
    an explicit null observation and remains in raw/source audit counts.
    """
    metadata = _series_metadata(series_metadata)
    receipt = _instant(received_at, "received_at")
    if availability not in ("receipt", "vintage_next_day"):
        raise FREDError("availability must be receipt or vintage_next_day")
    zone = _zone(vintage_timezone)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise FREDError("FRED observations require an array of objects")
    records = []
    for row in rows:
        period = _date(row.get("date"), "observation date")
        first = _date(row.get("realtime_start"), "realtime_start")
        last = _date(row.get("realtime_end"), "realtime_end")
        if first > last:
            raise FREDError("realtime_start cannot follow realtime_end")
        raw_value = row.get("value")
        if raw_value == ".":
            value = None
        elif isinstance(raw_value, str) and re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw_value
        ):
            value = float(raw_value)
            if not np.isfinite(value):
                raise FREDError("FRED value must be finite or the missing marker '.'")
        else:
            raise FREDError("FRED value requires a numeric source string or '.'")
        if availability == "receipt":
            available, basis = receipt, "receipt"
        else:
            try:
                tomorrow = date.fromisoformat(first) + timedelta(days=1)
                available = pd.Timestamp(tomorrow).tz_localize(zone).tz_convert("UTC")
            except (ValueError, OverflowError) as error:
                raise FREDError(
                    "Vintage date cannot be represented under next-day convention"
                ) from error
            basis = "reconstructed"
        # Real-time end is query-clipped, so it is evidence, not value-version identity.
        identity = [metadata["id"], period, first, raw_value, metadata["units"]]
        revision = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        records.append(
            dict(
                entity_id=metadata["id"],
                observation_date=period,
                published_at=None,
                available_at=available,
                value=value,
                source_id="fred",
                revision_id=revision,
                metric=metadata["id"],
                unit=metadata["units"],
                schema_version="1",
                ingested_at=receipt,
                availability_basis=basis,
                point_in_time=False,
                series_id=metadata["id"],
                realtime_start=first,
                realtime_end=last,
                frequency=metadata["frequency"],
                seasonal_adjustment=metadata["seasonal_adjustment"],
                title=metadata["title"],
                units_short=metadata.get("units_short"),
                last_updated=metadata.get("last_updated"),
                source_missing=value is None,
                vintage_timezone=vintage_timezone,
                availability_convention=availability,
                metadata_available_at=receipt,
                metadata_point_in_time=False,
            )
        )
    result = normalize_observations(pd.DataFrame(records, columns=FRED_COLUMNS))
    result["metadata_available_at"] = pd.Series(
        [receipt] * len(result), dtype="datetime64[ns, UTC]", index=result.index
    )
    return result


def fred_to_macro_series(observations, *, source_id=None, observation_timezone="UTC"):
    """Bridge every vintage to the existing sparse macro engine; never latest-only.

    Economic date anchoring is explicit midnight in observation_timezone, used
    only to order periods. It is not publication. Different sources must be
    selected explicitly. Initial missing periods are omitted from numeric input;
    later missing vintages are tombstones, retaining the economic period and
    preventing an earlier value from becoming visible again.
    Units, frequency and seasonal adjustment must remain consistent per metric
    across all periods/vintages; reconcile changed metadata before computing
    rolling statistics. These declarations remain in frame.attrs.
    """
    observation_zone = _zone(observation_timezone)
    out = normalize_observations(observations)
    if source_id is not None:
        out = out.loc[out.source_id.eq(source_id)]
    if out.source_id.nunique() > 1:
        raise FREDError("Select one macro source before bridging vintages")
    if out.groupby("metric").entity_id.nunique().gt(1).any():
        raise FREDError(
            "Multiple entities share one macro metric; select distinct series identities"
        )
    if out.groupby("metric").unit.nunique().gt(1).any():
        raise FREDError("Macro series units changed; reconcile explicitly before bridging")
    for field in ("frequency", "seasonal_adjustment"):
        if (
            field not in out
            or not out[field]
            .map(lambda value: isinstance(value, str) and bool(value) and value == value.strip())
            .all()
        ):
            raise FREDError(f"Macro series require explicit qualified {field} metadata")
        if out.groupby("metric")[field].nunique(dropna=False).gt(1).any():
            raise FREDError(f"Macro series {field} changed; reconcile explicitly before bridging")
    seen = set()
    records = []
    for row in out.itertuples(index=False):
        key = (row.entity_id, row.metric, row.observation_date)
        if row.value is None:
            if key not in seen:
                continue
        elif not isinstance(row.value, (int, float)):
            raise FREDError("Macro observations require numeric values")
        seen.add(key)
        period = _date(row.observation_date, "observation_date")
        anchor = pd.Timestamp(period).tz_localize(observation_zone).tz_convert("UTC")
        records.append(
            dict(
                series=row.metric,
                observation_at=anchor,
                available_at=row.available_at,
                value=row.value,
                point_in_time=row.point_in_time,
            )
        )
    result = normalize_macro_series(pd.DataFrame(records, columns=SERIES_COLUMNS))
    result.attrs["units"] = out.groupby("metric").unit.first().to_dict()
    for field in ("frequency", "seasonal_adjustment"):
        result.attrs[field] = out.groupby("metric")[field].first().to_dict()
    result.attrs["economic_date_anchor"] = f"midnight {observation_timezone}; not publication"
    return result


def compute_fred_macro_features(
    bars, *, observations, pairs=(), source_id=None, observation_timezone="UTC", **options
):
    """Compute existing sparse macro features after checking spread source units.

    Pair units, frequency and seasonal adjustment must match exactly. Values
    retain provider scales: two Percent rates produce a spread in percentage
    points, not a silently divided decimal rate. All PIT flags remain False for
    this adapter's receipt/reconstructed data.
    """
    from pfe.features.macro import compute_macro_features

    pairs = tuple(tuple(pair) for pair in pairs)

    out = normalize_observations(observations)
    if source_id is not None:
        out = out.loc[out.source_id.eq(source_id)]
    units = {}
    for pair in pairs:
        if len(pair) != 2:
            raise FREDError("Spread pairs require two series IDs")
        left, right = (out.loc[out.metric.eq(series)] for series in pair)
        for field in ("unit", "frequency", "seasonal_adjustment"):
            if (
                field not in out
                or left[field].nunique() != 1
                or right[field].nunique() != 1
                or left[field].iloc[0] != right[field].iloc[0]
            ):
                raise FREDError(
                    "Spread units, frequency and seasonal adjustment "
                    "must be qualified and compatible"
                )
        units[f"macro.pair[{pair[0]},{pair[1]}].spread"] = (
            "percentage_points" if left.unit.iloc[0] == "Percent" else left.unit.iloc[0]
        )
    result = compute_macro_features(
        bars,
        observations=fred_to_macro_series(
            out, source_id=source_id, observation_timezone=observation_timezone
        ),
        pairs=pairs,
        **options,
    )
    result.attrs["spread_units"] = units
    return result


def load_fred_observations(path):
    """Load validated common-contract FRED Parquet observations."""
    return normalize_observations(pd.read_parquet(Path(path)))


def compute_fred_features(
    decisions, *, observations, pairs=(), source_id=None, observation_timezone="UTC", **options
):
    """Align macro features directly to explicit decision instants.

    Accept an aware DatetimeIndex or a DataFrame with typed entity_id/decision_at.
    Input order and index are preserved, including repeated decisions across
    entities. DataFrame results retain entity_id and UTC decision_at. The sparse
    macro engine runs once on unique decisions; the one-nanosecond internal bar
    bridge is exactly cancelled by its close offset and never dates publication.
    source_frequency is not accepted because decisions are already explicit.
    """
    if "source_frequency" in options:
        raise FREDError("Explicit decisions do not require source_frequency")
    if isinstance(decisions, pd.DataFrame):
        if not decisions.columns.is_unique or not {"entity_id", "decision_at"} <= set(decisions):
            raise FREDError("Decision frames require unique entity_id and decision_at columns")
        if not decisions.entity_id.map(
            lambda value: isinstance(value, str) and bool(value) and value == value.strip()
        ).all():
            raise FREDError("Decision entity_id requires non-empty typed strings")
        instants = pd.DatetimeIndex(
            [_instant(value, "decision_at") for value in decisions.decision_at], tz="UTC"
        )
        labels = decisions.loc[:, ["entity_id", "decision_at"]].copy()
        labels["decision_at"] = instants
        original_index = decisions.index
    elif isinstance(decisions, pd.DatetimeIndex):
        if decisions.tz is None or decisions.hasnans:
            raise FREDError("Decision index requires non-missing aware timestamps")
        instants, labels, original_index = decisions.tz_convert("UTC"), None, decisions
    else:
        raise FREDError("Decisions require an aware DatetimeIndex or typed decision DataFrame")
    try:
        engine_index = instants.unique().sort_values() - pd.Timedelta("1ns")
    except (ValueError, OverflowError) as error:
        raise FREDError("Decision timestamps cannot be represented by the macro bridge") from error
    computed = compute_fred_macro_features(
        engine_index,
        observations=observations,
        pairs=pairs,
        source_id=source_id,
        observation_timezone=observation_timezone,
        source_frequency="1ns",
        **options,
    )
    result = computed.reindex(instants - pd.Timedelta("1ns"))
    result.index = original_index
    attributes = computed.attrs.copy()
    masks = result.notna().rename(columns=lambda name: f"mask.{name}")
    result = pd.concat([result, masks], axis=1)
    result["available_at"] = instants
    if labels is not None:
        result = pd.concat([labels, result], axis=1)
    result.attrs.update(attributes)
    result.attrs.update(timestamp_semantics="explicit_decision", available_at="decision_at")
    return result


def collect_fred_series(
    transport,
    store,
    *,
    series_id,
    start,
    end,
    snapshot_id,
    realtime_start,
    realtime_end,
    api_key=None,
    availability="receipt",
    vintage_timezone="America/New_York",
    page_size=1000,
    max_rows=20_000,
    max_bytes=1_048_576,
    max_requests=20,
    source_metadata=None,
):
    """Collect metadata, observations, vintage dates and date-only release metadata.

    Native units/output_type=1 only; query dates are explicit and fixed across
    pagination. max_rows/bytes cover all saved source rows/responses, including
    cached pages; max_requests bounds new HTTP calls in this invocation. Caps
    produce an explicit incomplete summary and continuation; no absence inferred.
    A network/byte failure leaves already saved pages resumable via RawStore.
    API keys are added only to transient query params, never to storage identity.
    """
    for name, value in (("series_id", series_id), ("snapshot_id", snapshot_id)):
        if not isinstance(value, str) or not value or value != value.strip():
            raise FREDError(f"{name} requires a non-empty identifier")
    start, end = _date(start, "start"), _date(end, "end")
    first, last = _date(realtime_start, "realtime_start"), _date(realtime_end, "realtime_end")
    if start > end or first > last:
        raise FREDError("Observation and real-time date bounds must be ordered")
    if (
        availability not in ("receipt", "vintage_next_day")
        or availability == "receipt"
        and first != last
    ):
        raise FREDError(
            "Receipt mode requires one real-time snapshot; history needs vintage_next_day"
        )
    _zone(vintage_timezone)
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise FREDError("api_key requires a non-empty credential string or null")
    for name, value in (
        ("page_size", page_size),
        ("max_rows", max_rows),
        ("max_bytes", max_bytes),
        ("max_requests", max_requests),
    ):
        if type(value) is not int or value < 1:
            raise FREDError(f"{name} must be a positive integer")
    if page_size > 10_000:
        raise FREDError("Common FRED page_size must not exceed 10000")
    request = dict(
        series_id=series_id,
        start=start,
        end=end,
        snapshot_id=snapshot_id,
        realtime_start=first,
        realtime_end=last,
        availability=availability,
        vintage_timezone=vintage_timezone,
        page_size=page_size,
        max_rows=max_rows,
        max_bytes=max_bytes,
        max_requests=max_requests,
        output_type=1,
        units="lin",
    )
    if source_metadata is None:
        source_metadata = default_source_registry().get("fred").to_dict()
    collection_id = store.begin("fred", "macro_vintages", request, source_metadata=source_metadata)
    store.recover(collection_id)
    manifest = store.manifest(collection_id)
    saved = {page["page_id"] for page in manifest["pages"]}
    source_pages, used_bytes, used_rows, requests = [], 0, 0, 0
    coverage, issues, continuation = {}, [], None

    def fetch(endpoint, params, page_id):
        nonlocal used_bytes, requests
        if page_id in saved:
            page = json.loads(store.read_page(collection_id, page_id))
        else:
            if requests >= max_requests or used_bytes >= max_bytes:
                return None
            if transport is None:
                raise FREDError(
                    "Uncached FRED page requires FRED_API_KEY and an explicit transport"
                )
            requests += 1
            response = transport.get(
                FRED_BASE_URL + endpoint,
                params={**params, **({"api_key": api_key} if api_key else {})},
                max_bytes=max_bytes - used_bytes,
            )
            if len(response.body) > max_bytes - used_bytes:
                raise FREDError("FRED response exceeds total byte cap")
            try:
                payload = json.loads(response.body)
            except (ValueError, UnicodeError, RecursionError):
                raise FREDError("Invalid FRED JSON response") from None
            payload = sanitize_metadata(payload, secrets=(api_key,))
            page = dict(
                endpoint=endpoint,
                params=params,
                received_at=response.received_at,
                response_bytes=len(response.body),
                payload=payload,
                original_sha256=hashlib.sha256(response.body).hexdigest(),
                content_type=response.content_type,
            )
            _instant(page["received_at"], "received_at")
            expected_key = {
                "/series": "seriess",
                "/series/release": "releases",
                "/series/observations": "observations",
                "/series/vintagedates": "vintage_dates",
                "/release/dates": "release_dates",
            }[endpoint]
            if not isinstance(payload, dict) or not isinstance(payload.get(expected_key), list):
                raise FREDError("FRED response requires its documented row array")
            count = len(payload[expected_key])
            if count > max_rows - used_rows:
                raise FREDError("FRED response exceeds total row cap")
            store.write_page(collection_id, page_id, page, row_count=count)
        if (
            page.get("endpoint") != endpoint
            or page.get("params") != params
            or type(page.get("response_bytes")) is not int
            or page["response_bytes"] < 1
        ):
            raise FREDError("Invalid saved FRED page identity")
        used_bytes += page["response_bytes"]
        if used_bytes > max_bytes:
            raise FREDError("Saved FRED pages exceed total byte cap")
        source_pages.append(page)
        return page

    def walk(endpoint, params, key, stage):
        nonlocal used_rows, continuation
        offset, count, rows, receipts = 0, None, [], []
        coverage[stage] = False
        while True:
            remaining = max_rows - used_rows
            if remaining <= 0 or used_bytes >= max_bytes:
                issues.append("total_row_or_byte_cap")
                continuation = continuation or json.dumps(
                    dict(stage=stage, offset=offset), sort_keys=True
                )
                break
            limit = min(page_size, remaining)
            query = {**params, "limit": limit, "offset": offset}
            page = fetch(endpoint, query, f"{stage}-{offset}")
            if page is None:
                issues.append("request_cap")
                continuation = continuation or json.dumps(
                    dict(stage=stage, offset=offset), sort_keys=True
                )
                break
            payload = page["payload"]
            if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
                raise FREDError("FRED pagination requires an explicit row array")
            batch = payload[key]
            if (
                any(type(payload.get(name)) is not int for name in ("count", "offset", "limit"))
                or payload["count"] < 0
                or payload["offset"] != offset
                or payload["limit"] != limit
                or len(batch) > limit
                or offset + len(batch) > payload["count"]
            ):
                raise FREDError("Invalid FRED pagination counts or offset")
            if count is not None and count != payload["count"]:
                raise FREDError("FRED count changed across pages; start a new snapshot")
            count = payload["count"]
            rows.extend(batch)
            receipts.extend([page["received_at"]] * len(batch))
            used_rows += len(batch)
            offset += len(batch)
            if offset == count:
                coverage[stage] = True
                break
            if not batch:
                raise FREDError("FRED pagination did not advance")
        return rows, receipts

    fixed = dict(series_id=series_id, file_type="json", realtime_start=last, realtime_end=last)
    metadata_page = fetch("/series", fixed, "series")
    if metadata_page is None:
        raise FREDError("FRED series metadata unavailable under request cap")
    payload = metadata_page["payload"]
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("seriess"), list)
        or len(payload["seriess"]) != 1
    ):
        raise FREDError("Expected exactly one FRED series metadata record")
    metadata = _series_metadata(payload["seriess"][0])
    if metadata["id"] != series_id:
        raise FREDError("FRED metadata disagrees with requested series")
    used_rows += 1
    release_page = fetch("/series/release", fixed, "release") if used_rows < max_rows else None
    releases = [] if release_page is None else release_page["payload"].get("releases")
    if not isinstance(releases, list) or any(
        not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] < 1
        for row in releases
    ):
        raise FREDError("FRED series releases require explicit release IDs")
    used_rows += len(releases)
    if used_rows > max_rows:
        raise FREDError("FRED releases exceed total row cap")
    coverage["release"] = release_page is not None
    if release_page is None:
        issues.append("release_metadata_cap")
        continuation = json.dumps(dict(stage="release", offset=0), sort_keys=True)
    realtime = dict(series_id=series_id, file_type="json", realtime_start=first, realtime_end=last)
    rows, receipts = walk(
        "/series/observations",
        {
            **realtime,
            "observation_start": start,
            "observation_end": end,
            "units": "lin",
            "output_type": 1,
            "sort_order": "asc",
        },
        "observations",
        "observations",
    )
    receipt_rows = {}
    for row, receipt in zip(rows, receipts, strict=True):
        if (
            not isinstance(row, dict)
            or not start <= _date(row.get("date"), "observation date") <= end
        ):
            raise FREDError("FRED observation outside requested economic date bounds")
        if not first <= _date(row.get("realtime_start"), "realtime_start") <= last:
            raise FREDError("FRED row starts outside requested real-time bounds")
        if _date(row.get("realtime_end"), "realtime_end") > last:
            raise FREDError("FRED row ends outside requested real-time bounds")
        receipt_rows.setdefault(receipt, []).append(row)
    observations = [
        normalize_fred_observations(
            batch,
            metadata,
            received_at=receipt,
            availability=availability,
            vintage_timezone=vintage_timezone,
        )
        for receipt, batch in receipt_rows.items()
    ]
    frame = (
        normalize_observations(pd.concat(observations, ignore_index=True))
        if observations
        else normalize_fred_observations(
            [],
            metadata,
            received_at=metadata_page["received_at"],
            availability=availability,
            vintage_timezone=vintage_timezone,
        )
    )
    frame["source_document_id"] = f"fred-raw:{collection_id}"
    frame["metadata_available_at"] = pd.Series(
        [_instant(metadata_page["received_at"], "metadata_available_at")] * len(frame),
        dtype="datetime64[ns, UTC]",
        index=frame.index,
    )
    vintages, _ = walk(
        "/series/vintagedates", {**realtime, "sort_order": "asc"}, "vintage_dates", "vintages"
    )
    vintages = [_date(value, "vintage date") for value in vintages]
    release_dates = []
    for release in releases:
        dates, _ = walk(
            "/release/dates",
            dict(
                release_id=release["id"],
                file_type="json",
                realtime_start=first,
                realtime_end=last,
                sort_order="asc",
                include_release_dates_with_no_data="false",
            ),
            "release_dates",
            f"release-dates-{release['id']}",
        )
        for row in dates:
            if not isinstance(row, dict) or row.get("release_id") != release["id"]:
                raise FREDError("Invalid FRED release-date identity")
            _date(row.get("date"), "release date")
        release_dates.extend(dates)
    summary = dict(
        collection_id=collection_id,
        series_metadata=metadata,
        releases=releases,
        vintage_dates=vintages,
        release_dates=release_dates,
        coverage=coverage,
        pagination_complete=all(coverage.values()),
        rows=len(frame),
        missing_rows=int(frame.value.isna().sum()),
        source_rows=used_rows,
        response_bytes=used_bytes,
        issues=sorted(set(issues)),
        publication_timing="unqualified",
        absence_scope="unqualified",
        documentation=FRED_DOCUMENTATION,
    )
    summary_id = f"summary-{len(source_pages)}"
    if summary_id not in saved:
        store.write_page(
            collection_id,
            summary_id,
            summary,
            next_cursor=continuation,
            complete=summary["pagination_complete"],
            row_count=0,
            metadata={
                "completion_scope": "requested_api_pages",
                "incomplete": not summary["pagination_complete"],
            },
        )
    return collection_id, frame, summary
