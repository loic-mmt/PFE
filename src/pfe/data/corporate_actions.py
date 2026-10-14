"""Revision-aware corporate actions and local, availability-safe return bridges.

Ratios always mean new shares per old share. Daily returns bridge only the
current price interval; no future split back-adjusts absolute historical prices.
LSE 0.14 documents reference paths and dates, but not amount/ratio row schemas:
provider numeric fields and share conventions therefore require explicit maps.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from numbers import Real

import numpy as np
import pandas as pd

from pfe.data.contracts import (
    _frame,
    _identifiers,
    _instant,
    _missing,
    _provenance,
    _timestamps,
    _vintages,
)
from pfe.data.qualification import DiscoveryError, DiscoveryLimitError, sanitize_metadata
from pfe.data.sources import default_source_registry

ACTION_COLUMNS = (
    "action_id",
    "entity_id",
    "instrument_id",
    "source_id",
    "revision_id",
    "schema_version",
    "action_type",
    "announcement_at",
    "published_at",
    "available_at",
    "ex_date",
    "effective_date",
    "ratio",
    "amount",
    "currency",
)
ACTION_TYPES = ("split", "dividend")
DIVIDEND_BASES = ("pre_split_share", "post_split_share", "unknown")
PRICE_ADJUSTMENTS = ("raw", "split_adjusted", "total_return", "unknown")
VOLUME_ADJUSTMENTS = ("raw", "split_adjusted", "unknown")
LSE_ACTION_ENDPOINTS = {"split": "/ref/stock_splits", "dividend": "/ref/dividends"}
LSE_ACTION_DOCUMENTATION = "https://github.com/londonstrategicedge/lse-data/blob/main/lse/client.py"


def _date(value, name, *, nullable=False):
    if nullable and _missing(value):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            pass
    raise ValueError(f"{name} requires an explicit valid ISO session date")


def _number(value, name, *, nullable=False, positive=False):
    if nullable and _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} requires a finite numeric value")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} requires a finite numeric value") from error
    if not np.isfinite(number) or number < 0 or positive and number == 0:
        raise ValueError(f"{name} requires a {'positive' if positive else 'non-negative'} value")
    return number


def normalize_corporate_actions(frame):
    """Validate immutable source/action revisions, preserving economic date precision.

    ACTION_COLUMNS are required; announcement/publication, revision_id, currency
    for splits, and split ex_date can be null. available_at is always an aware
    instant. Optional ingestion, availability_basis and point_in_time follow the
    common observation contract. A receipt is not a historical announcement.

    Split ratio is positive new-shares/old-share; split amount must be null.
    Dividend amount is non-negative currency per declared share and requires
    currency/ex_date; dividend ratio must be null. Optional amount_unit defaults
    'unknown', dividend_basis defaults 'unknown': returns cannot use an unknown
    cash/share convention. status defaults confirmed; a cancelled revision
    suppresses an earlier action after that revision becomes available.

    IDs are stable per source/action across revisions, including date corrections.
    Equal-time revisions and reused revision IDs are rejected. Sources remain
    independent and are never silently added as duplicate economic actions.
    """
    out = _frame(frame, ACTION_COLUMNS, "Corporate actions")
    _identifiers(
        out,
        (
            "action_id",
            "entity_id",
            "instrument_id",
            "source_id",
            "revision_id",
            "schema_version",
            "currency",
        ),
        nullable=("revision_id", "currency"),
    )
    if not out.action_type.isin(ACTION_TYPES).all():
        raise ValueError(f"action_type must be one of {ACTION_TYPES}")
    out["announcement_at"] = _timestamps(out.announcement_at, "announcement_at", nullable=True)
    _provenance(out)
    if (out.available_at < out.announcement_at).any():
        raise ValueError("available_at cannot precede announcement_at")
    out["ex_date"] = pd.Series(
        [_date(v, "ex_date", nullable=True) for v in out.ex_date], dtype=object
    )
    out["effective_date"] = pd.Series(
        [_date(v, "effective_date") for v in out.effective_date], dtype=object
    )
    ratios, amounts = [], []
    for row in out.itertuples(index=False):
        if row.action_type == "split":
            ratios.append(_number(row.ratio, "ratio", positive=True))
            if not _missing(row.amount):
                raise ValueError("A split cannot carry a dividend amount")
            amounts.append(None)
        else:
            if not _missing(row.ratio):
                raise ValueError("A dividend cannot carry a split ratio")
            if row.ex_date is None or row.currency is None:
                raise ValueError("A dividend requires ex_date and currency")
            ratios.append(None)
            amounts.append(_number(row.amount, "amount"))
    out["ratio"] = pd.Series(ratios, dtype=object)
    out["amount"] = pd.Series(amounts, dtype=object)
    if "status" not in out:
        out["status"] = "confirmed"
    if not out.status.isin(("confirmed", "cancelled")).all():
        raise ValueError("status must be confirmed or cancelled")
    if "dividend_basis" not in out:
        out["dividend_basis"] = "unknown"
    if not out.dividend_basis.isin(DIVIDEND_BASES).all():
        raise ValueError(f"dividend_basis must be one of {DIVIDEND_BASES}")
    if "amount_unit" not in out:
        out["amount_unit"] = "unknown"
    if not out.amount_unit.isin(("currency_per_share", "unknown")).all():
        raise ValueError("amount_unit must be currency_per_share or unknown")
    keys = ["source_id", "action_id"]
    _vintages(out, keys)
    if out.groupby(keys, dropna=False).action_type.nunique().gt(1).any():
        raise ValueError("An action_id cannot change action_type across revisions")
    return out.sort_values("available_at", kind="stable", ignore_index=True)


def corporate_actions_asof(frame, decision_at, effective_date=None):
    """Latest known revision per source/action, then optional economic cutoff.

    Cancelled actions are removed after selecting their latest known revision.
    Without effective_date, known future announcements remain visible. Economic
    cutoff uses split effective_date or dividend ex_date; it never creates an
    availability timestamp from that date.
    """
    decision = _instant(decision_at, "decision_at")
    out = normalize_corporate_actions(frame)
    out = _known_actions(out, decision)
    if effective_date is not None:
        cutoff = _date(effective_date, "effective_date")
        economic = out.effective_date.where(out.action_type.eq("split"), out.ex_date)
        out = out.loc[economic <= cutoff]
    return out.reset_index(drop=True)


def _known_actions(history, decision):
    known = history.loc[history.available_at <= decision].drop_duplicates(
        ["source_id", "action_id"], keep="last"
    )
    return known.loc[known.status.eq("confirmed")]


def corporate_action_returns(
    prices,
    actions,
    *,
    price_adjustment,
    volume_adjustment,
    include_dividends=False,
    action_source_id=None,
    coverage_complete=False,
):
    """Return interval-local price returns and a separate optional cash total return.

    Prices require instrument_id/entity_id/session_date/session_close/close/currency/available_at.
    available_at is each return's decision cutoff; both prices must be available
    then. Bar declarations, when present, must agree with explicit arguments.
    Unknown price/volume conventions and total-return-adjusted input are rejected.
    Split-adjusted prices are used directly; adding raw-share dividends to them
    is rejected because their monetary/share scale is not established.

    For raw prices: price_return = close * known split factors / previous close -1.
    Optional total_return also adds same-currency cash per original held share.
    pre_split_share dividends precede same-day splits, post_split_share dividends
    follow them. Only available actions effective in this interval are used;
    late receipt/revision never rewrites an earlier return decision. Currency
    conversion, taxes, reinvestment and volume adjustment are not performed.
    coverage_complete defaults False; no-known-action does not prove absence.
    """
    if price_adjustment not in PRICE_ADJUSTMENTS or price_adjustment in ("unknown", "total_return"):
        raise ValueError("Returns require raw or split_adjusted prices; double/unknown adjustment")
    if volume_adjustment not in VOLUME_ADJUSTMENTS or volume_adjustment == "unknown":
        raise ValueError("An explicit raw or split_adjusted volume convention is required")
    if type(include_dividends) is not bool or type(coverage_complete) is not bool:
        raise ValueError("include_dividends and coverage_complete require explicit booleans")
    if include_dividends and price_adjustment != "raw":
        raise ValueError("Cash dividends require raw prices with an established per-share scale")
    out = _frame(
        prices,
        (
            "instrument_id",
            "entity_id",
            "session_date",
            "session_close",
            "close",
            "currency",
            "available_at",
        ),
        "Daily return prices",
    )
    _identifiers(out, ("instrument_id", "entity_id", "currency"))
    out["session_date"] = [_date(v, "session_date") for v in out.session_date]
    out["available_at"] = _timestamps(out.available_at, "available_at")
    out["session_close"] = _timestamps(out.session_close, "session_close")
    if (out.available_at < out.session_close).any():
        raise ValueError("A daily price cannot be available before its session_close")
    out["close"] = [_number(v, "close", positive=True) for v in out.close]
    if out.duplicated(["instrument_id", "session_date"]).any():
        raise ValueError("Select one price vintage per instrument/session before computing returns")
    for column, declaration in (
        ("price_adjustment", price_adjustment),
        ("volume_adjustment", volume_adjustment),
    ):
        if column in out and not out[column].eq(declaration).all():
            raise ValueError(f"{column} disagrees with the explicit input convention")
    if "price_unit" in out and not out.price_unit.eq("currency_per_share").all():
        raise ValueError("Prices require currency_per_share units")
    action_history = normalize_corporate_actions(actions)
    if action_source_id is not None:
        if not isinstance(action_source_id, str) or not action_source_id:
            raise ValueError("action_source_id requires a non-empty source identifier")
        action_history = action_history.loc[action_history.source_id.eq(action_source_id)]
    out = out.sort_values(["instrument_id", "session_date"], kind="stable", ignore_index=True)
    records = []
    for _, group in out.groupby("instrument_id", sort=False):
        previous = None
        for row in group.to_dict("records"):
            record = dict(row)
            record.update(
                input_price_return=np.nan,
                price_return=np.nan,
                total_return=np.nan,
                split_factor=1.0,
                cash_dividend=0.0,
                split_action_ids=[],
                dividend_action_ids=[],
                reconciliation_status="first_observation",
                action_coverage_complete=coverage_complete,
            )
            if previous is not None:
                if (
                    previous["entity_id"] != row["entity_id"]
                    or previous["currency"] != row["currency"]
                ):
                    raise ValueError("Price identity/currency changed; reconcile explicitly")
                if previous["available_at"] > row["available_at"]:
                    raise ValueError(
                        "Previous price is not available at the current return decision"
                    )
                if previous["session_close"] >= row["session_close"]:
                    raise ValueError("Daily session closes must strictly increase")
                known = _known_actions(action_history, row["available_at"])
                known = known.loc[known.instrument_id.eq(row["instrument_id"])]
                if not known.entity_id.eq(row["entity_id"]).all():
                    raise ValueError("Action entity disagrees with the price instrument")
                economic = known.effective_date.where(known.action_type.eq("split"), known.ex_date)
                interval = known.loc[
                    (economic > previous["session_date"]) & (economic <= row["session_date"])
                ]
                if interval.source_id.nunique() > 1:
                    raise ValueError(
                        "Select one action source; cross-source reconciliation is ambiguous"
                    )
                splits = interval.loc[interval.action_type.eq("split")]
                factor = math.prod(splits.ratio.to_numpy(dtype=float).tolist())
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("Combined split ratio is invalid")
                used_factor = factor if price_adjustment == "raw" else 1.0
                record["input_price_return"] = row["close"] / previous["close"] - 1
                record["price_return"] = row["close"] * used_factor / previous["close"] - 1
                if not np.isfinite(record["price_return"]):
                    raise ValueError("Reconciled price return is not finite")
                record["split_factor"] = used_factor
                record["split_action_ids"] = splits.action_id.tolist()
                record["reconciliation_status"] = (
                    "already_split_adjusted"
                    if price_adjustment == "split_adjusted"
                    else "split_reconciled"
                    if len(splits)
                    else "no_known_split"
                )
                if include_dividends:
                    dividends = interval.loc[interval.action_type.eq("dividend")]
                    cash = 0.0
                    for dividend in dividends.itertuples(index=False):
                        if dividend.currency != row["currency"]:
                            raise ValueError(
                                "Dividend and price currency must agree; no implicit FX"
                            )
                        if (
                            dividend.amount_unit != "currency_per_share"
                            or dividend.dividend_basis == "unknown"
                        ):
                            raise ValueError(
                                "Dividend cash/share convention must be explicitly qualified"
                            )
                        before = (
                            splits.effective_date <= dividend.ex_date
                            if dividend.dividend_basis == "post_split_share"
                            else splits.effective_date < dividend.ex_date
                        )
                        shares = math.prod(
                            splits.loc[before, "ratio"].to_numpy(dtype=float).tolist()
                        )
                        cash += dividend.amount * shares
                    if not np.isfinite(cash):
                        raise ValueError("Reconciled cash dividend is not finite")
                    record["cash_dividend"] = cash
                    record["dividend_action_ids"] = dividends.action_id.tolist()
                    record["total_return"] = (row["close"] * factor + cash) / previous["close"] - 1
                    if not np.isfinite(record["total_return"]):
                        raise ValueError("Reconciled total return is not finite")
            records.append(record)
            previous = row
    return pd.DataFrame(
        records,
        columns=[
            *out.columns,
            "input_price_return",
            "price_return",
            "total_return",
            "split_factor",
            "cash_dividend",
            "split_action_ids",
            "dividend_action_ids",
            "reconciliation_status",
            "action_coverage_complete",
        ],
    )


def adapt_lse_actions(
    rows,
    *,
    action_type,
    entity_id,
    instrument_id,
    received_at,
    field_map,
    ratio_convention=None,
    dividend_basis="unknown",
    expected_symbol=None,
):
    """Adapt one dated LSE page using explicitly supplied provider field names.

    field_map maps canonical names to source field names. action_id/effective_date
    and ratio (split) or amount/currency (dividend) are required mappings. Dividend
    effective_date is the SDK's ex-date; ex_date can be mapped separately.
    Numeric ratio direction is required: new_per_old or old_per_new. No schema,
    currency, percentage or historical publication is guessed. Date-only source
    announcements remain raw evidence with null announcement/publication instants.
    """
    if action_type not in ACTION_TYPES or not isinstance(field_map, Mapping):
        raise ValueError("An action type and explicit provider field_map are required")
    required = {"action_id", "effective_date", "ratio" if action_type == "split" else "amount"}
    if action_type == "dividend":
        required.add("currency")
    allowed = {
        "symbol",
        "action_id",
        "effective_date",
        "ex_date",
        "ratio",
        "amount",
        "currency",
        "announcement_at",
        "published_at",
        "status",
    }
    if (
        not required <= set(field_map)
        or set(field_map) - allowed
        or any(not isinstance(v, str) or not v for v in field_map.values())
    ):
        raise ValueError("field_map must explicitly map required canonical action fields")
    if action_type == "split" and ratio_convention not in ("new_per_old", "old_per_new"):
        raise ValueError("Split ratio_convention must be new_per_old or old_per_new")
    if dividend_basis not in DIVIDEND_BASES:
        raise ValueError(f"dividend_basis must be one of {DIVIDEND_BASES}")
    if expected_symbol is not None and "symbol" not in field_map:
        raise ValueError("A collector requires an explicit provider symbol field mapping")
    receipt = _instant(received_at, "received_at")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("LSE action response requires a list of objects")
    records = []
    for source in rows:
        mapped = {name: source.get(field) for name, field in field_map.items()}
        if any(_missing(mapped[name]) for name in required):
            raise ValueError("Provider row is missing an explicitly mapped required field")
        if expected_symbol is not None and mapped["symbol"] != expected_symbol:
            raise ValueError("Provider corporate action symbol disagrees with requested identity")
        effective = _date(mapped["effective_date"], "effective_date")
        if action_type == "dividend" and "ex_date" in mapped and mapped["ex_date"] != effective:
            raise ValueError("LSE effective_date is the documented dividend ex-date")
        action_id = mapped["action_id"]
        if type(action_id) is int:
            action_id = str(action_id)
        ratio = None
        if action_type == "split":
            ratio = _number(mapped["ratio"], "ratio", positive=True)
            if ratio_convention == "old_per_new":
                ratio = 1.0 / ratio
        evidence = sanitize_metadata(source)
        revision = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        record = dict(
            action_id=action_id,
            entity_id=entity_id,
            instrument_id=instrument_id,
            source_id="lse",
            revision_id=revision,
            schema_version="1",
            action_type=action_type,
            announcement_at=None,
            published_at=None,
            available_at=receipt,
            ingested_at=receipt,
            availability_basis="receipt",
            point_in_time=False,
            effective_date=effective,
            ex_date=mapped.get("ex_date", effective if action_type == "dividend" else None),
            ratio=ratio,
            amount=mapped.get("amount") if action_type == "dividend" else None,
            currency=mapped.get("currency"),
            dividend_basis=dividend_basis,
            amount_unit="currency_per_share"
            if action_type == "dividend" and dividend_basis != "unknown"
            else "unknown",
            status=mapped.get("status", "confirmed"),
            source_payload=evidence,
        )
        for name in ("announcement_at", "published_at"):
            value = mapped.get(name)
            if not _missing(value):
                if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    _date(value, name)
                else:
                    record[name] = _instant(value, name)
        records.append(record)
    return normalize_corporate_actions(
        pd.DataFrame(
            records,
            columns=[
                *ACTION_COLUMNS,
                "ingested_at",
                "availability_basis",
                "point_in_time",
                "dividend_basis",
                "amount_unit",
                "status",
                "source_payload",
            ],
        )
    )


def collect_lse_actions(
    transport,
    store,
    *,
    symbol,
    entity_id,
    instrument_id,
    start,
    end,
    snapshot_id,
    field_maps,
    limit=5000,
    max_bytes=1_048_576,
    ratio_convention=None,
    dividend_basis="unknown",
    source_metadata=None,
):
    """Collect two bounded dated pages, journaling receipts before normalization.

    SDK interactive pages have no documented cursor. A full limit is explicitly
    incomplete and retained; use smaller reviewed date windows or a separately
    qualified export. Under-limit responses do not prove provider completeness:
    an account cap may be lower than the requested limit. A complete raw manifest
    means only both local pages were normalized without reaching their row caps.
    Repeating snapshot_id reuses saved pages; revisions require a new snapshot.
    max_bytes bounds total original response bytes and each stored page.
    """
    start, end = _date(start, "start"), _date(end, "end")
    if start > end or type(limit) is not int or not 1 <= limit <= 5000:
        raise ValueError("Require an ordered date window and a limit between 1 and 5000")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if any(
        not isinstance(v, str) or not v.strip()
        for v in (symbol, entity_id, instrument_id, snapshot_id)
    ):
        raise ValueError("Symbol, identities and snapshot_id must be explicit")
    if not isinstance(field_maps, Mapping) or set(field_maps) != set(ACTION_TYPES):
        raise ValueError("field_maps must separately declare split and dividend mappings")
    for kind in ACTION_TYPES:
        adapt_lse_actions(
            [],
            action_type=kind,
            entity_id=entity_id,
            instrument_id=instrument_id,
            received_at=datetime.now(UTC),
            field_map=field_maps[kind],
            ratio_convention=ratio_convention,
            dividend_basis=dividend_basis,
            expected_symbol=symbol,
        )
    request = dict(
        symbol=symbol,
        entity_id=entity_id,
        instrument_id=instrument_id,
        start=start,
        end=end,
        snapshot_id=snapshot_id,
        limit=limit,
        max_bytes=max_bytes,
        field_maps=dict(field_maps),
        ratio_convention=ratio_convention,
        dividend_basis=dividend_basis,
    )
    if source_metadata is None:
        source_metadata = default_source_registry().get("lse").to_dict()
    collection_id = store.begin(
        "lse", "corporate_actions", request, source_metadata=source_metadata
    )
    store.recover(collection_id)
    manifest = store.manifest(collection_id)
    saved = {page["page_id"] for page in manifest["pages"]}
    frames, pages, used_bytes = [], {}, 0
    for kind, endpoint in LSE_ACTION_ENDPOINTS.items():
        if kind in saved:
            page = json.loads(store.read_page(collection_id, kind))
        else:
            if transport is None:
                raise DiscoveryError(
                    "Uncached action page requires an explicit transport; online: set LSE_API_KEY"
                )
            remaining = max_bytes - used_bytes
            if remaining < 1:
                raise DiscoveryLimitError("Corporate actions exceed max_bytes")
            response = transport.get(
                endpoint,
                dict(symbol=symbol, start=start, end=end, order="asc", limit=limit),
                max_bytes=remaining,
            )
            receipt = datetime.now(UTC).isoformat()
            page = dict(
                response=response.payload,
                response_bytes=response.byte_count,
                received_at=receipt,
                endpoint=endpoint,
                capped=len(response.payload) >= limit,
            )
            store.write_page(
                collection_id,
                kind,
                page,
                row_count=len(response.payload),
                metadata={"completion_scope": "symbol_date_window", "capped": page["capped"]},
            )
        if (
            not isinstance(page, dict)
            or page.get("endpoint") != endpoint
            or not isinstance(page.get("response"), list)
            or any(not isinstance(row, dict) for row in page["response"])
            or len(page["response"]) > limit
            or type(page.get("response_bytes")) is not int
            or page["response_bytes"] < 1
            or type(page.get("capped")) is not bool
            or page["capped"] != (len(page["response"]) >= limit)
        ):
            raise DiscoveryError("Invalid saved LSE corporate action page")
        used_bytes += page["response_bytes"]
        if used_bytes > max_bytes:
            raise DiscoveryLimitError("Corporate actions exceed max_bytes")
        frames.append(
            adapt_lse_actions(
                page["response"],
                action_type=kind,
                entity_id=entity_id,
                instrument_id=instrument_id,
                received_at=page["received_at"],
                field_map=field_maps[kind],
                ratio_convention=ratio_convention,
                dividend_basis=dividend_basis,
                expected_symbol=symbol,
            )
        )
        if not frames[-1].effective_date.between(start, end).all():
            raise ValueError("Provider corporate action is outside the requested date window")
        pages[kind] = page
    out = normalize_corporate_actions(pd.concat(frames, ignore_index=True))
    summary = dict(
        collection_id=collection_id,
        symbol=symbol,
        start=start,
        end=end,
        rows=len(out),
        request_completed=True,
        local_normalization_complete=True,
        row_cap_reached=any(page["capped"] for page in pages.values()),
        provider_completeness="unqualified",
        window_complete=False,
        absence_scope="unqualified",
        historical_publication_qualified=False,
        documentation=LSE_ACTION_DOCUMENTATION,
    )
    if "summary" not in saved:
        store.write_page(
            collection_id,
            "summary",
            summary,
            complete=not summary["row_cap_reached"],
            row_count=0,
            metadata={
                "completion_scope": "local_normalization_only",
                "incomplete": summary["row_cap_reached"],
            },
        )
    return collection_id, out, summary
