"""Reference prices/factors with exact identities, quotation units and availability."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from pfe.data.contracts import (
    OBSERVATION_COLUMNS,
    _identifiers,
    _timestamps,
    normalize_observations,
)

ASSET_CLASSES = (
    "equity",
    "benchmark",
    "sector_etf",
    "fx",
    "volatility",
    "commodity",
    "energy",
    "foreign_equity",
    "other",
)
REFERENCE_COLUMNS = (
    *OBSERVATION_COLUMNS,
    "instrument_id",
    "symbol",
    "asset_class",
    "currency",
    "calendar",
    "frequency",
    "price_adjustment",
)


def normalize_reference_assets(frame):
    """Preserve canonical vintages and explicitly declared reference conventions.

    Values are finite numbers or missing, not structured values. Metadata must
    identify an exact instrument/symbol, economic frequency, calendar and unit.
    Currency is nullable for non-currency indices. For FX, base_currency and
    quote_currency are mandatory and unit must be quote/base; value means units
    of quote currency per one base currency. No ticker aliases, exchange mapping,
    monetary conversion, corporate adjustment or holiday filling is inferred.

    Optional observed_at is an aware economic market close, not publication.
    Date-only economic labels retain their precision. Without observed_at,
    time-horizon benchmark correlations cannot be qualified. Rows with explicit
    quality.feature_eligible=False remain evidence but cannot produce features.
    A reference ID has one source/metric/convention per feature invocation;
    reconcile alternative sources and scales explicitly upstream.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Reference assets require a DataFrame")
    missing = set(REFERENCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Reference assets require columns {sorted(missing)}")
    out = normalize_observations(frame)
    _identifiers(
        out,
        ("instrument_id", "symbol", "calendar", "frequency", "currency"),
        nullable=("currency",),
    )
    if not out.asset_class.isin(ASSET_CLASSES).all():
        raise ValueError(f"asset_class must be one of {ASSET_CLASSES}")
    if not out.price_adjustment.isin(("raw", "split_adjusted", "total_return", "unknown")).all():
        raise ValueError("price_adjustment must be raw/split_adjusted/total_return/unknown")
    if out.value.map(lambda value: value is not None and not isinstance(value, (int, float))).any():
        raise ValueError("Reference values must be numeric or missing")
    for name in ("base_currency", "quote_currency"):
        if name not in out:
            out[name] = None
        _identifiers(out, (name,), nullable=(name,))
    fx = out.asset_class.eq("fx")
    if out.loc[fx, ["base_currency", "quote_currency"]].isna().any().any():
        raise ValueError("FX requires explicit base_currency and quote_currency")
    if (out.loc[fx, "base_currency"] == out.loc[fx, "quote_currency"]).any():
        raise ValueError("FX base and quote currencies must differ")
    expected_unit = out.base_currency.copy()
    expected_unit.loc[fx] = out.loc[fx, "quote_currency"] + "/" + out.loc[fx, "base_currency"]
    if not out.loc[fx, "unit"].eq(expected_unit.loc[fx]).all():
        raise ValueError("FX unit must explicitly be quote_currency/base_currency")
    if not out.loc[fx, "currency"].eq(out.loc[fx, "quote_currency"]).all():
        raise ValueError("FX currency must equal its quoted monetary currency")
    if (out.loc[fx, "value"].dropna().astype(float) <= 0).any():
        raise ValueError("FX quoted values must be positive when known")
    for row in out.itertuples(index=False):
        monetary = re.match(r"^([A-Z]{3})(?:/|$)", row.unit)
        if monetary and monetary.group(1) != row.currency:
            raise ValueError("Monetary unit differs from declared reference currency")
    if "observed_at" not in out:
        out["observed_at"] = pd.NaT
    out["observed_at"] = _timestamps(out.observed_at, "observed_at", nullable=True)
    if (out.available_at < out.observed_at).any():
        raise ValueError("Reference availability cannot precede observed market close")
    if "quality.feature_eligible" not in out:
        out["quality.feature_eligible"] = True
    if not out["quality.feature_eligible"].map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("quality.feature_eligible requires booleans")
    metadata = (
        "entity_id",
        "source_id",
        "metric",
        "symbol",
        "asset_class",
        "currency",
        "calendar",
        "frequency",
        "unit",
        "price_adjustment",
        "base_currency",
        "quote_currency",
    )
    for _, group in out.groupby("instrument_id", sort=False):
        if any(group[name].nunique(dropna=False) > 1 for name in metadata):
            raise ValueError(
                "Reconcile reference identity/source/unit/conventions before combining"
            )
        if group.groupby("observed_at").observation_date.nunique().gt(1).any():
            raise ValueError("A reference market close cannot label different economic periods")
    return out


def reference_assets_from_daily(
    frame,
    *,
    asset_class,
    unit,
    symbol=None,
    entity_id=None,
    base_currency=None,
    quote_currency=None,
):
    """Adapt canonical daily rows while preserving actual close and source receipts.

    Missing entity IDs require an explicit factor/entity identifier. No current
    ticker is promoted into a historical issuer/security identity. Canonical
    daily snapshots must be selected before adaptation. Missing prices retain
    their masks. Source price-adjustment metadata is carried unchanged.
    """
    required = {
        "instrument_id",
        "entity_id",
        "session_date",
        "session_close",
        "available_at",
        "close",
        "source_id",
        "revision_id",
        "schema_version",
        "currency",
        "price_adjustment",
    }
    if not required.issubset(frame):
        raise ValueError("Expected canonical daily bars with identities and provenance")
    out = frame.copy()
    if entity_id is not None:
        out["entity_id"] = out.entity_id.where(out.entity_id.notna(), entity_id)
    out["symbol"] = symbol if symbol is not None else out.get("symbol", out.get("ticker"))
    if out.symbol.isna().any():
        raise ValueError("Reference symbol must be supplied explicitly")
    out["value"], out["observation_date"], out["observed_at"] = (
        out.close,
        out.session_date,
        out.session_close,
    )
    out["metric"], out["asset_class"], out["unit"], out["frequency"] = (
        "close",
        asset_class,
        unit,
        "daily",
    )
    out["calendar"] = frame.attrs.get("calendar", "unknown")
    out["base_currency"], out["quote_currency"] = base_currency, quote_currency
    if "published_at" not in out:
        out["published_at"] = pd.NaT
    return normalize_reference_assets(out)


def reference_assets_from_observations(
    frame,
    *,
    instrument_id,
    symbol,
    asset_class,
    currency=None,
    calendar=None,
    frequency=None,
    price_adjustment="unknown",
    base_currency=None,
    quote_currency=None,
):
    """Adapt one selected canonical factor without changing its units or timing.

    Select one entity/source/metric upstream. Identity and market conventions
    are caller declarations, not ticker aliases inferred from provider labels.
    Existing currency/calendar/frequency metadata cannot be silently overridden.
    With no observed_at market close, levels/returns remain usable but benchmark
    horizon comparisons remain unqualified. Monthly commodity averages retain
    their monthly frequency; they do not become daily market closing prices.
    """
    out = normalize_observations(frame)
    out["instrument_id"], out["symbol"], out["asset_class"] = (
        instrument_id,
        symbol,
        asset_class,
    )
    for name, supplied in (
        ("currency", currency),
        ("calendar", calendar),
        ("frequency", frequency),
    ):
        if name in out:
            if supplied is not None and out[name].dropna().ne(supplied).any():
                raise ValueError(f"Existing {name} differs; reconcile conventions explicitly")
            if supplied is not None:
                out[name] = out[name].fillna(supplied)
        else:
            out[name] = supplied
    out["price_adjustment"] = price_adjustment
    out["base_currency"], out["quote_currency"] = base_currency, quote_currency
    return normalize_reference_assets(out)


def load_reference_assets(path):
    """Load canonical CSV/Parquet with exact identifiers and no calendar inference."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        identifiers = (
            "entity_id",
            "instrument_id",
            "source_id",
            "revision_id",
            "schema_version",
            "symbol",
            "metric",
            "unit",
            "currency",
            "base_currency",
            "quote_currency",
        )
        frame = pd.read_csv(path, dtype=dict.fromkeys(identifiers, str))
        for name in ("point_in_time", "quality.feature_eligible"):
            if name in frame:
                frame[name] = frame[name].map(
                    lambda value: (
                        {"true": True, "false": False}.get(value.lower(), value)
                        if isinstance(value, str)
                        else value
                    )
                )
    else:
        raise ValueError("Expected reference .csv or .parquet file")
    return normalize_reference_assets(frame)
