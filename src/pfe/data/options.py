"""Canonical option chain snapshots, prints and completed contract candles."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from pfe.data.macro import _numbers, _quality, _timestamps

IDENTITY = ("underlying", "ticker", "contract_type", "strike", "expiry")
GREEKS = ("delta", "gamma", "theta", "vega", "rho")
COMMON_NUMERIC = ("underlying_price", "iv", *GREEKS)
CHAIN_NUMERIC = ("last_price", "volume", "premium", "open_interest", "bid", "ask")


def _option_numbers(values, name, *, nullable=False):
    if is_numeric_dtype(values) and not is_bool_dtype(values) and not is_complex_dtype(values):
        result = values.astype(float)
        if np.isinf(result).any() or (not nullable and result.isna().any()):
            raise ValueError(f"{name} must contain finite numbers")
        return result
    return _numbers(values, name, nullable=nullable)


def _option_times(values, name):
    if isinstance(values.dtype, pd.DatetimeTZDtype) and not values.isna().any():
        return values.dt.tz_convert("UTC")
    return _timestamps(values, name)


def _base(frame, required):
    if not isinstance(frame, pd.DataFrame) or not set(required).issubset(frame.columns):
        raise ValueError(f"Option data require {sorted(required)}")
    if not frame.columns.is_unique:
        raise ValueError("Option columns must be unique")
    out = frame.copy()
    for name in ("underlying", "ticker"):
        if not out[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{name} must contain non-empty strings")
        out[name] = out[name].str.strip().str.upper()
    if not out.contract_type.isin(("call", "put")).all():
        raise ValueError("contract_type must be call or put")
    out["strike"] = _option_numbers(out.strike, "strike")
    if (out.strike <= 0).any():
        raise ValueError("strike must be positive")
    expiry = {}
    for value in out.expiry.unique():
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp != stamp.normalize():
            raise ValueError("expiry must contain non-missing calendar dates")
        # Expiry is a calendar date, not a known intraday exercise/settlement timestamp.
        expiry[value] = pd.Timestamp(stamp.date(), tz="UTC")
    out["expiry"] = pd.to_datetime(out.expiry.map(expiry), utc=True)
    _quality(out)
    return out


def _numeric(out, names, required=()):
    for name in names:
        if name not in out:
            out[name] = np.nan
        out[name] = _option_numbers(out[name], name, nullable=name not in required)
        if name in ("strike", "underlying_price") and (out[name].dropna() <= 0).any():
            raise ValueError(f"{name} must be positive when present")
        if name in (
            "iv",
            "gamma",
            "vega",
            "volume",
            "premium",
            "open_interest",
            "last_price",
            "bid",
            "ask",
            "open",
            "high",
            "low",
            "close",
            "print_count",
        ):
            if (out[name].dropna() < 0).any():
                raise ValueError(f"{name} must be non-negative")
    if "delta" in out and (out.delta.dropna().abs() > 1).any():
        raise ValueError("delta must lie in [-1, 1]")


def _times(out, observed):
    for name in (observed, "available_at"):
        out[name] = _option_times(out[name], name)
    if (out.available_at < out[observed]).any():
        raise ValueError(f"available_at cannot precede {observed}")


def _contracts(out):
    if len(out) and (out.groupby("ticker")[list(IDENTITY)].nunique() > 1).any().any():
        raise ValueError("A ticker must identify a single option contract")


def normalize_option_chain(frame):
    """Validate latest-contract rows grouped into explicit, atomic snapshots.

    Required: underlying, ticker, contract_type, strike, expiry, snapshot_at,
    available_at. Optional prices, volume, premium, IV/greeks, OI, bid/ask remain
    NaN when missing. Each (underlying, snapshot_at) has one availability timestamp.
    quote_at is the latest TRADE timestamp, not evidence of a contemporaneous quote.
    """
    out = _base(frame, (*IDENTITY, "snapshot_at", "available_at"))
    _times(out, "snapshot_at")
    _numeric(out, (*COMMON_NUMERIC, *CHAIN_NUMERIC))
    if "quote_at" not in out:
        out["quote_at"] = pd.NaT
    present = out.quote_at.notna()
    stamps = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    stamps.loc[present] = _option_times(out.loc[present, "quote_at"], "quote_at")
    out["quote_at"] = stamps
    if (out.loc[present, "quote_at"] > out.loc[present, "snapshot_at"]).any():
        raise ValueError("quote_at cannot follow snapshot_at")
    if (out.bid.notna() & out.ask.notna() & (out.bid > out.ask)).any():
        raise ValueError("Crossed bid/ask")
    if out.duplicated(["underlying", "snapshot_at", "ticker"]).any():
        raise ValueError("Duplicate contract in snapshot")
    if (out.groupby(["underlying", "snapshot_at"]).available_at.nunique() > 1).any():
        raise ValueError("Chain snapshot availability must be atomic")
    _contracts(out)
    return out.sort_values(
        ["underlying", "available_at", "snapshot_at", "ticker"], kind="stable", ignore_index=True
    )


def normalize_option_flow(frame):
    """Validate distinct trade prints, keeping event time separate from availability."""
    out = _base(frame, (*IDENTITY, "trade_id", "trade_at", "available_at", "volume", "premium"))
    if out.trade_id.isna().any() or out.duplicated(["underlying", "trade_id"]).any():
        raise ValueError("trade_id must be non-missing and unique per underlying")
    _times(out, "trade_at")
    _numeric(out, (*COMMON_NUMERIC, "last_price", "volume", "premium"), ("volume", "premium"))
    _contracts(out)
    return out.sort_values(["available_at", "trade_at"], kind="stable", ignore_index=True)


def normalize_option_candles(frame, *, source_frequency="1min"):
    """Validate per-contract OHLC; rows become usable no earlier than their close."""
    out = _base(
        frame,
        (
            *IDENTITY,
            "bar_at",
            "available_at",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "premium",
            "print_count",
        ),
    )
    step = pd.Timedelta(source_frequency)
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("source_frequency must be a positive fixed duration")
    _times(out, "bar_at")
    _numeric(
        out,
        (*COMMON_NUMERIC, "open", "high", "low", "close", "volume", "premium", "print_count"),
        ("open", "high", "low", "close", "volume", "premium", "print_count"),
    )
    if (out.available_at < out.bar_at + step).any():
        raise ValueError("Option candles are unavailable before bar close")
    if (
        (out.high < out[["open", "close", "low"]].max(axis=1))
        | (out.low > out[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("Invalid option OHLC ordering")
    if out.duplicated(["ticker", "bar_at"]).any():
        raise ValueError("Duplicate option candle")
    _contracts(out)
    out = out.sort_values(["available_at", "bar_at"], kind="stable", ignore_index=True)
    out.attrs["source_frequency"] = str(step)
    return out


@dataclass(frozen=True)
class OptionData:
    chain: pd.DataFrame
    flow: pd.DataFrame
    candles: pd.DataFrame
    snapshot_at: pd.Timestamp
    limit_reached: dict


def load_options(path, *, max_bytes=1048576, availability="receipt", latency="0s"):
    """Load a bounded LSE sample; never interpret the current chain as past history.

    ``receipt`` uses response receipt for ALL fields (default). ``event`` explicitly
    assumes prints usable at trade_at + latency and candles at close + latency;
    historical delivery/revision times are unverified, so provenance remains False.
    Chain snapshots ALWAYS become available at receipt. Units are preserved from
    LSE: IV is decimal annualized volatility, greeks use the provider's model units.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if availability not in ("receipt", "event"):
        raise ValueError("availability must be receipt or event")
    delay = pd.Timedelta(latency)
    if pd.isna(delay) or delay < pd.Timedelta(0):
        raise ValueError("latency must be a non-negative duration")
    with Path(path).open("rb") as file:
        raw = file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Option file exceeds max_bytes")
    return option_data_from_payload(json.loads(raw), availability=availability, latency=latency)


def option_data_from_payload(payload, *, availability="receipt", latency="0s"):
    """Normalize an already bounded LSE payload without a temporary source file."""
    if availability not in ("receipt", "event"):
        raise ValueError("availability must be receipt or event")
    delay = pd.Timedelta(latency)
    if pd.isna(delay) or delay < pd.Timedelta(0):
        raise ValueError("latency must be a non-negative duration")
    if payload.get("provider") != "LSE":
        raise ValueError("Expected an LSE options sample")
    snapshot = _timestamps([payload["snapshot_at"]], "snapshot_at")[0]
    frames = {}
    for kind in ("chain", "flow", "candles"):
        rows = []
        for row in payload.get(kind, []):
            item = {key: row.get(key) for key in IDENTITY}
            item.update(
                {key: row.get(f"{key}_avg" if kind == "candles" else key) for key in COMMON_NUMERIC}
            )
            # underlying_price is not averaged by the endpoint schema.
            item["underlying_price"] = row.get("underlying_price")
            item["point_in_time"] = False
            if kind == "chain":
                item.update({key: row.get(key) for key in CHAIN_NUMERIC})
                item.update(
                    snapshot_at=snapshot,
                    available_at=snapshot,
                    quote_at=row.get("last_trade_at"),
                    volume=row.get("volume_today"),
                    premium=row.get("premium_today"),
                )
            else:
                source_time = _timestamps([row["ts" if kind == "flow" else "minute"]], kind)[0]
                earliest = source_time + (pd.Timedelta("1min") if kind == "candles" else delay)
                if kind == "candles":
                    earliest += delay
                item["available_at"] = (
                    max(snapshot, earliest) if availability == "receipt" else earliest
                )
                item.update(volume=row.get("volume"), premium=row.get("premium"))
                if kind == "flow":
                    item.update(
                        trade_id=row["id"], trade_at=source_time, last_price=row.get("last_price")
                    )
                else:
                    item.update(
                        bar_at=source_time,
                        print_count=row.get("print_count"),
                        **{key: row.get(key) for key in ("open", "high", "low", "close")},
                    )
            rows.append(item)
        required = {
            "chain": (*IDENTITY, "snapshot_at", "available_at"),
            "flow": (*IDENTITY, "trade_id", "trade_at", "available_at", "volume", "premium"),
            "candles": (
                *IDENTITY,
                "bar_at",
                "available_at",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "premium",
                "print_count",
            ),
        }[kind]
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=required)
        frames[kind] = {
            "chain": normalize_option_chain,
            "flow": normalize_option_flow,
            "candles": normalize_option_candles,
        }[kind](frame)
    flags = payload.get(
        "limit_reached", {key: len(payload.get(key, [])) == payload.get("limit") for key in frames}
    )
    for key, frame in frames.items():
        frame.attrs["limit_reached"] = bool(flags.get(key, False))
    return OptionData(**frames, snapshot_at=snapshot, limit_reached=flags)
