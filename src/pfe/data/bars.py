"""The canonical single-instrument OHLCV contract."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

OHLCV = ("open", "high", "low", "close", "volume")


def normalize_bars(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    source_timezone: str | None = None,
    frequency: str | pd.Timedelta = "1min",
) -> pd.DataFrame:
    """Return sorted OHLCV with a unique, frequency-aligned UTC DatetimeIndex.

    Accept a timestamp column or a DatetimeIndex. Naive timestamps require an
    explicit source timezone. Numeric epochs are rejected: convert their units
    explicitly before calling. Gaps are preserved; no resampling or filling.
    Input data is never mutated. Additional columns are discarded. The UTC index
    uses nanoseconds so downstream integer time arithmetic has a fixed unit.
    """
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    missing = set(OHLCV) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No bars supplied")

    if timestamp_column in frame.columns:
        times = frame[timestamp_column]
        if is_numeric_dtype(times):
            raise ValueError("Numeric timestamps require an explicit epoch unit conversion")
        index = pd.DatetimeIndex(pd.to_datetime(times, errors="raise", format="mixed"))
    elif isinstance(frame.index, pd.DatetimeIndex):
        index = frame.index.copy()
    else:
        raise ValueError(f"Expected '{timestamp_column}' column or a DatetimeIndex")
    if index.hasnans:
        raise ValueError("Timestamps must not be missing")
    if index.tz is None:
        if source_timezone is None:
            raise ValueError("Naive timestamps require source_timezone")
        index = index.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
    index = index.tz_convert("UTC").as_unit("ns").rename("timestamp")
    if index.has_duplicates:
        raise ValueError("Duplicate bar timestamps")
    step = pd.Timedelta(frequency)
    if step <= pd.Timedelta(0):
        raise ValueError("frequency must be a positive fixed duration")
    if not index.equals(index.floor(step)):
        label = "whole minutes" if step == pd.Timedelta("1min") else str(step)
        raise ValueError(f"Bars must be aligned to {label}")

    bars = frame.loc[:, list(OHLCV)].copy()
    bars.index = index
    bars = bars.apply(pd.to_numeric, errors="raise").astype("float64")
    if not np.isfinite(bars.to_numpy()).all():
        raise ValueError("OHLCV values must be finite and non-missing")
    if (bars.loc[:, ["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    if (bars["volume"] < 0).any():
        raise ValueError("Volume must be non-negative")
    if (bars["high"] < bars[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("High must be at least open, low and close")
    if (bars["low"] > bars[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Low must be at most open, high and close")
    return bars.sort_index()


def load_bars(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    source_timezone: str | None = None,
    frequency: str | pd.Timedelta = "1min",
) -> pd.DataFrame:
    """Load a local CSV or Parquet file and validate the canonical bar contract."""
    path = Path(path)
    match path.suffix.lower():
        case ".csv":
            frame = pd.read_csv(path)
        case ".parquet":
            frame = pd.read_parquet(path)
        case _:
            raise ValueError("Expected a .csv or .parquet file")
    return normalize_bars(
        frame,
        timestamp_column=timestamp_column,
        source_timezone=source_timezone,
        frequency=frequency,
    )
