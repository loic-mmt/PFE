"""Verify timestamp semantics, data integrity, and local file round trips."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe import load_bars, normalize_bars


def test_normalization_sorts_without_mutation(raw_bars):
    raw_bars["extra"] = "provider metadata"
    before = raw_bars.copy(deep=True)
    bars = normalize_bars(raw_bars.iloc[::-1])
    assert_frame_equal(raw_bars, before)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert str(bars.index.tz) == "UTC"
    assert bars.index.name == "timestamp"
    assert bars.index.is_monotonic_increasing
    assert bars.iloc[0]["close"] == 101.0
    assert all(dtype == "float64" for dtype in bars.dtypes)


def test_naive_timestamps_require_explicit_timezone(raw_bars):
    raw_bars["timestamp"] = ["2026-01-05 09:30:00", "2026-01-05 09:31:00"]
    with pytest.raises(ValueError, match="source_timezone"):
        normalize_bars(raw_bars)
    bars = normalize_bars(raw_bars, source_timezone="America/New_York")
    assert bars.index[0] == pd.Timestamp("2026-01-05 14:30:00", tz="UTC")


def test_aware_timestamps_convert_to_utc(raw_bars):
    raw_bars["timestamp"] = pd.date_range(
        "2026-07-06 09:30", periods=2, freq="min", tz="America/New_York"
    )
    assert normalize_bars(raw_bars).index[0] == pd.Timestamp("2026-07-06 13:30", tz="UTC")


def test_gaps_are_preserved(raw_bars):
    raw_bars.loc[1, "timestamp"] = "2026-01-06T14:30:00Z"
    assert len(normalize_bars(raw_bars)) == 2


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("timestamp", "2026-01-05T14:30:00Z", "Duplicate"),
        ("timestamp", "2026-01-05T14:31:01Z", "whole minutes"),
        ("timestamp", None, "must not be missing"),
        ("close", float("nan"), "finite"),
        ("close", float("inf"), "finite"),
        ("close", 0.0, "positive"),
        ("volume", -1.0, "non-negative"),
        ("high", 100.0, "High"),
        ("low", 102.5, "Low"),
    ],
)
def test_invalid_bars_raise(raw_bars, column, value, message):
    raw_bars.loc[1, column] = value
    with pytest.raises(ValueError, match=message):
        normalize_bars(raw_bars)


def test_missing_columns_raise(raw_bars):
    with pytest.raises(ValueError, match="Missing OHLCV"):
        normalize_bars(raw_bars.drop(columns="volume"))


def test_empty_data_raises(raw_bars):
    with pytest.raises(ValueError, match="No bars"):
        normalize_bars(raw_bars.iloc[:0])


def test_numeric_epochs_are_not_guessed(raw_bars):
    raw_bars["timestamp"] = [1767623400000, 1767623460000]
    with pytest.raises(ValueError, match="epoch unit"):
        normalize_bars(raw_bars)


def test_vault_timestamp_column(raw_bars):
    raw_bars = raw_bars.rename(columns={"timestamp": "ts"})
    assert len(normalize_bars(raw_bars, timestamp_column="ts")) == 2


def test_normalization_is_idempotent(raw_bars):
    bars = normalize_bars(raw_bars)
    assert_frame_equal(normalize_bars(bars), bars)


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_local_file_round_trip(raw_bars, tmp_path, suffix):
    expected = normalize_bars(raw_bars)
    path = tmp_path / f"bars.{suffix}"
    if suffix == "csv":
        expected.to_csv(path)
    else:
        expected.to_parquet(path)
    assert_frame_equal(load_bars(path), expected)


def test_unsupported_file_type(tmp_path):
    with pytest.raises(ValueError, match="csv or .parquet"):
        load_bars(tmp_path / "bars.json")
