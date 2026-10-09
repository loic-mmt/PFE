"""Selection must preserve formulas, skip unrelated work and stream exact state."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pfe import (
    FEATURE_GROUPS,
    aggregate_selected_features,
    build_features,
    compute_feature,
    compute_features,
    compute_selected_features,
    enrich_bars,
    iter_features,
)


@pytest.fixture
def bars():
    n = 90
    close = 100 + np.sin(np.arange(n) / 4) + np.arange(n) * 0.005
    opening = np.r_[100.0, close[:-1]]
    return pd.DataFrame(
        dict(
            open=opening,
            high=np.maximum(opening, close) + 0.2,
            low=np.minimum(opening, close) - 0.2,
            close=close,
            volume=(100 + np.arange(n) % 17).astype(float),
        ),
        index=pd.date_range("2026-01-05 14:30", periods=n, freq="min", tz="UTC", name="timestamp"),
    )


@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_every_source_feature_independently_matches_group(bars, group):
    before = bars.copy(deep=True)
    expected = compute_features(bars, windows=(5,), groups=(group,))
    combined = compute_selected_features(bars, tuple(expected.columns))
    assert_frame_equal(expected, combined, rtol=1e-10, atol=1e-10)
    for name in expected:
        result = compute_feature(bars, name)
        assert_series_equal(expected[name], result, rtol=1e-10, atol=1e-10)
    prefix = compute_selected_features(bars.iloc[:45], tuple(expected.columns))
    altered = bars.copy()
    altered.iloc[45:] *= 2
    future = compute_selected_features(altered, tuple(expected.columns))
    assert_frame_equal(prefix, future.loc[prefix.index], rtol=1e-10, atol=1e-10)
    assert_frame_equal(before, bars)


@pytest.mark.parametrize("holes", [False, True])
@pytest.mark.parametrize("frequency", ["7min", "30min"])
def test_every_intrabar_feature_matches_group(bars, holes, frequency):
    if holes:
        bars = bars.drop(bars.index[[8, 31, 59]])
    expected = enrich_bars(bars, frequency, segments=3, bins=4)
    names = tuple(name for name in expected if name != "available_at")
    result = aggregate_selected_features(bars, names, frequency, segments=3, bins=4)
    assert_frame_equal(expected[result.columns], result, rtol=1e-10, atol=1e-10)
    for name in names:
        single = aggregate_selected_features(bars, name, frequency, segments=3, bins=4)
        assert_series_equal(expected[name], single[name], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("policy", ["reset", "allow"])
@pytest.mark.parametrize("batch_size", [1, 7, 19, 100])
def test_streaming_is_exact_across_gaps_and_warmup(bars, policy, batch_size):
    bars = bars.drop(bars.index[[7, 33, 70]])
    names = (
        "returns.log",
        "statistics.close_mean_w5",
        "technical.rsi_w5",
        "technical.ema_w5",
        "technical.atr_w5",
        "technical.wma_w5",
    )
    full = compute_selected_features(bars, names, gap_policy=policy)
    batches = (bars.iloc[i : i + batch_size] for i in range(0, len(bars), batch_size))
    result = pd.concat(iter_features(batches, names, gap_policy=policy))
    assert_frame_equal(full, result, check_freq=False, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("frequency", ["7min", "30min"])
@pytest.mark.parametrize("batch_size", [1, 13, 100])
def test_streaming_target_keeps_provisional_bin_and_state(bars, frequency, batch_size):
    bars = bars.drop(bars.index[[7, 33, 70]])
    names = ("path.realized_variance", "profile.tpo_poc", "technical.rsi_w5", "returns.log")
    full = compute_selected_features(bars, names, frequency=frequency)
    batches = (bars.iloc[i : i + batch_size] for i in range(0, len(bars), batch_size))
    result = pd.concat(iter_features(batches, names, frequency=frequency))
    assert_frame_equal(full, result, check_freq=False, rtol=1e-10, atol=1e-10)


def test_selection_skips_whole_groups_and_unrelated_rolling(bars, monkeypatch):
    from pandas.core.window.rolling import Rolling

    import pfe.features.api as api
    import pfe.features.targeted as targeted

    def forbidden(*args, **kwargs):
        raise AssertionError("Unrelated computation executed")

    monkeypatch.setattr(api, "_calculate", forbidden)
    monkeypatch.setattr(targeted, "intrabar_features", forbidden)
    monkeypatch.setattr(Rolling, "apply", forbidden)
    compute_feature(bars, "technical.rsi", window=5)
    compute_feature(bars, "statistics.close_mean", window=5)
    compute_feature(bars, "path.realized_variance", frequency="7min")


def test_shared_native_primitive_runs_once(bars, monkeypatch):
    from pfe.features import selected

    selected._native_catalog((5,))  # Populate metadata before counting numeric calls.
    original = selected.abstract.Function
    calls = []

    class Tracked:
        def __init__(self, name):
            self.function = original(name)
            self.name = name

        def __getattr__(self, name):
            return getattr(self.function, name)

        def __call__(self, inputs):
            calls.append(self.name)
            return self.function(inputs)

    monkeypatch.setattr(selected.abstract, "Function", Tracked)
    names = ("technical.macd_macd_w5", "technical.macd_macdsignal_w5")
    compute_selected_features(bars, names)
    assert calls == ["MACD"]


def test_additive_api_and_window_shortcut(bars):
    result = compute_features(bars, names=("technical.rsi_w5",))
    assert_series_equal(result.iloc[:, 0], compute_feature(bars, "technical.rsi", window=5))
    result = build_features(bars, "7min", names=("path.realized_variance",))
    expected = compute_selected_features(bars, "path.realized_variance", frequency="7min")
    assert_frame_equal(result, expected)
    with pytest.raises(ValueError, match="groups"):
        compute_features(bars, groups=("technical",), names=("technical.rsi_w5",))
    with pytest.raises(ValueError, match="window"):
        compute_feature(bars, "technical.rsi_w5", window=5)


@pytest.mark.parametrize("names", [(), ("unknown.x",), ("returns.log", "returns.log")])
def test_bad_selection_rejected(bars, names):
    with pytest.raises(ValueError):
        compute_selected_features(bars, names)


def test_streaming_rejects_unimplemented_state_and_overlapping_batches(bars):
    with pytest.raises(ValueError, match="Streaming state"):
        list(iter_features([bars], "technical.macd_macd_w5"))
    with pytest.raises(ValueError, match="sorted"):
        list(iter_features([bars.iloc[:20], bars.iloc[19:]], "technical.rsi_w5"))


def test_flat_zero_volume_and_data_context_not_cached(bars):
    flat = bars.copy()
    flat.loc[:, ["open", "high", "low", "close"]] = 100.0
    flat.volume = 0.0
    names = ("returns.log", "statistics.close_mean_w5", "technical.rsi_w5")
    first = compute_selected_features(bars, names)
    second = compute_selected_features(flat, names)
    third = compute_selected_features(bars, names)
    assert_frame_equal(first, third)
    assert not np.array_equal(first.to_numpy(), second.to_numpy(), equal_nan=True)
    streamed = pd.concat(
        iter_features((flat.iloc[i : i + 3] for i in range(0, len(flat), 3)), names)
    )
    assert_frame_equal(second, streamed, check_freq=False)


@pytest.mark.parametrize("index_name", ["timestamp", "custom_time", None])
def test_parquet_batch_loader_and_streaming_cli(bars, tmp_path, index_name):
    import subprocess
    import sys
    from pathlib import Path

    from pfe import iter_bars

    bars.index = bars.index.rename(index_name)
    source, output = tmp_path / "input.parquet", tmp_path / "output.parquet"
    bars.assign(unneeded="large unused field").to_parquet(source, row_group_size=13)
    batches = list(iter_bars(source, batch_size=11))
    expected_bars = bars.copy()
    expected_bars.index = expected_bars.index.rename("timestamp")
    assert_frame_equal(pd.concat(batches), expected_bars, check_freq=False)
    names = ("path.realized_variance", "technical.rsi_w5")
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/build_features.py"),
        "--input",
        str(source),
        "--output",
        str(output),
        "--frequency",
        "7min",
        "--features",
        *names,
        "--batch-size",
        "11",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    expected = compute_selected_features(expected_bars, names, frequency="7min")
    assert_frame_equal(pd.read_parquet(output), expected, check_freq=False)
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "Output already exists" in failed.stderr


def test_empty_target_still_validates_standard_names(bars):
    with pytest.raises(ValueError, match="Unknown feature"):
        compute_selected_features(bars.iloc[:1], "technical.no_such_feature", frequency="1h")


def test_streaming_hull_average_has_nested_lookbacks(bars):
    names = ("technical.hma_w5", "technical.coppock_w5", "trend.sma_spread_w5_w20")
    full = compute_selected_features(bars, names)
    result = pd.concat(iter_features((bars.iloc[i : i + 3] for i in range(0, len(bars), 3)), names))
    assert_frame_equal(full, result, check_freq=False, atol=1e-10, rtol=1e-10)
