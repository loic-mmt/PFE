"""Plot content and timestamps, with an offscreen renderer and synthetic data."""

import importlib.util
import io
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pfe import (  # noqa: E402
    build_features,
    compute_features,
    normalize_bars,
    plot_bars,
    plot_feature_correlation,
    plot_feature_distribution,
    plot_features,
    plot_intrabar,
    plot_price_time_profile,
    plot_profile_bars,
    price_time_profile,
)


@pytest.fixture
def bars():
    close = 100 + np.sin(np.arange(42) / 3)
    opening = np.r_[100, close[:-1]]
    return normalize_bars(
        pd.DataFrame(
            {
                "open": opening,
                "high": np.maximum(opening, close) + 0.1,
                "low": np.minimum(opening, close) - 0.1,
                "close": close,
                "volume": np.arange(42) + 100,
            },
            index=pd.date_range("2026-01-05 14:30", periods=42, freq="min", tz="UTC"),
        )
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def render(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=60)
    assert buffer.getvalue().startswith(b"\x89PNG")


def test_candles_aggregate_at_custom_frequency_and_overlay_at_close(bars):
    before = bars.copy(deep=True)
    features = build_features(
        bars, "7min", origin=bars.index[0], windows=(2,), groups=("technical",)
    )
    fig = plot_bars(
        bars,
        frequency="7min",
        origin=bars.index[0],
        overlays=features,
        columns="technical.sma_w2",
        timezone="America/New_York",
    )
    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 6
    first = fig.axes[0].patches[0]
    assert first.get_y() == min(bars.open.iloc[0], bars.close.iloc[6])
    assert first.get_height() == abs(bars.close.iloc[6] - bars.open.iloc[0])
    overlay = fig.axes[0].lines[0]
    assert pd.DatetimeIndex(overlay.get_xdata()).equals(pd.DatetimeIndex(features.available_at))
    assert [p.get_height() for p in fig.axes[1].patches] == [
        bars.volume.iloc[i : i + 7].sum() for i in range(0, 42, 7)
    ]
    render(fig)
    assert_frame_equal(bars, before)


def test_features_use_availability_and_preserve_warmup(bars):
    features = compute_features(bars, windows=(5,), groups=("returns", "technical"))
    before = features.copy(deep=True)
    fig = plot_features(features, ["returns.log", "technical.rsi_w5"])
    assert len(fig.axes) == 2
    assert pd.DatetimeIndex(fig.axes[0].lines[0].get_xdata()).equals(
        bars.index + pd.Timedelta("1min")
    )
    np.testing.assert_allclose(
        fig.axes[1].lines[0].get_ydata(), features["technical.rsi_w5"], equal_nan=True
    )
    render(fig)
    assert_frame_equal(features, before)
    plain = features.copy()
    plain.attrs.clear()
    with pytest.raises(ValueError, match="Availability metadata"):
        plot_features(plain, "returns.log", time="available")
    explicit = plot_features(plain, "returns.log", time="available", source_frequency="1min")
    assert pd.DatetimeIndex(explicit.axes[0].lines[0].get_xdata())[0] == bars.index[1]


def test_intrabar_path_variance_and_missing_slots(bars):
    first = bars.iloc[:7]
    fig = plot_intrabar(first, start=first.index[0], frequency="7min", segments=3)
    returns = np.r_[np.log(first.close.iloc[0] / first.open.iloc[0]), np.diff(np.log(first.close))]
    np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), np.cumsum(returns**2))
    path = fig.axes[0].lines[0]
    assert path.get_ydata()[0] == first.open.iloc[0]
    assert pd.DatetimeIndex(path.get_xdata())[-1] == first.index[0] + pd.Timedelta("7min")
    render(fig)
    missing = first.drop(first.index[2])
    gaps = plot_intrabar(missing, start=first.index[0], frequency="7min")
    assert np.isnan(gaps.axes[0].lines[0].get_ydata()[3])
    assert np.isnan(gaps.axes[2].lines[0].get_ydata()[2:]).all()
    assert "coverage 86%" in gaps._suptitle.get_text()
    render(gaps)


def test_profile_matches_feature_matrix_and_has_letters(bars):
    group = bars.iloc[:6].drop(bars.index[2:4])
    matrix, _, _, _ = price_time_profile(
        group, start=bars.index[0], duration=pd.Timedelta("6min"), bins=4, segments=3
    )
    fig = plot_price_time_profile(
        group, start=bars.index[0], frequency="6min", bins=4, segments=3, mode="timeline"
    )
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), matrix)
    assert {text.get_text() for text in fig.axes[0].texts} == {"A", "C"}
    assert [p.get_width() for p in fig.axes[1].patches] == list(matrix.sum(axis=1))
    render(fig)


def test_compact_profile_packs_each_row_and_preserves_missing_letter(bars):
    group = bars.iloc[:6].drop(bars.index[2:4])
    matrix, _, _, _ = price_time_profile(
        group, start=bars.index[0], duration=pd.Timedelta("6min"), bins=4, segments=3
    )
    fig = plot_price_time_profile(group, start=bars.index[0], frequency="6min", bins=4, segments=3)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    for row in range(4):
        texts = [t for t in ax.texts if t.get_position()[1] == row]
        active = np.flatnonzero(matrix[row])
        assert [t.get_text() for t in texts] == [chr(65 + int(s)) for s in active]
        assert [t.get_position()[0] for t in texts] == [i + 0.5 for i in range(len(active))]
    assert len(ax.patches) == int(matrix.sum())
    assert "B" not in {t.get_text() for t in ax.texts}
    render(fig)


def test_profile_candles_preserve_price_levels_and_target_intervals(bars):
    source = bars.iloc[:14].drop(bars.index[2:4])
    fig = plot_profile_bars(source, frequency="7min", origin=bars.index[0], bins=4, segments=7)
    texts = fig.axes[0].texts
    expected_count = 0
    for start in (bars.index[0], bars.index[7]):
        group = source.loc[(source.index >= start) & (source.index < start + pd.Timedelta("7min"))]
        matrix, _, centers, _ = price_time_profile(
            group, start=start, duration=pd.Timedelta("7min"), bins=4, segments=7
        )
        expected_count += int(matrix.sum())
        import matplotlib.dates as dates

        left = dates.date2num(start.to_pydatetime())
        width = 7 / 1440
        local_texts = [t for t in texts if left <= t.get_position()[0] < left + width]
        assert len(local_texts) == int(matrix.sum())
        for row in range(4):
            row_texts = [t for t in local_texts if t.get_position()[1] == centers[row]]
            assert [t.get_text() for t in row_texts] == [
                chr(65 + int(s)) for s in np.flatnonzero(matrix[row])
            ]
            np.testing.assert_allclose(
                [t.get_position()[0] for t in row_texts],
                [left + width * 0.05 + (i + 0.5) * width * 0.85 / 7 for i in range(len(row_texts))],
            )
    assert len(texts) == expected_count
    render(fig)


def test_flat_zero_volume_and_second_granularity(bars):
    flat = bars.iloc[:4].copy()
    flat.loc[:, ["open", "high", "low", "close"]] = 100
    flat.volume = 0
    flat.index = pd.date_range(bars.index[0], periods=4, freq="s")
    render(plot_bars(flat, source_frequency="1s", volume=False))
    render(plot_intrabar(flat, start=flat.index[0], frequency="4s", source_frequency="1s"))
    render(plot_profile_bars(flat, frequency="4s", source_frequency="1s", origin=flat.index[0]))
    render(
        plot_price_time_profile(
            flat, start=flat.index[0], frequency="4s", source_frequency="1s", bins=4, segments=27
        )
    )


@pytest.mark.parametrize("method", ["spearman", "pearson", "kendall"])
def test_distributions_and_pairwise_correlation(bars, method):
    frame = pd.DataFrame(
        {"a": [1, 2, 3, np.nan, np.inf], "b": [3, 2, 1, 8, 9], "constant": [1] * 5},
        index=bars.index[:5],
    )
    original = frame.copy(deep=True)
    hist = plot_feature_distribution(frame, ["a", "constant"], bins=3)
    assert sum(p.get_height() for p in hist.axes[0].patches) == 3
    assert "omitted=2" in hist.axes[0].get_title()
    correlation = plot_feature_correlation(frame, ["a", "b", "constant"], method=method)
    actual = correlation.axes[0].images[0].get_array()
    assert actual[0, 1] == pytest.approx(-1)
    assert np.ma.getmaskarray(actual)[2].all()
    render(hist)
    render(correlation)
    assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "call",
    [
        lambda b: plot_bars(b, frequency="90s"),
        lambda b: plot_bars(b.iloc[:2], frequency="7min", origin=b.index[0]),
        lambda b: plot_features(b, ["missing"]),
        lambda b: plot_features(b, []),
        lambda b: plot_features(b.tz_localize(None), "close"),
        lambda b: plot_features(b, "close", time="future"),
        lambda b: plot_intrabar(b, start="2026-01-05", frequency="7min"),
        lambda b: plot_intrabar(b, start=b.index[0], frequency="7min", segments=0),
        lambda b: plot_feature_distribution(b, "close", bins=0),
        lambda b: plot_feature_correlation(b, ["close"], method="unknown"),
        lambda b: plot_price_time_profile(b, start=b.index[0], frequency="7min", mode="unknown"),
        lambda b: plot_profile_bars(b.iloc[:2], frequency="7min", origin=b.index[0]),
    ],
)
def test_invalid_plot_inputs(bars, call):
    with pytest.raises(ValueError):
        call(bars)


def test_import_pfe_does_not_load_matplotlib():
    subprocess.run(
        [sys.executable, "-c", "import sys; import pfe; assert 'matplotlib' not in sys.modules"],
        check=True,
    )


def test_optional_dependency_error(bars, monkeypatch):
    import builtins

    original = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("blocked optional dependency")
        return original(name, *args, **kwargs)

    assert importlib.util.find_spec("matplotlib") is not None
    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    with pytest.raises(ImportError, match="uv sync --extra plots"):
        plot_bars(bars)
