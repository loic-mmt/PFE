"""Optional Matplotlib plots for OHLCV, features and source-bar paths.

Functions return a Figure without displaying or saving it. Matplotlib is loaded
only when plotting; importing pfe does not require the ``plots`` extra.
Black/orange/white styling is local to each figure, with no global rcParams edits.
"""

from collections.abc import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from pfe.data import normalize_bars
from pfe.features.aggregation import enrich_bars, fixed_duration
from pfe.features.profiles import price_time_profile

_BLACK, _ORANGE, _WHITE = "#000000", "#ff8c00", "#f4f4f5"
_UP, _DOWN = _ORANGE, _WHITE
_CURVES = (_ORANGE, _WHITE, "#ffbd69", "#a3a3a3")


def _matplotlib():
    try:
        import matplotlib.dates as dates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError("Plotting requires Matplotlib: uv sync --extra plots") from exc
    return plt, dates, Rectangle


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a non-empty DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Expected a timezone-aware DatetimeIndex")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be non-missing, unique and sorted")


def _columns(frame, columns):
    selected = [columns] if isinstance(columns, str) else list(columns)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one column, without duplicates")
    missing = set(selected) - set(frame.columns)
    if missing:
        raise ValueError(f"Unknown columns: {sorted(missing)}")
    if any(not is_numeric_dtype(frame[c]) for c in selected):
        raise ValueError("Selected columns must be numeric")
    return selected


def _times(frame, time, source_frequency):
    if time not in {"auto", "available", "index"}:
        raise ValueError("time must be 'auto', 'available' or 'index'")
    if time == "index":
        return frame.index, "index"
    if "available_at" in frame:
        values = pd.DatetimeIndex(frame["available_at"])
        if values.tz is None or values.hasnans or not values.is_monotonic_increasing:
            raise ValueError("available_at must contain sorted, timezone-aware timestamps")
        return values, "availability"
    step = source_frequency
    if step is None and frame.attrs.get("available_at") == "index + source_frequency":
        step = frame.attrs.get("source_frequency")
    if step is not None:
        return frame.index + fixed_duration(step, "source_frequency"), "availability"
    if time == "available":
        raise ValueError("Availability metadata missing; provide source_frequency or time='index'")
    return frame.index, "index"


def _date_axis(ax, dates, timezone, *, semantics="bar open"):
    tz = ZoneInfo(timezone)
    locator = dates.AutoDateLocator(tz=tz, minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator, tz=tz))
    ax.set_xlabel(f"Time ({timezone}; {semantics})")


def _style(ax):
    ax.grid(color="#292929", alpha=0.7, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _finish(fig, title):
    """Apply a local terminal theme without changing Matplotlib's global rcParams."""
    fig.set_facecolor(_BLACK)
    for ax in fig.axes:
        ax.set_facecolor(_BLACK)
        ax.tick_params(colors=_WHITE, labelsize=9, length=3, width=0.6)
        for spine in ax.spines.values():
            spine.set_color("#454545")
        for label in (
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
            ax.xaxis.get_offset_text(),
            ax.yaxis.get_offset_text(),
        ):
            label.set_color(_WHITE)
            label.set_fontfamily("monospace")
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
            label.set_color(_ORANGE)
            label.set_fontfamily("monospace")
            label.set_fontsize(9)
        for text in ax.texts:
            text.set_fontfamily("monospace")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(_BLACK)
            legend.get_frame().set_edgecolor("#454545")
            legend.get_frame().set_alpha(0.9)
            for text in legend.get_texts():
                text.set_color(_WHITE)
                text.set_fontfamily("monospace")
    fig.suptitle(
        title,
        x=0.02,
        ha="left",
        color=_ORANGE,
        fontfamily="monospace",
        fontsize=12,
        fontweight="bold",
    )
    return fig


def _cell_alpha(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("cell_alpha must be a finite number between 0 and 1")
    return float(value)


def _candles(ax, data, step, dates, rectangle):
    # Center each body inside its actual [open, close) interval, preserving gaps.
    # Establish date units before adding collections: a later datetime overlay
    # would otherwise trigger relim(), which ignores wick collections.
    ax.xaxis_date(tz="UTC")
    x = dates.date2num(data.index.to_pydatetime())
    width = step.total_seconds() / 86400
    centers = x + width / 2
    colors = np.where(data.close >= data.open, _UP, _DOWN)
    ax.vlines(centers, data.low, data.high, colors=colors, linewidth=0.9)
    for center, opening, close, color in zip(centers, data.open, data.close, colors, strict=True):
        height = abs(close - opening)
        if height:
            ax.add_patch(
                rectangle(
                    (center - width * 0.35, min(opening, close)),
                    width * 0.7,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.6,
                )
            )
        else:
            ax.hlines(close, center - width * 0.35, center + width * 0.35, color=color)
    ax.set_xlim(x[0] - width * 0.2, x[-1] + width * 1.2)
    ax.set_ylabel("Price")
    _style(ax)
    return centers, width, colors


def plot_bars(
    bars: pd.DataFrame,
    *,
    frequency: str | pd.Timedelta | None = None,
    source_frequency: str | pd.Timedelta = "1min",
    origin: str | pd.Timestamp = "start_day",
    offset: str | pd.Timedelta | None = None,
    volume: bool = True,
    overlays: pd.DataFrame | None = None,
    columns: str | Sequence[str] | None = None,
    timezone: str = "UTC",
    title: str = "OHLCV",
    figsize: tuple[float, float] = (12, 6),
):
    """Plot candles and optional volume/price overlays at any fixed source multiple.

    Supplying frequency aggregates with enrich_bars and drops truncated edge bins.
    Without frequency, existing bars are drawn using attrs['frequency'] or
    source_frequency. Overlay times use availability metadata when present.
    """
    if frequency is None:
        data = normalize_bars(bars, frequency="1ns")
        step = fixed_duration(bars.attrs.get("frequency", source_frequency), "source_frequency")
    else:
        data = enrich_bars(
            bars, frequency, source_frequency=source_frequency, origin=origin, offset=offset
        )
        if data.empty:
            raise ValueError("No completed target bars to plot")
        step = fixed_duration(frequency, "frequency")
    if ((data.index - data.index[0]).asi8 % step.value != 0).any():
        raise ValueError("Bars must lie on the supplied frequency grid")
    selected = None
    if overlays is not None:
        _frame(overlays)
        if columns is None:
            raise ValueError("Select overlay columns explicitly")
        selected = _columns(overlays, columns)
        overlay_times, _ = _times(overlays, "auto", None)
    elif columns is not None:
        raise ValueError("columns requires overlays")
    ZoneInfo(timezone)
    plt, dates, rectangle = _matplotlib()
    fig, axes = plt.subplots(
        2 if volume else 1,
        1,
        sharex=True,
        squeeze=False,
        figsize=figsize,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]} if volume else None,
    )
    ax = axes[0, 0]
    centers, width, colors = _candles(ax, data, step, dates, rectangle)
    if selected:
        for i, name in enumerate(selected):
            ax.plot(
                overlay_times,
                overlays[name].replace([np.inf, -np.inf], np.nan),
                label=name,
                linewidth=1.2,
                color=_CURVES[i % len(_CURVES)],
            )
        ax.legend(fontsize=8)
    if volume:
        axes[1, 0].bar(centers, data.volume, width=width * 0.7, color=colors)
        axes[1, 0].set_ylabel("Volume")
        _style(axes[1, 0])
    _date_axis(axes[-1, 0], dates, timezone, semantics="bar intervals")
    return _finish(fig, title)


def plot_features(
    features: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    separate: bool = True,
    time: str = "auto",
    source_frequency: str | pd.Timedelta | None = None,
    timezone: str = "UTC",
    title: str = "Features",
    figsize: tuple[float, float] | None = None,
):
    """Plot selected features, preserving warm-up NaNs and separating units by default.

    auto uses available_at or compute_features metadata, else the index. Explicit
    time='available' requires metadata or source_frequency. Nothing is filled.
    """
    _frame(features)
    selected = _columns(features, columns)
    times, semantics = _times(features, time, source_frequency)
    ZoneInfo(timezone)
    plt, dates, _ = _matplotlib()
    count = len(selected) if separate else 1
    fig, axes = plt.subplots(
        count,
        1,
        sharex=True,
        squeeze=False,
        figsize=figsize or (12, max(3, 2.2 * count)),
        layout="constrained",
    )
    for i, name in enumerate(selected):
        ax = axes[i if separate else 0, 0]
        ax.plot(
            times,
            features[name].replace([np.inf, -np.inf], np.nan),
            label=name,
            linewidth=1.2,
            color=_CURVES[i % len(_CURVES)],
        )
        ax.legend(loc="upper left", fontsize=8)
        _style(ax)
    _date_axis(axes[-1, 0], dates, timezone, semantics=semantics)
    return _finish(fig, title)


def _window(bars, start, frequency, source_frequency):
    data = normalize_bars(bars, frequency="1ns")
    start = pd.Timestamp(start)
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    start = start.tz_convert("UTC")
    duration = fixed_duration(frequency, "frequency")
    source = fixed_duration(source_frequency, "source_frequency")
    if duration < source or duration.value % source.value:
        raise ValueError("frequency must be an integer multiple of source_frequency")
    group = data.loc[(data.index >= start) & (data.index + source <= start + duration)]
    if group.empty:
        raise ValueError("No source bars in the requested window")
    if ((group.index - start).asi8 % source.value != 0).any():
        raise ValueError("start must align with the source_frequency grid")
    return group, start, duration, source


def plot_intrabar(
    bars: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    frequency: str | pd.Timedelta,
    source_frequency: str | pd.Timedelta = "1min",
    segments: int = 5,
    timezone: str = "UTC",
    title: str = "Intrabar path",
    figsize: tuple[float, float] = (12, 8),
):
    """Plot source close path/ranges, volume, cumulative variance and close drawdown.

    Segments divide elapsed time equally. Gaps break the path and invalidate
    cumulative variance from the first missing slot. Source ranges are not ticks.
    """
    group, start, duration, source = _window(bars, start, frequency, source_frequency)
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
        raise ValueError("segments must be a positive integer")
    ZoneInfo(timezone)
    plt, dates, _ = _matplotlib()
    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=figsize,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )
    closes = group.index + source
    grid = pd.date_range(start, start + duration - source, freq=source)
    complete = group.reindex(grid)
    path = pd.Series(np.nan, index=pd.date_range(start, start + duration, freq=source))
    path.iloc[1:] = complete.close.to_numpy()
    path.iloc[0] = complete.open.iloc[0]
    axes[0].plot(path.index, path, color=_ORANGE, marker=".", markersize=3, label="Source closes")
    midpoint = group.index + source / 2
    axes[0].vlines(
        dates.date2num(midpoint.to_pydatetime()),
        group.low,
        group.high,
        color=_WHITE,
        alpha=0.3,
        label="Source OHLC ranges",
    )
    axes[0].axhline(group.open.iloc[0], color="gray", linestyle="--", linewidth=0.8)
    for i in range(segments):
        left, right = start + duration * i / segments, start + duration * (i + 1) / segments
        axes[0].axvspan(left, right, color=_ORANGE, alpha=0.025 if i % 2 else 0.065)
        axes[0].text(
            left + (right - left) / 2,
            0.98,
            _letter(i),
            ha="center",
            va="top",
            transform=axes[0].get_xaxis_transform(),
            fontsize=9,
            color=_WHITE,
        )
    colors = np.where(group.close >= group.open, _UP, _DOWN)
    axes[1].bar(
        dates.date2num(midpoint.to_pydatetime()),
        group.volume,
        width=source.total_seconds() / 86400 * 0.75,
        color=colors,
    )
    log_returns = np.log(complete.close).diff()
    log_returns.iloc[0] = np.log(complete.close.iloc[0] / complete.open.iloc[0])
    axes[2].plot(grid + source, log_returns.pow(2).cumsum(skipna=False), color=_ORANGE)
    # Drawdowns remain estimates from observed opens/closes, not intraminute ticks.
    observed = np.r_[group.open.iloc[0], group.close.to_numpy()]
    drawdown = observed[1:] / np.maximum.accumulate(observed)[1:] - 1
    axes[3].plot(closes, drawdown * 100, color=_DOWN, marker=".", linewidth=1)
    for ax, label in zip(
        axes, ("Price", "Volume", "Cumulative RV", "Close drawdown (%)"), strict=True
    ):
        ax.set_ylabel(label)
        ax.set_xlim(start, start + duration)
        _style(ax)
    axes[0].legend(fontsize=8, loc="lower left")
    _date_axis(axes[-1], dates, timezone, semantics="source closes / intervals")
    coverage = len(group) / len(grid)
    return _finish(fig, f"{title} · {duration} · coverage {coverage:.0%}")


def _letter(index):
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label


def plot_price_time_profile(
    bars: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    frequency: str | pd.Timedelta,
    source_frequency: str | pd.Timedelta = "1min",
    bins: int = 10,
    segments: int = 5,
    letters: bool = True,
    cell_alpha: float = 0.14,
    mode: str = "compact",
    timezone: str = "UTC",
    title: str = "TPO profile (source-range proxy)",
    figsize: tuple[float, float] = (10, 6),
):
    """Show a left-aligned TPO silhouette; A/B/... identify original time segments.

    Reuses price_time_profile exactly. Presence means a source [low, high] range
    intersects the row, not exact tick visits, traded volume or dwell time.
    Occupied letters are packed left per price row, retaining chronological order
    and their original labels even when a time segment is missing. Width is TPO
    count, not traded volume. mode='timeline' restores the price/time matrix.
    cell_alpha controls only cell fill opacity; letters remain fully opaque.
    """
    if mode not in {"compact", "timeline"}:
        raise ValueError("mode must be 'compact' or 'timeline'")
    cell_alpha = _cell_alpha(cell_alpha)
    group, start, duration, _ = _window(bars, start, frequency, source_frequency)
    presence, _, centers, _ = price_time_profile(
        group, start=start, duration=duration, bins=bins, segments=segments
    )
    tz = ZoneInfo(timezone)
    plt, _, _ = _matplotlib()
    from matplotlib.colors import ListedColormap, to_rgba

    labels = [_letter(i) for i in range(segments)]
    if mode == "compact":
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        counts = presence.sum(axis=1)
        for row in range(bins):
            active = np.flatnonzero(presence[row])
            for slot, segment in enumerate(active):
                ax.barh(
                    row,
                    1,
                    left=slot,
                    height=0.96,
                    color=to_rgba(_ORANGE, cell_alpha),
                    edgecolor=to_rgba(_ORANGE, 0.25),
                    linewidth=0.45,
                )
                if letters:
                    ax.text(
                        slot + 0.5,
                        row,
                        labels[segment],
                        ha="center",
                        va="center",
                        fontfamily="monospace",
                        color=_WHITE,
                        fontsize=max(5, min(12, 160 / max(1, int(counts.max())))),
                    )
        rows = np.unique(np.linspace(0, bins - 1, min(bins, 12), dtype=int))
        ax.set_yticks(rows, [f"{centers[i]:.6g}" for i in rows])
        ax.set_ylim(-0.5, bins - 0.5)
        ax.set_xlim(0, int(counts.max()) + 0.5)
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("TPO count (letters packed left)")
        ax.set_ylabel("Price bin center")
        _style(ax)
        return _finish(fig, f"{title}\n{start.tz_convert(tz):%Y-%m-%d %H:%M:%S} · {duration}")

    fig, axes = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        layout="constrained",
        gridspec_kw={"width_ratios": [4, 1]},
    )
    axes[0].imshow(
        presence.astype(int),
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap=ListedColormap([to_rgba(_BLACK, 0), to_rgba(_ORANGE, cell_alpha)]),
        interpolation="nearest",
    )
    if letters:
        for row, segment in zip(*np.where(presence), strict=True):
            axes[0].text(
                segment,
                row,
                labels[segment],
                ha="center",
                va="center",
                color=_WHITE,
                fontsize=max(5, min(12, 120 / segments)),
            )
    shown = np.unique(np.linspace(0, segments - 1, min(segments, 12), dtype=int))
    times = [(start + duration * int(i) / segments).tz_convert(tz) for i in shown]
    axes[0].set_xticks(
        shown, [f"{labels[i]}\n{t:%H:%M:%S}" for i, t in zip(shown, times, strict=True)]
    )
    rows = np.unique(np.linspace(0, bins - 1, min(bins, 12), dtype=int))
    axes[0].set_yticks(rows, [f"{centers[i]:.6g}" for i in rows])
    axes[0].set_ylabel("Price bin center")
    axes[0].set_xlabel(f"Equal-time segment starts ({timezone})")
    axes[1].barh(np.arange(bins), presence.sum(axis=1), color=_ORANGE, alpha=0.5, height=0.8)
    axes[1].set_xlabel("TPO blocks")
    axes[1].set_xlim(0, segments + 0.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, f"{title}\n{start.tz_convert(tz):%Y-%m-%d %H:%M:%S} · {duration}")


def plot_profile_bars(
    bars: pd.DataFrame,
    *,
    frequency: str | pd.Timedelta,
    source_frequency: str | pd.Timedelta = "1min",
    origin: str | pd.Timestamp = "start_day",
    offset: str | pd.Timedelta | None = None,
    bins: int = 20,
    segments: int = 5,
    letters: bool = True,
    cell_alpha: float = 0.14,
    timezone: str = "UTC",
    title: str = "TPO profile candles (source-range proxy)",
    figsize: tuple[float, float] = (14, 7),
):
    """Draw a series of target candles as left-packed letter profiles.

    Each target uses its own high-low price bins, on a shared actual price axis.
    One TPO occupies the same fraction of a target interval across all candles;
    counts are comparable with fixed segments. Only completed target intervals
    are shown; missing source periods do not create letters. No volume allocation
    or buy/sell pressure is inferred from a profile's shape.
    cell_alpha controls only cell fill opacity; letters remain fully opaque.
    """
    cell_alpha = _cell_alpha(cell_alpha)
    target = enrich_bars(
        bars,
        frequency,
        source_frequency=source_frequency,
        origin=origin,
        offset=offset,
        bins=bins,
        segments=segments,
    )
    if target.empty:
        raise ValueError("No completed target bars to plot")
    data = normalize_bars(bars, frequency="1ns")
    duration = fixed_duration(frequency, "frequency")
    source = fixed_duration(source_frequency, "source_frequency")
    ZoneInfo(timezone)
    plt, dates, rectangle = _matplotlib()
    from matplotlib.colors import to_rgba

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    width = duration.total_seconds() / 86400
    cell_width = width * 0.85 / segments
    for start in target.index:
        group = data.loc[(data.index >= start) & (data.index + source <= start + duration)]
        presence, _, centers, _ = price_time_profile(
            group, start=start, duration=duration, bins=bins, segments=segments
        )
        low, high = float(group.low.min()), float(group.high.max())
        height = (high - low) / bins
        # Give a flat profile a small visible price band, without changing its center.
        if height == 0:
            height = max(low * 1e-5, np.finfo(float).eps)
        left = dates.date2num(start.to_pydatetime()) + width * 0.05
        for row in range(bins):
            for slot, segment in enumerate(np.flatnonzero(presence[row])):
                x = left + slot * cell_width
                ax.add_patch(
                    rectangle(
                        (x, centers[row] - height / 2),
                        cell_width,
                        height,
                        facecolor=to_rgba(_ORANGE, cell_alpha),
                        edgecolor=to_rgba(_ORANGE, 0.25),
                        linewidth=0.45,
                    )
                )
                if letters:
                    ax.text(
                        x + cell_width / 2,
                        centers[row],
                        _letter(segment),
                        ha="center",
                        va="center",
                        color=_WHITE,
                        fontfamily="monospace",
                        fontsize=8,
                        clip_on=True,
                    )
    ax.autoscale_view()
    ax.set_xlim(
        dates.date2num(target.index[0].to_pydatetime()),
        dates.date2num((target.index[-1] + duration).to_pydatetime()),
    )
    ax.set_ylabel("Price · horizontal width = TPO count")
    shown = np.unique(np.linspace(0, len(target) - 1, min(len(target), 8), dtype=int))
    starts = target.index[shown].tz_convert(timezone)
    ax.set_xticks(
        dates.date2num(starts.to_pydatetime()),
        [f"{value:%m-%d}\n{value:%H:%M:%S}" for value in starts],
    )
    ax.set_xlabel(f"Target bar opens ({timezone}); letters packed within each profile")
    _style(ax)
    return _finish(fig, title)


def plot_feature_distribution(
    features: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    bins: int = 40,
    title: str = "Feature distributions",
    figsize: tuple[float, float] | None = None,
):
    """Histograms of finite values; missing/infinite values are omitted and counted."""
    _frame(features)
    selected = _columns(features, columns)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    plt, _, _ = _matplotlib()
    fig, axes = plt.subplots(
        len(selected),
        1,
        squeeze=False,
        figsize=figsize or (10, max(3, 2.5 * len(selected))),
        layout="constrained",
    )
    for ax, name in zip(axes[:, 0], selected, strict=True):
        values = features[name].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(values, bins=bins, color=_ORANGE, alpha=0.7)
        ax.set_title(
            f"{name} · n={len(values)} · omitted={len(features) - len(values)}", fontsize=9
        )
        ax.set_ylabel("Count")
        _style(ax)
    return _finish(fig, title)


def plot_feature_correlation(
    features: pd.DataFrame,
    columns: Sequence[str],
    *,
    method: str = "spearman",
    min_periods: int = 3,
    annotate: bool = True,
    title: str = "Feature correlation",
    figsize: tuple[float, float] | None = None,
):
    """Pairwise finite-value correlations; undefined cells stay blank.

    Descriptive over the supplied sample (not a causal rolling feature).
    Spearman is default; Pearson and Kendall are also supported.
    """
    _frame(features)
    selected = _columns(features, columns)
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be 'pearson', 'spearman' or 'kendall'")
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 2:
        raise ValueError("min_periods must be an integer >= 2")
    values = features[selected].replace([np.inf, -np.inf], np.nan)
    correlation = values.corr(method=method, min_periods=min_periods)
    # Pandas forces a diagonal of 1 for Kendall, even on constant columns.
    undefined = values.nunique(dropna=True).lt(2) | values.count().lt(min_periods)
    correlation.loc[undefined, :] = np.nan
    correlation.loc[:, undefined] = np.nan
    plt, _, _ = _matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    count = len(selected)
    side = max(6, count * 0.65)
    fig, ax = plt.subplots(figsize=figsize or (side, side), layout="constrained")
    image = ax.imshow(
        np.ma.masked_invalid(correlation.to_numpy()),
        vmin=-1,
        vmax=1,
        cmap=LinearSegmentedColormap.from_list("pfe_terminal", [_WHITE, "#111111", _ORANGE]),
        interpolation="nearest",
    )
    ax.set_xticks(range(count), selected, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(count), selected, fontsize=8)
    if annotate:
        for row in range(count):
            for col in range(count):
                value = correlation.iloc[row, col]
                if pd.notna(value):
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=_BLACK if abs(value) > 0.6 else _WHITE,
                        fontsize=8,
                    )
    fig.colorbar(image, ax=ax, label=f"{method.capitalize()} correlation")
    return _finish(fig, f"{title} · pairwise observations")
