# Plotting

Install `uv sync --extra plots` (or `pip install 'pfe[plots]'`). Combine with
`--extra lse` when downloading. Matplotlib remains optional: importing `pfe` and
computing features never imports it. `make setup` installs both extras.

All functions are available from `pfe` or `pfe.plots`. They return a Matplotlib
`Figure`, leave input data unchanged, and let the caller choose when to `show` or
`savefig`. Use `fig.savefig('chart.png', dpi=150)` for PNG, or choose `.svg` / `.pdf`
for vector exports ([Matplotlib savefig
API](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html)).
In Jupyter, leave the figure as the last expression. In a desktop script use
`matplotlib.pyplot.show()` after creating your figures; close finished figures with
`matplotlib.pyplot.close(fig)`. For batch servers use `MPLBACKEND=Agg` before launching
Python. Set `MPLCONFIGDIR` to a writable cache directory if necessary.

All plots use a terminal-inspired theme: black canvas, orange/white curves and
labels, monospace typography, and subdued grid lines. The theme is applied only
to returned figures; global Matplotlib settings remain unchanged. Candle rises
are orange and declines white.

TPO cell fills use 14% orange opacity by default; letters retain full contrast.
Both profile functions accept `cell_alpha` between 0 and 1 (`0` removes the fill,
`1` makes it opaque). For example:

```python
fig = plot_profile_bars(bars, frequency="30min", bins=20, segments=5, cell_alpha=0.10)
```

## Candles and volume

```python
from pfe import build_features, plot_bars

features = build_features(
    bars, "7min", windows=(5, 20), groups=("technical",), origin=bars.index[0]
)
fig = plot_bars(
    bars,
    frequency="7min",
    origin=bars.index[0],
    overlays=features,
    columns=("technical.sma_w5", "technical.sma_w20"),
    timezone="America/New_York",
    title="QQQ · 7 minutes",
)
fig.savefig("qqq.png", dpi=150)
```

`frequency=None` draws existing bars; their width uses `bars.attrs['frequency']` when
present, otherwise `source_frequency='1min'`. Pass the actual frequency for frames
loaded from files without metadata. With `frequency='7min'`, aggregation uses the
existing `enrich_bars` rules: any fixed integer source multiple, configurable `origin`
and `offset`, truncated edge bars dropped. Empty intervals remain gaps. Candle bodies
are centered inside each actual bar interval; temporal gaps stay visible. `volume=False`
removes the volume panel. `figsize=(12, 6)` controls size.

`overlays` is a separate numeric frame; `columns` must select a small explicit
set of price-scale columns. Overlays use availability times when metadata exists,
otherwise their index. Use `plot_features` for indicators with different units.

## Feature time series

```python
from pfe import plot_features

fig = plot_features(features, ["technical.rsi_w5", "technical.atr_w20"])
```

One panel per column by default; `separate=False` puts all lines on one axis.
`time='auto'` uses `available_at` for enriched frames, or index + source frequency
for `compute_features` / `multi_timeframe_features` metadata. Without availability
metadata it uses the index and labels it as such. After CSV round trips, pass
`source_frequency='7min', time='available'` explicitly if index labels target
opens. `time='available'` rejects missing metadata; `time='index'` explicitly draws
at index times. Warm-up NaNs remain gaps, with no filling or interpolation.
Features plotted at open times can describe future information relative to that
open; the explicit availability mode avoids that visual misalignment.

## One target bar's internal path

```python
from pfe import plot_intrabar, plot_price_time_profile

fig = plot_intrabar(
    bars, start=bars.index[0], frequency="45min", source_frequency="1min", segments=5
)
profile = plot_price_time_profile(bars, start=bars.index[0], frequency="45min", bins=12, segments=5)
```

`start` must be timezone-aware and align with the source grid. Functions inspect
source bars whose complete intervals lie inside `[start, start + frequency)`.
They can show a partially observed window: coverage is shown on the path figure;
the profile leaves unobserved segments empty. They do not assert the target has
closed. Use an explicit data slice ending at your decision time for historical
inspection without later observations. `source_frequency` supports seconds, too.

`plot_intrabar` has four panels: opening/source-close path with source high-low
ranges, source volume, cumulative squared log returns, and drawdown based on the
first observed open and observed closes. The first return is first close / first
open; subsequent returns require adjacent source slots. Missing slots break the
price path and invalidate cumulative variance from that point. Drawdown uses
observed points and is not a tick-level maximum drawdown. Equal elapsed-time
segments are shaded and labeled A, B, C…; the range markers do not reconstruct the
order of prices within a source bar. No annualization is applied.

`plot_price_time_profile` defaults to `mode='compact'`: letters in each price row
are packed against the left edge, retaining their original labels and chronological
order. A row containing A and E displays `AE`, with no empty B/C/D slots. Its width
is the number of occupied time segments, producing the familiar TPO silhouette.
`letters=False` hides text while retaining the silhouette. `mode='timeline'`
restores the time-column heatmap and row-count panel.

Both modes use the exact `price_time_profile` presence matrix. An occupied cell
means a source OHLC range intersects the price row, not actual tick dwell time,
trade count, or volume. High price bins are at the top; flat prices use the central
row. Shapes describe the observed distribution, not a measured buy/sell imbalance.

To construct a series of candles with these profiles:

```python
from pfe import plot_profile_bars

fig = plot_profile_bars(bars, frequency="45min", bins=20, segments=5, origin=bars.index[0])
```

Each completed target interval contains a left-packed profile on a shared actual price
axis. One letter has the same horizontal width across candles. Each target divides its
own high-low range into `bins` price rows; unobserved segments remain empty.
Granularity, `source_frequency`, `origin`, `offset`, `timezone` and `letters` are
configurable. Slice the displayed interval to keep letters readable.

## Distributions and correlation

```python
from pfe import plot_feature_distribution, plot_feature_correlation

columns = ["technical.rsi_w5", "technical.atr_w20"]
histograms = plot_feature_distribution(features, columns, bins=40)
correlation = plot_feature_correlation(features, columns, method="spearman", min_periods=20)
```

Histograms exclude NaNs/infinities and report retained/omitted counts. Correlations
use pairwise finite observations, require `min_periods` (default 3), and leave
undefined cells blank. `method` supports Spearman (default), Pearson, Kendall;
`annotate=False` hides cell values. These charts summarize the entire supplied
sample: they are exploratory plots, not causal rolling features or evidence of
predictive value. Select columns explicitly to keep plots readable.

## Executable example

`make plots` exports seven PNG files to ignored `data/plots/`, using synthetic data.
`uv run --extra plots python examples/plots.py --frequency 45min --output-dir data/demo`
changes target granularity/output directory. The example contains 210 source
minutes; choose a target that fits that sample. Plot functions themselves have no
such sample-size limit.
