# Feature API and conventions

## Five operations

- `compute_features(bars, windows=(5, 20, 60), groups=None)`: OHLCV feature frame,
  same bar-open index, all groups by default. Windows count source observations.
- `enrich_bars(bars, frequency, ...)`: target OHLCV plus path, profile and quality
  columns. Accepts arbitrary positive fixed source multiples: 7min, 45min, 2h,
  1D, 3s from 1s input. The target duration must be at least the source duration.
- `build_features(bars, frequency, windows=..., groups=..., ...)`: both families;
  rolling windows count target bars, internal descriptors use source bars.
- `transform_features(frame, ...)`: positive lags, differences, trailing mean/std,
  z-score, percentile rank, slope, tanh and explicitly selected products/ratios.
- `multi_timeframe_features(bars, frequencies=("7min", "45min"), ...)`: combines
  several completed target resolutions on source observations with a backward
  as-of join of target availability to source close timestamps. Prefixes identify
  each resolution; `age_seconds` exposes stale last-known values across gaps.

Use `feature_catalog(result)` for the exact output-column inventory. It includes
configured windows, bins, segments and dtypes. `feature_catalog()` without an
argument lists groups and all supported native TA-Lib functions. The CLI can
export this inventory as CSV with `--catalog`.

## OHLCV groups

| Group | Implemented families |
|---|---|
| `price` | Median, typical, OHLC mean, weighted close, log prices/log1p volume |
| `returns` | Simple/log, absolute changes, open-close, gaps (`open_gap` aliases `gap`), cumulative horizons, acceleration, volatility scaling |
| `candles` | Body, range, wicks, ratios, direction, overlap, expansion, inside/outside |
| `statistics` | Mean/median/extrema/std/variance, unbiased skew and excess kurtosis, quantiles, IQR, MAD, ranks, classical/robust z-scores on each OHLCV field, return, range, body and wicks |
| `volatility` | True range, simple ATR, close std, EWMA, realized variance/volatility, positive/negative semivariance, vol-of-vol, ranks/short-long ratio, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang |
| `trend` | SMA/EMA, distance/slope/curvature/cross/age, regression slope/intercept/R²/residual, efficiency, VHF, choppiness |
| `volume` | Changes, dollar-volume/Amihud proxies, signed-volume proxy, sum/mean/EMA/std, relative volume, rank/z-score, zeros/concentration, rolling VWAP proxy and price-volume correlations |
| `levels` | Rolling high/low/position/age, prior-window breakouts, drawdowns/drawups, maximum drawdown, time below current rolling maximum |
| `sequences` | Positive fraction, direction changes, runs, return/absolute/squared autocorrelations |
| `technical` | All TA-Lib overlap, momentum, volume, volatility, price transforms, cycle and statistics functions; plus extensions below |
| `patterns` | All 61 TA-Lib candle patterns, numeric direction/score, age since pattern, trailing frequency |
| `complexity` | Return/log-price/log-volume distribution tails, historical expected shortfall, entropy, Hurst estimate, Petrosian dimension, AR(1)/lag-2 PACF, FFT frequency/energy/entropy/coefficients, Haar detail energy/coefficients |
| `structure` | Higher/lower highs/lows, confirmed swing levels/events/ages, tolerance-based double tops/bottoms, slope convergence, consolidation width, confirmed-threshold ZigZag |
| `calendar` | Local clock encodings, session progress, evolving session levels/volume/VWAP, opening range, same-slot relative volume/movement, previous-session pivots, night/weekend opening gaps |

Native TA-Lib extensions implemented here: Wilder RMA, ALMA, HMA, ZLEMA, VIDYA,
FRAMA, TSI, Vortex, Fisher, SMI, KDJ, Schaff Trend Cycle, Awesome Oscillator,
Coppock, Connors RSI, KST, Elder Ray, Force Index, Ease of Movement, CMF, PVT,
NVI/PVI, volume oscillator, Market Facilitation Index, Chaikin volatility,
Mass Index, RVI, Bollinger derived measures, Keltner, Donchian, envelopes,
Supertrend, causal DPO, Heikin-Ashi and available Ichimoku spans.

TA-Lib functions with periods get one variant per configured window. Fast periods
use `max(2, w//2)`, slow `max(3,w)`, signal `max(2,w//3)`; stochastic periods follow
the same rule; Ultimate Oscillator uses `max(2,w//2), w, 2w`. Other parameters keep
the SDK defaults. Functions without periods appear once. Native BETA/CORREL use
the abstract API's high/low inputs; these are internal high-low relationships,
not a beta against an external market benchmark. MAVP receives a constant period
equal to the window (no external adaptive period series). See the actual input
mapping in `feature_catalog()`.

Extension parameters are explicit conventions: ALMA offset=.85/sigma=6; envelopes ±2%;
Supertrend multiplier=3; Keltner multiplier=2; FRAMA rounds odd windows up to an even
length; Connors RSI periods `max(2,w//3), max(2,w//5), w`; KST horizons w/2w/3w/4w;
Ichimoku horizons w/2w/4w, visible spans shifted by 2w. Fisher uses rolling median-price
range, clipped at ±.999. ZigZag and repeated confirmed pivot tolerance use
`structure_threshold=.01` by default. Pattern definitions follow these conventions;
visual interpretations may differ. Signals become available when their pivots are
confirmed.

## Linear-model outputs

Walk-forward OLS/Ridge outputs use the separate `compute_linear_feature` /
`compute_linear_features` APIs, with prediction, fit, coefficients and errors
groups. See [linear-model features](linear.md) for the catalog and label timing.

External macro observations and announcement calendars use `compute_macro_feature`
and `compute_macro_features`. Groups cover series, calendar, surprises and ordered
spreads. See [macro data and features](macro.md) for schemas and release availability.
They do not add model fitting to ordinary OHLCV group calls.

Options use `compute_option_feature` / `compute_option_features`, with activity,
volatility, greeks, positioning, flow, contract and quality groups. They consume
external chain/print/contract frames; see [options features](options.md).

## Intrabar groups

| Prefix | Preserved internal information |
|---|---|
| `path` variability | Realized variance/volatility, observed variance, absolute/quadratic variation, semivariances, fine ranges |
| `path` direction | Efficiency, positive/negative/flat fractions and sums, imbalance, turns, mean/longest runs |
| `path` timing | First/last high/low, retests, ordering, delays, extreme-to-close time |
| `path` excursions | Close-path drawdown/drawup, peak-to-trough and recovery times, recovered flag/fraction, threshold episodes, underwater observations/seconds/runs |
| `path` concentration | Largest signed/absolute move, top 1/3/5 energy share, energy concentration, threshold counts |
| `path` organization | Early/late variance shares, variance migration across segments, absolute-return slope/autocorrelation, largest-move timing |
| `path.segmentN` | Return/range/variance/volatility/volume/share/close position per equal-time segment and changes from preceding segment |
| `profile` | Close-zone shares/entropy/concentration, stays/transitions/revisits, TPO matrix, POC/70% value area, single-block rows, intersegment overlap and migration |
| `path` shape | Regression slope/intercept/R²/error on relative log closes, curvature, signed/absolute area, straight-path deviation, open crossings |
| `path` volume | Concentration/entropy, temporal center, peak/slope/zeros, high-volume episodes/runs, segment shares |
| `path` coupling | Return/absolute-return/range-volume correlation, signed volume shares, weighted returns, top-move volume, low-volume large moves and delayed correlations |
| `path.scaleN` | Realized variance, ratio, turns and directional agreement at internal source multiples |
| `quality` | Observed/expected/missing source bars, coverage, return coverage, largest gap, span, unchanged closes, local-date boundary, complete/closed flags |

`segments=5` divides target elapsed duration into five equal time spans (A…E).
Rows are assigned by their source-bar open timestamp, never by row count; missing
segments stay empty. Columns use numeric segment IDs to also support more than
26 segments. `bins=10` divides the target high-low range into price rows. Close
occupation counts sampled closes. TPO presence marks rows intersecting a source
bar's low-high range. `price_time_profile` returns its boolean `[bins,segments]`
matrix, close zones, price-row centers and segment assignments.

A flat target range occupies one central bin. TPO POC ties choose the lower
central tied row. The contiguous 70% value area expands from POC, adding both
neighbors on equal counts. Its boundaries are row centers. VWAP uses typical
price times bar volume; TPO/occupation and VWAP remain OHLCV proxies, not exact
transaction-level quantities. No bid/ask, trade count, true order imbalance or
volume-at-price is reconstructed.

## Time, gaps and missing values

Input and output indexes label opens. A source bar becomes available at
`index + source_frequency`. Enriched bars have explicit `available_at = target
open + target duration`. A full 45-minute feature cannot be used before that
close. Joining higher-timeframe data back to source observations must use
availability timestamps, with source decisions also timestamped at bar close.

`origin="start_day"` anchors a UTC elapsed-time grid at the first input day's
midnight; `offset` shifts it. Use an explicit timezone-aware `origin` to anchor
at a desired session open. Alignment is not inferred from sessions. Fixed 1D
means 24 elapsed hours, not one trading session; no exchange-holiday/early-close
calendar or exclusion of closed-market slots is inferred by aggregation.

`partial="drop"` (default) excludes truncated first/last target bins; `keep`
returns provisional edge bins, and `raise` rejects them. Internal holes are
retained and reported. Entirely empty bins are omitted. `as_of` selects only
source bars whose closes have arrived and cannot exceed the last observed close.
`build_features` requires drop to keep its standard indicators causal.

Intrabar returns start with `log(first_close / first_open)`, then use consecutive
source closes. A source-time gap invalidates the bridging return. Full realized
variance is `sum(r²)` only with complete source coverage; otherwise it is NaN.
`realized_variance_observed` reports the sum of the valid observed increments.
Other descriptive path metrics summarize the observations that exist; they do
not infer missing movements. Fine price quadratic variation is also NaN with
holes. Internal scale comparisons require full coverage and divisibility by the
scale; otherwise they stay NaN. Values are not annualized.

Close-path drawdowns/drawups use the target open as initial sample. Drawups apply
the drawdown algorithm to reciprocal prices (magnitude `1 - trough/peak`).
Recovery fractions can exceed 1 after overshoot. Underwater seconds use observed
sample intervals, not exact tick timing. High/low timing denotes their source
bar, with order NaN when both occur in that same bar. `threshold=.001` controls
large log-return counts and relative-price excursion thresholds; high-volume
episodes are bars with volume greater than twice the target mean.

OHLCV rolling state defaults to `gap_policy="reset"`; `allow` deliberately counts
consecutive observations across gaps. Sessions default to America/New_York,
09:30–16:00, weekdays by clock only. They do not claim holiday-calendar coverage.
Only source bars wholly inside those hours participate. Previous levels use the
previous observed clock session, with coverage reported. Opening-range features
appear only once their configurable duration (default 15min) has elapsed.
Same-slot baselines use preceding observed sessions, with no current-session fit.

### Opening gaps

| Feature | Definition |
|---|---|
| `returns.open_gap` | Alias of `returns.gap`: current bar open / preceding bar close − 1. With `gap_policy="reset"`, timestamp gaps yield NaN; `allow` bridges consecutive observed rows. |
| `session.night_gap` | Current regular-session open / previous observed regular-session close − 1, for closures containing no Saturday/Sunday. |
| `session.weekend_gap` | The same session opening return, for closures containing Saturday/Sunday (including a Monday holiday followed by a Tuesday open). |

Night/weekend gaps use `session_timezone`, `session_start`, `session_end` and
`source_frequency`. They require an observed bar opening exactly at session start
and a previous observed session's last bar ending exactly at session end.
Missing endpoints, the first session and rows outside regular hours yield NaN;
the other gap category is NaN, not zero. Valid gaps persist throughout the current
session and are usable after its first bar closes. Values are fractional returns:
0.01 means +1%. Extended-hours bars never replace the regular-session close.
Weekend detection uses local calendar days and is DST-safe. Holidays and missing
whole sessions are not inferred: the comparison uses the previous observed
regular session, and can span multiple dates. The existing `session.overnight_gap`
keeps its broader previous-observed-close behavior, including incomplete sessions.

```python
compute_feature(bars, "returns.open_gap")
compute_selected_features(bars, ("session.night_gap", "session.weekend_gap"))
compute_features(bars, groups=("returns", "calendar"))
```

Individual calculations evaluate only the gap dependencies; existing session
streaming limitations also apply to the two session gaps (see [performance](performance.md)).

Zero denominators, insufficient history and undefined standardized moments are NaN.
Warm-up values remain missing, and calculations use past observations only. Rolling
skew/kurtosis center within each past window so rounding cannot depend on future input
means.

Complexity methods are finite-window estimates, not claims that a financial
series follows a particular stochastic model. Sample entropy excludes self-pairs
and uses order 2/3, tolerance .2×window std; no matches yields NaN. Permutation
entropy uses order 3 with stable tie ordering. Hurst regresses increment std on
lags 2…min(20,n//2). Wavelets use Haar/periodization inside the past window only.
Spectrum and wavelet calculations never use the full future dataset.

## Cost and validation

Full output can contain over a thousand columns depending on windows. Statistics
and especially sample entropy scale with historical length/window sizes; select
groups and windows for large archives. Aggregation processes observed source rows
per bin and exposes a configurable profile size (`bins × segments`).

Tests cover manual OHLCV/realized volatility, identical coarse OHLCV with different
paths, all 61 native patterns, future-data invariance for each family, gap resets,
partial availability, missing segments, arbitrary minute/subminute granularities,
session look-ahead, zero-volume/constant prices and invalid configuration.

References: [TA-Lib](https://github.com/TA-Lib/ta-lib-python),
[pandas resampling](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html),
[PyWavelets](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html).
