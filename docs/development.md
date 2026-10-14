# Development

## First feature

1. Add a module under `src/pfe/features/`, for example `returns.py`.
2. Accept bars already normalized with `pfe.normalize_bars` or `pfe.load_bars`.
3. Return a named Series or DataFrame with the same UTC index; document parameters,
   units, warm-up length, and whether windows count observations or elapsed time.
4. Add hand-computed examples under `tests/`. Verify that appending or modifying
   future bars does not change previous feature values.
5. Run `make check` and `make build`.

Features at time t can use the completed bar at t and earlier bars. An order based
on that feature must execute after the bar closes. No negative shifts, centered
windows, backfilling warm-up values, or full-sample fitted normalization. Keep
forward-looking targets separate from features.

Keep provider/network code in scripts and data ingestion. Feature functions operate on
local data. The package imports without LSE installed and without an API key. Matplotlib
is optional (`plots` extra) and imported only inside plotting functions. Plot tests
inspect data/timestamps and render offscreen. CI installs the optional LSE, plots,
calendars and commodities extras; Excel is imported only for XLSX.

New collectors should use the [Lot 0 contracts and RawStore](qualification.md):
preserve economic periods, publication and availability separately; append source
revisions and raw pages, and declare completion explicitly. Qualify catalogue
coverage, units, adjustments and rights before fetching history. The observation
and security-master APIs add a common layer; existing price/macro/options loaders
retain their specialized input schemas.

Daily inputs use [exchange sessions and corporate actions](daily-market.md).
Use `load_daily_bars` to retain provenance/quality; `load_bars` is the specialized
OHLCV-only input. Reuse `compute_daily_features` for session windows instead of
running a UTC 24-hour resampler. Tests inject schedules and provider responses;
the `calendars` extra also validates actual XNYS holidays/DST/early closes.

## Feature APIs

`compute_feature` and `compute_selected_features` add exact-column computation; the
existing group APIs are unchanged. See `performance.md` for bounded batches, supported
streaming state, and measurements. Selection plans compile repository formulas into
dependency graphs. User-supplied code is excluded from compilation. Configuration
loops/dictionaries are specialized, numeric work is lazy, and cached plans retain
metadata only. Add formulas using the existing writer and assignment conventions.
Unsupported syntax raises an error rather than evaluating the whole family. The
selection tests compare every source and intrabar output individually against the group
implementation.

New recurrent indicators require explicit streaming state or must be marked as
having unbounded history. New finite helpers must expose their full lookback to
the planner, including nested windows. Tests must cross batch boundaries and
source gaps. Numba kernels keep `fastmath=False`; the first call can incur JIT
cost. Multi-output native primitives may compute coupled outputs internally.

`compute_features` handles source OHLCV; `enrich_bars` preserves internal paths
at a configurable target frequency; `build_features` combines both. Consult
`features.md` for the inventory and conventions. Every family has a test that
changes future data and checks that previous values remain unchanged. Enriched
bars also have hand-computed volatility and availability tests.

The downloader accepts multiple tickers for native daily data; QQQ remains the intraday
default. Daily features split by instrument and use actual exchange calendars. Source
frequency can be finer than one minute. Backtests and general model training pipelines
remain outside the package. The dedicated linear feature API fits OLS/Ridge on matured
rolling labels; see `linear.md` for timing and output selection. Forward targets stay
internal, outside feature outputs. Macro inputs use explicit vintage/schedule
availability and dedicated selected/group APIs; see `macro.md`. Their downloader is
bounded and the offline example uses synthetic observations. Network/provider data stays
outside tests and Git. Lot 2 [context features](context-features.md) accept explicit
decision instants. References require exact identities, units and matched benchmark
horizons; fundamentals require compatible accounting periods and concepts. FRED/ALFRED,
SEC, EIA, Cboe and World Bank collectors use immutable raw versions and bounded HTTP.
Feature calculations receive the collected data. Add hand-calculated tests for
unit/period compatibility and future-revision invariance. Use receipt time for current
snapshots and preserve the economic period separately. Options have separate canonical
chain, print and candle schemas, with explicit availability and optional missing
OI/quotes. Exact output selection skips unrelated input sources; see `options.md`.
Derive DTE from expiration dates for expiry filtering. Dependencies use TA-Lib for its
native indicator/pattern implementations, SciPy for sample entropy distances and
PyWavelets for finite-window Haar coefficients. NumPy is constrained below 2.4 for
pandas 2.x timedelta compatibility and tested together through `uv.lock`.

The C++17 extension is built through CMake, scikit-build-core and pybind11.
It is the default backend for TPO/intrabar and supported rolling callbacks;
SciPy/PyWavelets remain available in the reference backend. See `cpp.md` for
scope, buffer validation, fallback semantics and reproducible measurements.
Extend both the C++ dispatch and Python formulas when adding a native descriptor.
Keep C++ formatting consistent with `.clang-format`. Differential tests must
include constant inputs, gaps, warmup, odd window lengths and read-only strides.

## macOS editable installation

This checkout exhibited the [hidden `.pth`
issue](https://github.com/astral-sh/uv/issues/16977): Python ignores an editable-install
path when macOS marks its `.pth` file hidden. Clearing the flags in `.venv/` proved
temporary on this machine. On macOS, `make setup` now renames the existing environment
to `venv/` and creates a `.venv` symlink, preserving uv's default path and existing
activation scripts. It clears inherited directory flags and hidden `.pth` flags in that
environment. The migration stops if `venv/` already exists. It preserves the environment
and global Python installation. Linux uses the standard `.venv/`.

 Document tests use a synthetic corpus and injected Ollama/HTTP/PDF responses. Run
`python examples/document_features.py` for an offline end-to-end demonstration. See
[document features](document-features.md) for exact evidence, review/cache, identity
provenance, graph/event availability and the precision/recall release gate. Measure
model accuracy separately on an annotated real-world corpus.

[International feature tests](international-features.md) use exact provider selections,
synthetic multilingual HTML/CSV/JSON/TSV responses and injected HTTP. Test sparse
JSON-stat, ROC dates, fiscal bases, units, receipt gates, late/null revisions, graph
knowledge gates, GDELT recapture/caps, key redaction, recovery and offline CLI exports.
`examples/international_features.py` performs no network or model calls and removes its
temporary archive automatically.

Dataset and P3 tests use synthetic COT JSON, licensed-export fixtures, injected LSE row
transports and synthetic events. Check release gates, estimate fiscal/ unit/basis
comparability and strictly pre-announcement surprise, option snapshot versus receipt,
chronological label purging, validation outlier invariance of train scalers, chunk/batch
window equivalence, GNN/event knowledge/model gates, missing/padding masks, checksums
and bounded NPZ allocations. The offline example `examples/model_datasets.py` removes
its temporary export automatically.