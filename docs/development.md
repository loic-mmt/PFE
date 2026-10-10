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

Keep provider/network code in scripts and data ingestion, never in feature
functions. The package imports without LSE installed and without an API key.
Matplotlib is optional (`plots` extra) and imported only inside plotting functions.
Plot tests inspect data/timestamps and render offscreen; CI installs both extras.

## Feature APIs

`compute_feature` and `compute_selected_features` add exact-column computation;
the existing group APIs are unchanged. See `performance.md` for bounded batches,
supported streaming state, and measurements. Selection plans compile the
repository's own Python formulas into dependency graphs; never user-supplied
code. Configuration loops/dictionaries are specialized, numeric work is lazy,
and cached plans retain metadata only. Add formulas using the existing writer
and assignment conventions. Unsupported syntax fails explicitly; it must never
fall back to evaluating the whole family. The selection tests compare every
source and intrabar output individually against the group implementation.

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

The downloader accepts one LSE ticker per run; QQQ remains the default example.
Feature input always contains a single instrument. Source frequency can
be finer than one minute. True exchange calendars, backtests, and general model
training pipelines remain outside the package. The dedicated linear feature API
fits OLS/Ridge on matured rolling labels; see `linear.md` for timing and output
selection. Forward targets stay internal and are never exported as features.
Macro inputs use explicit vintage/schedule availability and dedicated selected/group
APIs; see `macro.md`. Their downloader is bounded and the offline example uses
synthetic observations. Network/provider data stays outside tests and Git.
Dependencies use TA-Lib for its native
indicator/pattern implementations, SciPy for sample entropy distances and
PyWavelets for finite-window Haar coefficients. NumPy is constrained below 2.4
for pandas 2.x timedelta compatibility and tested together through `uv.lock`.

The C++17 extension is built through CMake, scikit-build-core and pybind11.
It is the default backend for TPO/intrabar and supported rolling callbacks;
SciPy/PyWavelets remain available in the reference backend. See `cpp.md` for
scope, buffer validation, fallback semantics and reproducible measurements.
Extend both the C++ dispatch and Python formulas when adding a native descriptor.
Keep C++ formatting consistent with `.clang-format`. Differential tests must
include constant inputs, gaps, warmup, odd window lengths and read-only strides.

## macOS editable installation

This checkout exhibited the [hidden `.pth` issue](https://github.com/astral-sh/uv/issues/16977):
Python ignores an editable-install path when macOS marks its `.pth` file hidden.
Clearing the flags in `.venv/` proved temporary on this machine. On macOS,
`make setup` now renames the existing environment to `venv/` and creates a
`.venv` symlink, preserving uv's default path and existing activation scripts.
It clears inherited directory flags and hidden `.pth` flags in that environment.
The migration refuses to overwrite an existing `venv/`. Nothing is deleted;
no global Python installation is changed. Linux keeps the standard `.venv/`.
