# Options data and features

Options use dedicated `compute_option_feature` / `compute_option_features` APIs.
Select one exact name, several `names`, or `groups`. Market indexes label bar opens;
all outputs use information available by `index + source_frequency`, default `1min`.
Any positive fixed source duration works. Ordinary OHLCV group APIs remain unchanged.

## Small LSE samples

```bash
uv run --extra lse --env-file .env python scripts/download_options.py \
  --ticker QQQ --chain --flow --limit 20 --max-bytes 1048576
```

Use `--expiry YYYY-MM-DD`, `--type call|put`, and paired `--strike-min` / `--strike-max`
bounds to narrow a chain. An explicit `--contract OSI_TICKER` adds 1-minute candles for
one contract. Paired `--start` / `--end` bound prints/candles in UTC, with inclusive
start and exclusive end. They do **not** make the current chain historical. Each request
reads one response for the selected chain or contract. Defaults: 20 rows per query and a
1 MiB file cap. Existing outputs are protected; choose another `--output` for a new
capture.

The SDK may retrieve its instrument catalog internally to resolve a ticker; this
metadata stays in memory. Only requested option rows are saved. Samples and `.env`
stay outside Git; examples/tests use synthetic data without network access.
The [official SDK](https://github.com/londonstrategicedge/lse-data#options) documents
chain, prints and per-contract candles. IV/greeks are provider model outputs.

```python
from pfe import load_options, compute_option_feature, compute_option_features

data = load_options("data/raw/lse/options/QQQ_sample.json")
iv = compute_option_feature(bars, "options.volatility.iv_mean", chain=data.chain)
features = compute_option_features(
    bars,
    chain=data.chain,
    flow=data.flow,
    underlying="QQQ",
    names=(
        "options.volatility.atm_iv",
        "options.activity.put_call_volume_ratio",
        "options.flow.total_premium",
        "options.quality.expired_fraction",
    ),
    flow_window="30min",
    max_dte=90,
)
```

Chain rows contain the latest **traded** premium, IV/greeks, `volume_today`,
`premium_today`, and `last_trade_at`. They are not synchronized bid/ask quotes.
In the inspected sample, the DTE filter returned expired contracts with stale
stored DTE. Computation always derives DTE from expiration and reference date;
expired rows are excluded from financial summaries but counted by quality outputs.
The sampled responses contained no open interest, bid/ask or aggressor direction.
Missing fields remain NaN, never zero. Cap hits and filters mean summaries describe
the **supplied subset**, not guaranteed complete market totals.
Latest-trade records can carry old daily counters; `volume_today` is preserved
as observed, not independently verified as activity on the receipt date. Inspect
trade ages before interpreting an activity summary as a current-session measure.

## Availability and canonical inputs

`load_options` defaults to `availability="receipt"`: the chain, prints and candles
become usable when this response was received, with candles no earlier than close.
Past market bars therefore receive no current-chain values. For prints/candles,
`availability="event", latency="1s"` explicitly assumes delivery at trade time or
candle close plus a delay. That is an event-time research assumption, not verified
historical delivery or revision timing; `point_in_time` stays false. Chains always
use receipt time, regardless of this option. A chain must be recorded over time
to build historical surface features.

All canonical frames require `underlying`, `ticker`, `contract_type` (`call`/`put`),
positive `strike`, and `expiry` (a calendar date). Canonical time columns require
timezone-aware timestamps. Numeric optional fields include `underlying_price`, `iv`,
`delta`, `gamma`, `theta`, `vega`, `rho`. IV retains provider decimal units, e.g. 0.20;
greeks retain provider units and remain missing when absent from the source.

| Normalizer | Additional required fields | Optional fields |
|---|---|---|
| `normalize_option_chain` | `snapshot_at`, `available_at` | `last_price`, `volume`, `premium`, `quote_at`, `open_interest`, `bid`, `ask` |
| `normalize_option_flow` | `trade_id`, `trade_at`, `available_at`, `volume`, `premium` | `last_price` |
| `normalize_option_candles` | `bar_at`, `available_at`, OHLC, `volume`, `premium`, `print_count` | IV/greeks represent bar averages |

`quote_at` maps to the chain's latest trade time. Chain snapshots are atomic:
every contract in a `(underlying, snapshot_at)` batch has the same availability.
Duplicate contracts/prints/candles, crossed quotes, invalid OHLC and premature
availability are rejected. Late older snapshots cannot replace a newer chain.
Caller-supplied canonical data defaults to `point_in_time=True`; callers must
provide genuine known-at timestamps to justify that flag.

`min_dte` / `max_dte` filter chain rows using snapshot date and flow rows using
trade/candle date. Expiration day is DTE zero; no precise settlement time is
inferred. Chain summaries expire at the snapshot's **next UTC midnight** because
the membership filter changes daily. Quality ages remain visible. Record a new
snapshot each day; optional `max_age` imposes a stricter receipt-age limit.
For contract values, `max_age` measures age since the source candle close.
With multiple instruments, explicitly choose `underlying`; it must match the
market bars supplied by the caller.

## Feature groups

Names below are suffixes of `options.<group>.`, except the contract template.

| Group | Exact suffixes |
|---|---|
| `activity` | `call_volume`, `put_volume`, `total_volume`, `put_call_volume_ratio`, `volume_imbalance`; corresponding five `premium` suffixes |
| `volatility` | `iv_mean`, `iv_std`, `iv_volume_weighted`, `atm_call_iv`, `atm_put_iv`, `atm_iv`, `put_call_atm_iv_spread`, `risk_reversal`, `butterfly`, `term_spread`, `term_slope`, `expected_move_relative` |
| `greeks` | `delta_mean`, `delta_volume_weighted`; same suffixes for gamma/theta/vega/rho |
| `positioning` | `call_open_interest`, `put_open_interest`, `put_call_open_interest_ratio`, `oi_concentration`, `call_oi_peak_strike`, `put_oi_peak_strike`, `oi_weighted_strike`, `unsigned_gamma_oi` |
| `flow` | Activity suffixes, `print_count`, `average_premium`, `iv_volume_weighted`, `delta_contracts`, `point_in_time_fraction` |
| `contract` | `options.contract[TICKER].log_return`, `intrabar_log_return`, `range_relative`, `iv`, delta/gamma/theta/vega/rho, `volume`, `premium`, `print_count`, `age_seconds` |
| `quality` | `contract_count`, `active_contract_count`, `expired_fraction`, `iv_coverage`, `open_interest_coverage`, `quote_coverage`, `point_in_time`, `limit_reached`, `snapshot_age_seconds`, `oldest_trade_age_seconds` |

Activity totals use observed contracts' cumulative daily volume/premium. Imbalances
are `(call - put) / (call + put)`; put/call ratios divide by calls. Missing numeric
contributions make sums/weighted statistics NaN instead of silently dropping them.
Unweighted means use known values; quality coverage shows their missing fraction.

Surface features use the nearest selected expiry. ATM means choose the strike
nearest its row's `underlying_price` per side, within `atm_tolerance` (default 3%).
`atm_iv` averages available call/put estimates; the same rule applies per maturity.
The underlying price is the provider's contract observation, not a reconstructed
current spot. Wings select the closest call +0.25 and put -0.25 delta within
`delta_tolerance=0.10`, configurable via `delta_target`. No interpolation or
extrapolation is implied. Risk reversal is call-wing IV minus put-wing IV;
butterfly is their mean minus ATM IV. Term spread is farthest-expiry ATM IV minus
front ATM IV; slope divides it by the calendar-day separation. Expected move is
`ATM IV * sqrt(DTE / 365)`, a conventional approximation; zero-DTE stays NaN
without an intraday settlement timestamp. IV standard deviation uses `ddof=0`.

OI concentration is strike-level HHI; peaks use the greatest observed OI, with
the lower strike winning ties. Unsigned gamma/OI is `sum(gamma * open_interest)`
in unscaled provider units. It is **not dealer gamma exposure**: positions,
contract multipliers and dealer long/short signs are absent. Likewise calls minus
puts does not identify buying pressure, and trade delta is not aggressor direction.
See [OIC's explanation of option greeks](https://www.optionseducation.org/advancedconcepts/volatility-the-greeks).

Flow windows contain observed events in `(decision - flow_window, decision]`
that have also become available. A delayed print does not fill earlier windows;
if already outside its event-time window at arrival, it contributes nothing.
For candles the event time is source candle close. `flow_source="prints"` uses
prints; `"candles"` uses per-contract bars. Never sum both sources, which overlap.
With both inputs supplied, prints are the default. Candle coverage can be a small
set of contracts and is not an underlying-wide tape.
With candles, IV/delta aggregates weight the provider's bar averages by bar volume;
they approximate print-level aggregates when individual trade sizes differ.

Average premium divides by print count; IV is volume-weighted; `delta_contracts`
sums delta × traded contract volume, with no multiplier or dealer-sign assumption.
`point_in_time_fraction` is the fraction of contributing input rows with verified
provenance. Zero denominators and empty statistical windows give NaN; observed
empty counts/sums give zero. Last contract-bar values persist until a new bar or
`max_age`; consecutive log returns reset across missing source intervals.

Calculations transform sparse snapshots/prints, then align them using vectorized
searches with bounded market-row temporaries. Output memory scales with rows × selected
features. Process market chunks while retaining the same necessary option history for
equivalent results. Run [the offline example](../examples/option_features.py), or use
existing `plot_features` on the computed numeric DataFrame.