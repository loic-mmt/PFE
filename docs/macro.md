# Macro data and features

Macro inputs use separate APIs because observation periods, announcement schedules,
forecasts and outcomes have different availability times. Market output indexes
still label **bar opens**; calculations use `index + source_frequency` as the
decision time. Source duration defaults to `1min`, and can be any positive fixed
duration. `compute_macro_feature` calculates one requested output;
`compute_macro_features` accepts exact `names` or `groups`.

## Small LSE downloads

```bash
uv run --extra lse --env-file .env python scripts/download_macro.py \
  --series fdtr cpi_yoy --calendar --region US \
  --start 2025-01-01 --end 2025-01-08 --limit 20 \
  --max-bytes 1048576 --output data/raw/lse/macro/sample.json
```

Requires `LSE_API_KEY`. The script queries individual economics series and the economic
calendar, with one bounded response per request. Start is inclusive, end exclusive, UTC;
`YYYY-MM-DD` and timezone-aware ISO timestamps work. Each request returns at most
`--limit` rows (default 100, maximum 5000). The output cap defaults to 1 MiB; oversized
responses are rejected before writing. Existing files are protected from overwrite.
`limit_reached` flags samples that may be incomplete.

Downloaded files stay outside Git. Local snapshot loading and feature calculation run
offline with the core library.

```python
from pfe import load_macro, compute_macro_feature, compute_macro_features

data = load_macro("data/raw/lse/macro/sample.json")
inflation = compute_macro_feature(bars, "macro.cpi_yoy.level", observations=data.series)
macro = compute_macro_features(
    bars,
    observations=data.series,
    events=data.calendar,
    groups=("series", "calendar"),
    windows=(3, 12),
    event_window="30min",
)
```

**Latest LSE snapshots do not provide historical vintages.** Economics rows contain
`symbol`, `date`, `value`; `date` identifies an observation period. Calendar rows
include `datetime`, `event`, `actual`, `consensus`, `forecast`, `previous` and
ingestion timestamps. In the inspected sample, ingestion timestamps could be
months after the event. Neither schedule time nor ingestion metadata establishes
when a historical outcome or forecast first became available.

`load_macro` therefore sets **all availability and schedule-known timestamps to
snapshot receipt time**, and sets `point_in_time=False`. Historical bars preceding
receipt receive no values, even when observations describe earlier years. For
historical backtests, supply verified vintage/release data with explicit timestamps
using the canonical schemas below. An arbitrary delay or a revised latest value
does not reconstruct the original release. See the [FRED vintage-time explanation](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

## Canonical inputs

`normalize_macro_series(frame)` validates these long-form columns:

| Column | Meaning |
|---|---|
| `series` | Non-empty identifier, e.g. `policy_rate` |
| `observation_at` | Timezone-aware reference-period timestamp |
| `available_at` | When this exact observation vintage became usable |
| `value` | Finite numeric value; zero and negative values are allowed |
| `point_in_time` | Boolean provenance flag; defaults to true for explicit caller data |

Later vintages of the same period are allowed; identical period/availability keys
are rejected. Revision replay updates historical periods only after availability.
An old-period revision changes its rolling window but never replaces the current
level with an older observation. Simultaneous snapshot rows produce one final
state containing all known reference periods.

`normalize_macro_calendar(frame)` accepts one row per unique `event_id`:

| Column | Meaning |
|---|---|
| `event_id`, `event` | Unique occurrence ID and comparable event-type label |
| `scheduled_at`, `known_at` | Announcement time and when its schedule became known |
| `actual`, `forecast`, `previous` | Optional numeric values, NaN when unavailable |
| `<field>_available_at` | Required timezone-aware availability for each non-missing field |
| `point_in_time` | Boolean provenance flag, as above |

Canonical timestamps must be timezone-aware; normalizers convert them to UTC.
Field availability cannot precede `known_at`. This first version accepts one
schedule/field snapshot per event; it does **not** replay changing announcement
schedules or multiple outcome revisions. Resolve those histories before passing
a point-in-time calendar, rather than supplying a later schedule as its original.

LSE percentages remain percentage points (`"3.1%"` becomes `3.1`); K/M/B scale by
1e3/1e6/1e9. Missing values remain NaN; unsupported formats raise an error.
Consensus is used as `forecast`, falling back to the provider forecast when absent.
Raw event labels are retained. Rename them to consistent types such as `CPI` before
rolling surprise calculations; provider labels can contain month/period suffixes.
Mixing event types or incompatible units makes rolling statistics meaningless.

## Output catalog

| Group | Exact names or templates |
|---|---|
| `series` | `macro.<series>.level`, `change`, `pct_change`, `revision`, `is_revision`, `point_in_time`, `age_seconds`, `observation_age_seconds`, `mean_wN`, `std_wN`, `zscore_wN`, `slope_wN` |
| `calendar` | `macro.calendar.seconds_to_next_event`, `seconds_since_event`, `events_next_window`, `events_previous_window`, `event_now` |
| `surprise` | `macro.event[TYPE].actual`, `forecast`, `previous`, `surprise`, `surprise_relative`, `actual_change`, `age_seconds`, `point_in_time`, `surprise_zscore_wN` |
| `spreads` | `macro.pair[LEFT,RIGHT].spread`, `ratio`; enable with `pairs=(("LEFT", "RIGHT"),)` |

Series windows count distinct reference periods, not market bars. Standard deviation
uses `ddof=0`; slope is value change per observation ordinal. Zero denominators,
constant-series z-scores and insufficient histories produce NaN. Series provenance
describes the current level. Revisions describe the last received update; their
values persist until another observation update, rather than being market-row pulses.

Calendar features use only schedules known by the decision time. Next counts use
`(t, t + event_window]`; previous counts use `[t - event_window, t]`; `event_now`
counts exact-time announcements. No future outcome is needed for schedule features.

Event fields track the last received value for that type. Surprise is actual minus
forecast, available only after both inputs are known. Relative surprise divides
by the absolute forecast. Actual change requires the previous value as well.
Surprise z-scores use only the **previous** N known surprises of the same type;
simultaneously received outcomes cannot train each other's scale. A forecast first
known after the scheduled announcement gives `point_in_time=0`; a current snapshot
can show a numerical difference but cannot verify a pre-announcement consensus.

`max_age="30D"`, for example, masks series/event values after their receipt age
exceeds that duration; explicit age columns and calendar schedules stay visible.
Pair operands align independently by availability and require compatible units.

Sparse transforms run on macro records. Backward alignment and calendar counts
use vectorized searches with bounded temporary chunks. Output memory scales with
market rows × selected features; select names and process market chunks to control
memory. Retain sufficient macro vintage history for rolling windows in every chunk.
The [synthetic example](../examples/macro_features.py) runs entirely offline.
