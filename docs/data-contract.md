# Single-ticker OHLCV data contract

`pfe.normalize_bars` and `pfe.load_bars` return a pandas DataFrame with:

- A sorted, unique, non-null UTC DatetimeIndex named `timestamp`.
- Timestamps aligned to `frequency` (default `1min`) and exactly
  `open`, `high`, `low`, `close`, `volume`. Subminute input can use `frequency="1s"`.
- Float64, finite values; positive prices, non-negative volume, consistent OHLC.

The index represents the provider's bar-open time. The full OHLCV bar becomes
available only after that minute finishes. Additional provider columns are
discarded. Input is copied. Invalid values and duplicate timestamps raise errors;
rows are not silently removed or deduplicated.

Missing minutes, overnight gaps, holidays and extended sessions stay unchanged.
One-minute resolution does not mean every adjacent row is one minute apart.
No calendar filtering, resampling, filling, split adjustment or dividend
adjustment is performed. Adjustment status and coverage must be checked against
the downloaded dataset before research.

Timezone-aware inputs convert to UTC. Naive inputs require `source_timezone`.
Local-time DST ambiguity raises an error. Numeric epochs require conversion with
an explicit unit before normalization. For display/session logic, use
`America/New_York`, never a fixed UTC offset.

## London Strategic Edge

Use the official [lse-data SDK](https://github.com/londonstrategicedge/lse-data).
`history(ticker, timeframe="1m", start=..., end=..., dest=...)` downloads a bulk
Parquet export. The script retains raw files and creates a validated local copy.
Vault bars use `ts`; REST `candles` uses `timestamp`. Vault timestamps are UTC.
The script enforces a start-inclusive, end-exclusive window locally, independently
of provider endpoint boundary behavior.
The bulk export endpoint requires `YYYY-MM-DD` bounds. The script sends the UTC
start date and rounds an intraday end up to the next UTC midnight for export,
then filters the downloaded bars to the precise requested timestamps. A date-only
end is already midnight and is sent unchanged.

```bash
uv run --extra lse --env-file .env python scripts/download_data.py \
  --ticker SPY --start 2026-01-05 --end 2026-01-10
```

`--ticker` defaults to QQQ, trims whitespace and uppercases the symbol. Choose a
symbol available through your LSE account; the provider validates availability.
Raw files use `data/raw/lse/<TICKER>/1m/<WINDOW>/`; normalized files use
`data/processed/<TICKER>_1m_<WINDOW>.parquet`. A window includes UTC start/end
timestamps. Existing exports for that ticker/window are protected from overwrite.
Each file contains one instrument; load symbols separately for feature calculation.

A live API key is needed for downloads. The repository can be developed and tested
offline with synthetic data. Export quotas and available history are provider
settings; consult the SDK documentation and your account for current values.
