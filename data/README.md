# Local data

Downloaded market data stays outside Git. Tests use synthetic candles only.

- `raw/lse/<ticker>/1m/<window>/`: original LSE Parquet export, preserved as downloaded.
- `processed/<ticker>_1m_<window>.parquet`: validated canonical UTC OHLCV bars.
- `raw/lse/macro/*.json`: bounded macro snapshots from `scripts/download_macro.py`.

Generate both with `scripts/download_data.py --ticker QQQ --start YYYY-MM-DD --end YYYY-MM-DD`.
Choose another LSE ticker with `--ticker SPY`, for example. Symbols are uppercased;
exports for different tickers stay separate. Do not edit raw files. Retries after
a failed export require a fresh `--data-dir` or manual inspection of partial files.

Macro downloads default to 100 rows per query and a 1 MiB output cap, with no
pagination or bulk exports. Use `--limit 20` for an inspection sample. The loader
assigns latest-snapshot values to receipt time; observation dates do not establish
publication dates. See [macro inputs and availability](../docs/macro.md).
