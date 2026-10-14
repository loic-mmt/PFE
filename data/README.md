# Local data

Downloaded market data stays outside Git. Tests use synthetic inputs only.

The consuming project owns its data. `run_pipeline` publishes selected canonical tables
under a caller-chosen run directory, with immutable `raw/`, `stages/` and
`manifest.json`. Partitioned datasets use `part-000000/` shards and a final index.
Integration checks create small real snapshots; unit tests and examples use synthetic
inputs. See [reusable pipelines](../docs/reusable-pipelines.md).

- `raw/lse/<ticker>/1m/<window>/`: original LSE Parquet export, preserved as downloaded.
- `processed/<ticker>_1m_<window>.parquet`: validated canonical UTC OHLCV bars.
- `raw/lse/macro/*.json`: bounded macro snapshots from `scripts/download_macro.py`.
- `raw/lse/options/*.json`: bounded chain/print/contract samples from `scripts/download_options.py`.
- `raw/lse/discovery/<collection SHA256>/`: immutable metadata audit pages and manifests
  from `scripts/qualify_lse.py`; no market-data export.
- `raw/lse/snapshots/<collection SHA256>/`: daily/1m immutable pages, receipts and manifests.
- `processed/<ticker>_1d_<window>[_snapshot-id].parquet`: canonical exchange-session daily bars.
- `raw/lse/actions/<collection SHA256>/`: bounded split/dividend pages and audit summaries.
- `market/corporate_actions.parquet`: append-only normalized action revisions.
- `raw/fred/<collection SHA256>/`: capped metadata/observations/vintage/release pages.
- `macro/us_macro_vintages.parquet`: accumulated FRED/ALFRED canonical revisions.
- `raw/sec/<collection SHA256>/`: immutable submissions/history/companyfacts captures.
- `fundamentals/sec/`: per-collection filings and company facts; merge audits explicitly
  with `merge_company_facts`, retaining the first availability of identical revisions.
- `raw/context/<collection SHA256>/`: EIA pages and Cboe/World Bank file versions.
- `market/global_reference_assets.parquet`, `macro/commodities.parquet`: possible
  caller-selected canonical destinations; context CLI requires an explicit output path.

Shared data APIs provide versioned contracts, source declarations, an as-of security
master and a `RawStore` for collectors. See [qualification and
replay](../docs/qualification.md).

Generate both with `scripts/download_data.py --ticker QQQ --start YYYY-MM-DD --end YYYY-MM-DD`.
Choose another LSE ticker with `--ticker SPY`, for example. Symbols are uppercased;
exports for different tickers stay separate. Do not edit raw files. `--resume`
enables immutable replay for `1m`; daily snapshots resume automatically. An SDK
export interrupted before raw journaling needs a new snapshot or inspection.
Native daily, sessions, provenance and corporate-action conventions are described
in [Lot 1](../docs/daily-market.md).

Macro downloads default to 100 rows per query and a 1 MiB output cap, with no
pagination or bulk exports. Use `--limit 20` for an inspection sample. The loader
assigns latest-snapshot values to receipt time; observation dates do not establish
publication dates. See [macro inputs and availability](../docs/macro.md).

Options samples default to 20 rows/query and a 1 MiB cap. Current chains and
historical prints/candles are different inputs; receipt time is the loader's safe
default. See [options schemas and availability](../docs/options.md).

Context collectors preserve each source’s schema and units, including the difference
between monthly averages and daily closes. `download_fred.py`, `download_sec.py` and
`download_context.py` journal bounded responses for immutable replay. Changing provider
data requires a new snapshot/audit ID. Availability at receipt is the default; explicit
historical reconstructions do not establish verified PIT. See [Lot 2 commands, budgets
and timing policies](../docs/context-features.md).

 Document processing uses `raw/documents/` and `raw/extraction/` for immutable source
text, responses, cache and rejection journals. `metadata/documents.json` indexes
`source_documents.parquet`. Derived graphs/events and financial statements are exported
by `scripts/text_pipeline.py build`; unknown economic periods stay null, processing and
review retain their own availability times. Core JSON and Parquet budgets are explicit;
[commands and contracts](../docs/document-features.md) describe the files, review
workflow and optional model outputs. Model installation and data collection are separate
setup steps.

International captures use `raw/international/<collection-id>/` for source bytes,
receipt metadata, canonical Parquet and completion summaries. Selected numeric
observations can be published under `market/`, `macro/` or `fundamentals/`; GDELT
discovery/provider events stay separate from numeric Parquet and verified Lot 3 events.
Each capture is immutable; combine vintages explicitly with
`merge_international_observations`. Default response cap 512 KiB, payload archive cap 4
MiB. See [international contracts and CLI](../docs/international-features.md).

P3 collectors use `raw/p3/` for bounded immutable COT/authorized-estimate/option
captures. `merge_observation_captures` keeps the first receipt of overlapping
COT/estimate revisions. Option sample archives keep source snapshot time separate from
receipt; historical chain coverage requires a recorded sequence of snapshots.

`datasets/<run>/arrays.npz` and `manifest.json` preserve registry, per-cell
source/revision/document/time provenance, masks, inputs, separate future labels,
purged chronological splits and optional train-only scaling. GNN snapshots and
Event Encoder tensors are optional. Default artifact cap 8 MiB, no pickle or DL
framework; [model dataset contracts and CLI](../docs/model-datasets.md),
[P3 source commands](../docs/p3-sources.md).
