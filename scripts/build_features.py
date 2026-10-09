"""Build configurable enriched bars and OHLCV features from a local market-data file."""

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pfe import (
    FEATURE_GROUPS,
    build_features,
    enrich_bars,
    feature_catalog,
    iter_bars,
    iter_features,
    load_bars,
)


def save_batches(frames, destination):
    """Write atomically: failed computations never leave a partial output file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(suffix=".parquet", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    writer, first, count = None, None, 0
    try:
        for frame in frames:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
                first = frame.iloc[:0]
            writer.write_table(table)
            count += len(frame)
        if writer is None:
            raise ValueError("No closed target bars in the input window")
        writer.close()
        writer = None
        temporary.replace(destination)
        return count, first
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV or Parquet OHLCV file")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet file")
    parser.add_argument("--frequency", required=True, help="Target duration: 7min, 45min, 2h, etc.")
    parser.add_argument("--source-frequency", default="1min")
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--groups", nargs="+", choices=FEATURE_GROUPS)
    parser.add_argument(
        "--features", nargs="+", help="Exact feature names; computes only dependencies"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Stream Parquet in bounded batches; requires --features (e.g. 100000)",
    )
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--origin", default="start_day", help="start_day, epoch, start, or UTC ISO timestamp"
    )
    parser.add_argument("--offset", default=None)
    parser.add_argument("--gap-policy", choices=("reset", "allow"), default="reset")
    parser.add_argument(
        "--path-only", action="store_true", help="Only aggregation and intrabar features"
    )
    parser.add_argument(
        "--catalog", type=Path, help="Write the exact output-column inventory as CSV"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.features and (args.groups or args.path_only):
        parser.error("Choose --features, --groups or --path-only")
    if args.batch_size is not None and (
        not args.features or args.batch_size < 1 or args.input.suffix.lower() != ".parquet"
    ):
        parser.error("--batch-size requires --features, a positive size and Parquet input")
    if args.output.suffix.lower() != ".parquet":
        parser.error("--output must end in .parquet")
    outputs = [args.output] + ([args.catalog] if args.catalog else [])
    if len({path.resolve() for path in [args.input, *outputs]}) != 1 + len(outputs):
        parser.error("Input, output and catalog must be different paths")
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Output already exists; choose a new path or --overwrite")
    origin = args.origin
    if origin not in {"start_day", "epoch", "start"}:
        origin = pd.Timestamp(origin)
        if origin.tzinfo is None:
            parser.error("Explicit --origin must include a timezone")
    options = dict(
        source_frequency=args.source_frequency,
        segments=args.segments,
        bins=args.bins,
        origin=origin,
        offset=args.offset,
    )
    if args.batch_size is not None:
        batches = iter_bars(args.input, batch_size=args.batch_size)
        frames = iter_features(
            batches,
            args.features,
            frequency=args.frequency,
            gap_policy=args.gap_policy,
            validate=False,
            **options,
        )
        count, schema = save_batches(frames, args.output)
        if args.catalog:
            args.catalog.parent.mkdir(parents=True, exist_ok=True)
            feature_catalog(schema).to_csv(args.catalog, index=False)
        print(f"Saved {count:,} bars, {len(schema.columns):,} columns: {args.output}")
        return
    bars = load_bars(args.input, frequency="1ns")
    if args.path_only:
        result = enrich_bars(bars, args.frequency, **options)
    else:
        result = build_features(
            bars,
            args.frequency,
            windows=tuple(args.windows),
            groups=args.groups,
            names=args.features,
            gap_policy=args.gap_policy,
            **options,
        )
    if result.empty:
        parser.error("No closed target bars in the input window")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output)
    if args.catalog:
        args.catalog.parent.mkdir(parents=True, exist_ok=True)
        feature_catalog(result).to_csv(args.catalog, index=False)
    print(f"Saved {len(result):,} bars, {len(result.columns):,} columns: {args.output}")


if __name__ == "__main__":
    main()
