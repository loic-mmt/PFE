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
    parser.add_argument("--frequency", help="Intraday target duration; daily accepts only 1d")
    parser.add_argument("--source-frequency")
    parser.add_argument(
        "--daily", action="store_true", help="Use canonical exchange-session daily bars"
    )
    parser.add_argument("--calendar", help="Explicit common exchange calendar for daily inputs")
    parser.add_argument(
        "--schedule", type=Path, help="Explicit daily schedule CSV/Parquet instead of a calendar"
    )
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
    if not args.daily and not args.frequency:
        parser.error("Intraday calculation requires --frequency")
    if args.daily:
        if args.frequency not in (None, "1d", "1D") or args.source_frequency not in (
            None,
            "1d",
            "1D",
        ):
            parser.error(
                "Daily uses exchange sessions; --frequency/--source-frequency accept only 1d"
            )
        if bool(args.calendar) == bool(args.schedule):
            parser.error("Daily requires exactly one of --calendar or --schedule")
        if args.batch_size is not None or args.path_only:
            parser.error(
                "Daily requires a full canonical input; intrabar path/stream modes unavailable"
            )
        if args.origin != "start_day" or args.offset is not None:
            parser.error(
                "Daily schedule supplies actual opens/closes; origin/offset cannot be used"
            )
    elif args.calendar or args.schedule:
        parser.error("--calendar/--schedule require --daily")
    if args.features and (args.groups or args.path_only):
        parser.error("Choose --features, --groups or --path-only")
    if args.batch_size is not None and (
        not args.features or args.batch_size < 1 or args.input.suffix.lower() != ".parquet"
    ):
        parser.error("--batch-size requires --features, a positive size and Parquet input")
    if args.output.suffix.lower() != ".parquet":
        parser.error("--output must end in .parquet")
    outputs = [args.output] + ([args.catalog] if args.catalog else [])
    inputs = [args.input] + ([args.schedule] if args.schedule else [])
    if len({path.resolve() for path in [*inputs, *outputs]}) != len(inputs) + len(outputs):
        parser.error("Input, schedule, output and catalog must be different paths")
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Output already exists; choose a new path or --overwrite")
    if args.daily:
        from pfe.data.daily import load_daily_bars
        from pfe.data.sessions import exchange_schedule, normalize_session_schedule
        from pfe.features.daily import compute_daily_features

        try:
            if args.schedule:
                supplied = (
                    pd.read_parquet(args.schedule)
                    if args.schedule.suffix.lower() == ".parquet"
                    else pd.read_csv(args.schedule)
                )
                schedule = normalize_session_schedule(supplied)
            else:
                if args.input.suffix.lower() == ".parquet":
                    labels = pd.read_parquet(args.input, columns=["session_date"])
                elif args.input.suffix.lower() == ".csv":
                    labels = pd.read_csv(args.input, usecols=["session_date"], dtype=str)
                else:
                    parser.error("Daily input must be .csv or .parquet")
                if labels.empty:
                    parser.error(
                        "Daily input must contain canonical session_date and provenance columns"
                    )
                dates = labels.session_date.astype(str)
                schedule = exchange_schedule(dates.min(), dates.max(), calendar=args.calendar)
            bars = load_daily_bars(args.input, schedule=schedule)
            result = compute_daily_features(
                bars,
                schedule=schedule,
                windows=tuple(args.windows),
                groups=args.groups,
                names=args.features,
                gap_policy=args.gap_policy,
            )
            if result.empty:
                parser.error("No daily session bars in the input")
            count, schema = save_batches([result], args.output)
            if args.catalog:
                args.catalog.parent.mkdir(parents=True, exist_ok=True)
                catalog = feature_catalog(schema)
                catalog["availability"] = "explicit available_at"
                catalog.to_csv(args.catalog, index=False)
            print(f"Saved {count:,} daily bars, {len(schema.columns):,} columns: {args.output}")
            return
        except (ValueError, OSError) as error:
            parser.error(str(error))
    origin = args.origin
    if origin not in {"start_day", "epoch", "start"}:
        origin = pd.Timestamp(origin)
        if origin.tzinfo is None:
            parser.error("Explicit --origin must include a timezone")
    options = dict(
        source_frequency=args.source_frequency or "1min",
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
