"""Download bounded native daily candles or retain legacy one-minute exports.

Daily requires a declared calendar and bar label. Saved raw snapshots replay
without a key. SDK 1m transfers remain unbounded, with date/size/row checks before
Parquet loading; --resume explicitly enables legacy snapshot replay.
"""

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

try:
    from lse import LSE
except ImportError:
    LSE = None

from pfe import normalize_bars
from pfe.data.lse_transport import LSEDataTransport
from pfe.data.qualification import (
    SDK_CHANGELOG_URL,
    SDK_CLIENT_URL,
    DiscoveryError,
    DiscoveryLimitError,
)
from pfe.data.sources import default_source_registry
from pfe.data.storage import ImmutablePageError, RawStore, RawStoreError


class DownloadError(ValueError):
    """Safe failure without provider bodies or credentials."""


def _tickers(single, multiple):
    values = [*(single or []), *(multiple or [])] or ["QQQ"]
    symbols = [part.strip().upper() for value in values for part in value.split(",")]
    if any(not re.fullmatch(r"[A-Z0-9^][A-Z0-9._^-]{0,63}", symbol) for symbol in symbols):
        raise DownloadError("Tickers must contain letters, digits, '.', '-', '_' or '^'")
    if len(set(symbols)) != len(symbols):
        raise DownloadError("Tickers must be unique")
    return symbols


def _mapping(values, symbols, label):
    result = {}
    for value in values or []:
        parts = value.split("=", 1)
        if len(parts) != 2 or not parts[1].strip():
            raise DownloadError(f"Use --{label} TICKER=VALUE")
        symbol, item = parts[0].strip().upper(), parts[1].strip()
        if symbol not in symbols or symbol in result or any(ord(c) < 32 for c in item):
            raise DownloadError(f"Invalid or duplicate --{label} mapping")
        result[symbol] = item
    return result


def _time(value):
    try:
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")
    except (TypeError, ValueError):
        raise DownloadError("Dates must be valid ISO 8601 values") from None
    if pd.isna(stamp):
        raise DownloadError("Dates must be valid ISO 8601 values")
    return stamp


def _bytes(frame):
    stream = io.BytesIO()
    frame.to_parquet(stream)
    return stream.getvalue()


def _publish(path, body):
    """No overwrite; identical replay leaves mtime intact."""
    if path.exists():
        if path.read_bytes() != body:
            raise ImmutablePageError("Existing output differs from the saved snapshot")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(body)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise ImmutablePageError(
                    "Existing output differs from the saved snapshot"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _receipt():
    return datetime.now(UTC).isoformat()


class Budget:
    """Aggregate original source bytes and new requests across a ticker batch."""

    def __init__(self, max_bytes, max_requests, max_rows):
        self.max_bytes, self.max_requests, self.max_rows = max_bytes, max_requests, max_rows
        self.bytes, self.requests, self.rows = 0, 0, 0

    def request(self):
        if self.requests >= self.max_requests:
            raise DiscoveryLimitError("Download exceeds --max-requests")
        self.requests += 1
        return self.remaining

    @property
    def remaining(self):
        if self.bytes >= self.max_bytes:
            raise DiscoveryLimitError("Download exceeds --max-bytes")
        return self.max_bytes - self.bytes

    def account(self, count):
        self.bytes += count
        if self.bytes > self.max_bytes:
            raise DiscoveryLimitError("Download exceeds --max-bytes")

    @property
    def remaining_rows(self):
        if self.rows >= self.max_rows:
            raise DiscoveryLimitError("Download reaches --max-rows")
        return self.max_rows - self.rows

    def account_rows(self, count):
        self.rows += count
        if self.rows >= self.max_rows:
            raise DiscoveryLimitError("Download reaches --max-rows")


def _windows(start, end, days):
    while start < end:
        stop = min(start + pd.Timedelta(days=days), end)
        yield start, stop
        start = stop


def _daily(args, symbol, start, end, store, collection_id, pages, budget, calendar):
    from pfe.data.daily import normalize_daily_bars
    from pfe.data.sessions import _session_date

    rows, receipts, transport = [], [], None
    schedule = args.schedules[calendar]
    for index, (begin, stop) in enumerate(_windows(start, end, args.window_days)):
        page_id = f"window-{index:06d}"
        if page_id in pages:
            page = json.loads(store.read_page(collection_id, page_id))
            budget.account(page["response_bytes"])
            budget.account_rows(len(page["rows"]))
        else:
            if transport is None:
                key = os.environ.get("LSE_API_KEY", "").strip()
                if not key:
                    raise DownloadError("Set LSE_API_KEY for an uncached download")
                transport = LSEDataTransport(key)
            params = dict(
                symbol=symbol,
                timeframe="1d",
                dataset=args.dataset,
                start=begin.date().isoformat(),
                end=stop.date().isoformat(),
                order="asc",
                limit=args.limit,
            )
            response = transport.get("/candles", params, max_bytes=budget.request())
            page = dict(
                rows=response.payload,
                response_bytes=response.byte_count,
                captured_at=_receipt(),
                request=params,
                row_cap_reached=len(response.payload) >= params["limit"],
            )
            budget.account(page["response_bytes"])
            budget.account_rows(len(page["rows"]))
            store.write_page(
                collection_id,
                page_id,
                page,
                next_cursor=f"window-{index + 1:06d}",
                row_count=len(response.payload),
                metadata={"completion_scope": "bounded_provider_page"},
            )
        if page["row_cap_reached"]:
            raise DiscoveryLimitError(
                "Daily row cap reached; shorten --window-days or raise --limit"
            )
        rows.extend((row, page["captured_at"]) for row in page["rows"])
        receipts.append(page["captured_at"])
    if not rows:
        raise DownloadError(f"No {symbol} daily bars in the requested window")
    # Identical boundary-inclusive repeats deduplicate; conflicting session bars
    # remain and are rejected by normalization.
    unique = {}
    for row, captured in rows:
        key = json.dumps(row, sort_keys=True, allow_nan=False)
        previous = unique.get(key)
        if previous is None or pd.Timestamp(captured) < pd.Timestamp(previous[1]):
            unique[key] = row, captured
    frame = pd.DataFrame([row for row, _ in unique.values()])
    captured = pd.Series([pd.Timestamp(value) for _, value in unique.values()])
    column = "timestamp" if "timestamp" in frame else "ts"
    # Raw response remains intact in RawStore. Only the declared bar fields
    # enter normalization; unexpected provider metadata cannot override scope,
    # identifiers, adjustments or availability declared by this request.
    selected = [column, "open", "high", "low", "close", "volume"]
    if args.daily_label == "session_date" and "session_date" in frame:
        selected.append("session_date")
    frame = frame.loc[:, [name for name in selected if name in frame]]
    if column in frame:
        # SDK _isoify documents these ClickHouse strings as UTC. Preserve raw
        # strings in the snapshot and make only this known format explicitly UTC.
        frame[column] = frame[column].map(
            lambda value: (
                value.replace(" ", "T", 1) + "Z"
                if isinstance(value, str)
                and re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?", value)
                else value
            )
        )
    if args.daily_label == "session_date":
        labels = frame["session_date"] if "session_date" in frame else frame[column]
        dates = labels.map(_session_date)
        keep = (dates >= start.date().isoformat()) & (dates < end.date().isoformat())
    else:
        times = frame[column].map(pd.Timestamp)
        if any(value.tz is None for value in times):
            raise DownloadError(
                "Daily open/close labels require an aware timestamp or SDK UTC format"
            )
        bounds = schedule.open_at if args.daily_label == "open" else schedule.close_at
        lookup = dict(zip(bounds, schedule.index, strict=True))
        dates = times.map(lookup)
        inside_dates = dates.notna()
        keep = (
            inside_dates & (dates >= start.date().isoformat()) & (dates < end.date().isoformat())
        ) | (~inside_dates & (times >= schedule.open_at.min()) & (times <= schedule.close_at.max()))
    frame = frame.loc[keep].reset_index(drop=True)
    captured = captured.loc[keep].reset_index(drop=True)
    dates = dates.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise DownloadError(f"No {symbol} daily bars in the requested window")
    frame["ingested_at"] = captured
    close = dates.map(schedule.close_at)
    provisional = captured < close
    # A live daily quote received before the actual close never becomes a
    # confirmed final bar merely because wall-clock time subsequently advances.
    frame["quality.session_complete"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame.loc[provisional, "quality.session_complete"] = False
    frame["quality.feature_eligible"] = ~provisional
    receipt = max(pd.Timestamp(value) for value in receipts)
    result = normalize_daily_bars(
        frame,
        schedule=schedule,
        timestamp_column=column,
        label=args.daily_label,
        instrument_id=args.identities.get(symbol, f"lse_symbol:{symbol}"),
        entity_id=args.entities.get(symbol),
        source_id="lse",
        currency=args.currency,
        price_adjustment=args.price_adjustment,
        volume_adjustment="unknown",
        session_scope="unknown",
        ingested_at=receipt,
        availability_basis=args.availability_basis,
        latency=args.latency,
    )
    dates = result["session_date"].astype(str)
    result = result.loc[(dates >= start.date().isoformat()) & (dates < end.date().isoformat())]
    if result.empty:
        raise DownloadError(f"No {symbol} daily bars in the requested window")
    expected = {
        str(value)
        for value in schedule.index
        if start.date().isoformat() <= str(value) < end.date().isoformat()
    }
    observed = set(result["session_date"].astype(str))
    return result, dict(
        expected_sessions=len(expected),
        observed_sessions=len(observed),
        missing_sessions=sorted(expected - observed),
        provisional_sessions=sorted(
            result.loc[result["quality.session_complete"].eq(False).fillna(False), "session_date"]
        ),
        provider_completeness="unqualified",
        session_scope="unknown",
        availability_basis=args.availability_basis,
        adjustment_evidence=SDK_CHANGELOG_URL
        if args.price_adjustment == "split_adjusted"
        else None,
        timestamp_timezone_evidence=SDK_CLIENT_URL,
        ingested_at=receipt.isoformat(),
        byte_cap_scope="before_json_decoding",
    )


def _legacy(args, symbol, start, end, raw_dir, store, collection_id, pages, budget):
    if "history" in pages:
        raw = store.read_page(collection_id, "history")
        budget.account(len(raw))
        frame = pd.read_parquet(io.BytesIO(raw))
        budget.account_rows(len(frame))
    else:
        if raw_dir.exists():
            raise DownloadError("Unjournaled SDK export exists; choose a new --snapshot-id")
        if end.ceil("D") - start.normalize() > pd.Timedelta(days=args.max_export_days):
            raise DiscoveryLimitError("Legacy export exceeds --max-export-days")
        key = os.environ.get("LSE_API_KEY", "").strip()
        if not key:
            raise DownloadError("Set LSE_API_KEY (or use uv run --env-file .env)")
        if LSE is None:
            raise DownloadError("Install lse-data for legacy exports")
        remaining = budget.request()
        try:
            with LSE(api_key=key) as client:
                returned = client.history(
                    symbol,
                    timeframe="1m",
                    start=start.date().isoformat(),
                    end=end.ceil("D").date().isoformat(),
                    dest=str(raw_dir),
                    dataframe=False,
                    **({"dataset": args.dataset} if args.dataset else {}),
                )
        except Exception:
            raise DownloadError("LSE SDK history export failed") from None
        if isinstance(returned, pd.DataFrame):
            if len(returned) >= budget.remaining_rows:
                raise DiscoveryLimitError("Legacy export reaches --max-rows")
            frame, raw, origin = returned, _bytes(returned), "sdk_dataframe"
        else:
            path = Path(returned)
            if path.stat().st_size > remaining:
                raise DiscoveryLimitError("SDK export exceeds --max-bytes (checked after transfer)")
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows >= budget.remaining_rows:
                raise DiscoveryLimitError("Legacy export reaches --max-rows")
            raw, origin = path.read_bytes(), "sdk_parquet_export"
            frame = pd.read_parquet(io.BytesIO(raw))
        budget.account(len(raw))
        budget.account_rows(len(frame))
        store.write_page(
            collection_id,
            "history",
            raw,
            row_count=len(frame),
            metadata=dict(
                captured_at=_receipt(),
                origin=origin,
                byte_cap_scope="after_sdk_transfer_before_parquet_loading",
                provider_completeness="unqualified",
            ),
        )
    column = "timestamp" if "timestamp" in frame else "ts"
    bars = normalize_bars(frame, timestamp_column=column, source_timezone="UTC")
    bars = bars.loc[(bars.index >= start) & (bars.index < end)]
    if bars.empty:
        raise DownloadError(f"No {symbol} bars in the requested window")
    return bars, dict(
        provider_completeness="unqualified",
        session_scope="unknown",
        price_adjustment="unknown",
        byte_cap_scope="after_sdk_transfer_before_parquet_loading",
        sdk_evidence=SDK_CLIENT_URL,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", action="append", help="Repeatable/comma-separated; default QQQ")
    parser.add_argument("--tickers", nargs="+", help="Additional symbols; commas accepted")
    parser.add_argument("--start", required=True, help="Inclusive ISO start; daily needs a date")
    parser.add_argument("--end", required=True, help="Exclusive ISO end; daily needs a date")
    parser.add_argument("--timeframe", choices=("1m", "1d"), default="1m")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--calendar", help="Explicit calendar for the declared ticker universe")
    parser.add_argument("--exchange", action="append", help="Explicit TICKER=CALENDAR; repeatable")
    parser.add_argument(
        "--identity", action="append", help="Explicit TICKER=INSTRUMENT_ID; repeatable"
    )
    parser.add_argument("--entity", action="append", help="Explicit TICKER=ENTITY_ID; repeatable")
    parser.add_argument("--dataset", choices=("stocks", "etf"))
    parser.add_argument("--daily-label", choices=("session_date", "open", "close"))
    parser.add_argument("--currency", help="Explicit currency; no inference")
    parser.add_argument(
        "--price-adjustment", choices=("unknown", "split_adjusted"), default="unknown"
    )
    parser.add_argument(
        "--availability-basis", choices=("receipt", "event_time"), default="receipt"
    )
    parser.add_argument("--latency", default="0s")
    parser.add_argument(
        "--snapshot-id", help="Explicit fresh identity; old snapshots stay immutable"
    )
    parser.add_argument("--resume", action="store_true", help="Permit legacy 1m immutable replay")
    parser.add_argument("--window-days", type=int, default=31)
    parser.add_argument("--max-export-days", type=int, default=31)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--max-bytes", type=int, default=8_388_608)
    parser.add_argument("--max-requests", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        symbols = _tickers(args.ticker, args.tickers)
        start, end = _time(args.start), _time(args.end)
        if start >= end:
            raise DownloadError("--start must precede --end")
        limits = (
            "window_days",
            "max_export_days",
            "limit",
            "max_rows",
            "max_bytes",
            "max_requests",
        )
        if any(getattr(args, name) < 1 for name in limits):
            raise DownloadError("Download limits must be positive")
        if args.limit > 5000:
            raise DownloadError("--limit cannot exceed the documented cap 5000")
        if args.currency is not None and not re.fullmatch(r"[A-Z]{3}", args.currency):
            raise DownloadError("--currency must be an explicit uppercase three-letter code")
        if args.snapshot_id and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.snapshot_id
        ):
            raise DownloadError("--snapshot-id must be a safe identifier")
        exchanges = _mapping(args.exchange, symbols, "exchange")
        args.identities = _mapping(args.identity, symbols, "identity")
        args.entities = _mapping(args.entity, symbols, "entity")
        if args.timeframe == "1d":
            if not args.daily_label:
                raise DownloadError(
                    "Daily needs explicit --daily-label; provider labels unqualified"
                )
            if any(not (exchanges.get(symbol) or args.calendar) for symbol in symbols):
                raise DownloadError("Daily needs --calendar or --exchange for every ticker")
            if any(
                not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in (args.start, args.end)
            ):
                raise DownloadError("Daily --start/--end must be YYYY-MM-DD session-date bounds")
            if args.price_adjustment == "split_adjusted" and args.dataset not in ("stocks", "etf"):
                raise DownloadError("SDK split adjustment needs explicit --dataset stocks or etf")
            if pd.isna(pd.Timedelta(args.latency)) or pd.Timedelta(args.latency) < pd.Timedelta(0):
                raise DownloadError("--latency must be a non-negative fixed duration")
            from pfe.data.sessions import exchange_schedule

            args.schedules = {
                calendar: exchange_schedule(args.start, args.end, calendar=calendar)
                for calendar in {exchanges.get(symbol) or args.calendar for symbol in symbols}
            }
        elif not args.resume and not os.environ.get("LSE_API_KEY", "").strip():
            raise DownloadError("Set LSE_API_KEY (or use uv run --env-file .env)")
        window = f"{start.strftime('%Y%m%dT%H%M%SZ')}_{end.strftime('%Y%m%dT%H%M%SZ')}"
        suffix = f"_{args.snapshot_id}" if args.snapshot_id else ""
        outputs = {
            symbol: args.data_dir
            / "processed"
            / f"{symbol}_{args.timeframe}_{window}{suffix}.parquet"
            for symbol in symbols
        }
        raw_dirs = {
            symbol: args.data_dir / "raw" / "lse" / symbol / args.timeframe / f"{window}{suffix}"
            for symbol in symbols
        }
        if (
            args.timeframe == "1m"
            and not args.resume
            and any(outputs[symbol].exists() or raw_dirs[symbol].exists() for symbol in symbols)
        ):
            raise DownloadError("Window already exists; choose a new range/--data-dir or --resume")
        source = default_source_registry().get("lse").to_dict()
        store = RawStore(args.data_dir / "raw" / "lse" / "snapshots", max_page_bytes=args.max_bytes)
        budget, ids = Budget(args.max_bytes, args.max_requests, args.max_rows), []
        for symbol in symbols:
            calendar = exchanges.get(symbol, args.calendar)
            request = dict(
                symbol=symbol,
                timeframe=args.timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
                snapshot_id=args.snapshot_id,
                dataset=args.dataset,
                calendar=calendar,
                daily_label=args.daily_label,
                currency=args.currency,
                instrument_id=args.identities.get(symbol, f"lse_symbol:{symbol}"),
                entity_id=args.entities.get(symbol),
                price_adjustment=args.price_adjustment,
                availability_basis=args.availability_basis,
                latency=args.latency,
                limits={name: getattr(args, name) for name in limits},
            )
            collection_id = store.begin(
                "lse", f"bars_{args.timeframe}", request, source_metadata=source
            )
            manifest = store.recover(collection_id)
            pages = {page["page_id"] for page in manifest["pages"]}
            before_bytes, before_rows = budget.bytes, budget.rows
            if "processed" in pages:
                body = store.read_page(collection_id, "processed")
                summary = json.loads(store.read_page(collection_id, "summary"))
                budget.account(summary["source_bytes"])
                budget.account_rows(summary["source_rows"])
            else:
                if args.timeframe == "1d":
                    bars, coverage = _daily(
                        args,
                        symbol,
                        start,
                        end,
                        store,
                        collection_id,
                        pages,
                        budget,
                        calendar,
                    )
                else:
                    bars, coverage = _legacy(
                        args,
                        symbol,
                        start,
                        end,
                        raw_dirs[symbol],
                        store,
                        collection_id,
                        pages,
                        budget,
                    )
                body = _bytes(bars)
                summary = dict(
                    collection_id=collection_id,
                    source_id="lse",
                    source_revision_id=source["revision_id"],
                    request=request,
                    rows=len(bars),
                    sha256=hashlib.sha256(body).hexdigest(),
                    coverage=coverage,
                    completion_scope="local_normalization_only",
                    historical_point_in_time="unqualified",
                    source_bytes=budget.bytes - before_bytes,
                    source_rows=budget.rows - before_rows,
                )
                store.write_page(collection_id, "summary", summary)
                store.write_page(
                    collection_id,
                    "processed",
                    body,
                    complete=True,
                    row_count=len(bars),
                    metadata={
                        "completion_scope": "local_normalization_only",
                        "provider_completeness": "unqualified",
                    },
                )
            _publish(outputs[symbol], body)
            _publish(
                outputs[symbol].with_suffix(".manifest.json"),
                json.dumps(summary, sort_keys=True, indent=2, allow_nan=False).encode("utf-8"),
            )
            print(f"Saved {summary['rows']:,} {symbol} {args.timeframe} bars: {outputs[symbol]}")
            ids.append(collection_id)
        return ids
    except (DownloadError, DiscoveryError, RawStoreError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
