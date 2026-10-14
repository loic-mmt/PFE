"""Cboe VIX CSV history; index points, economic dates, unknown historical delivery.

Official source links: https://www.cboe.com/tradable_products/vix/vix_historical_data
Public access does not confer a general ML/commercial/redistribution licence:
https://www.cboe.com/terms and https://www.cboe.com/use-of-content .
"""

import csv
import hashlib
import io
from datetime import datetime

import pandas as pd

from pfe.data.contracts import _instant
from pfe.data.eia import (
    ContextError,
    ContextLimitError,
    _collect_file,
    _observation,
    _observations,
)

CBOE_VIX_URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
CBOE_HISTORY_URL = "https://www.cboe.com/tradable_products/vix/vix_historical_data"


def parse_cboe_vix_csv(
    body,
    *,
    received_at,
    metrics=("CLOSE",),
    date_column="DATE",
    date_format="%m/%d/%Y",
    entity_id="VIX",
    revision_id=None,
    availability_basis="receipt",
    close_schedule=None,
    latency="0s",
    published_at=None,
    max_rows=25_000,
):
    """Parse selected observed index levels without treating DATE as publication.

    receipt is the default. publication requires an explicit aware timestamp for
    this particular file version. reconstructed requires a caller-supplied map
    of economic dates to aware close instants; it remains non-PIT and does not
    assume stock-exchange closing times are VIX closing times.
    No missing level, holiday, unit conversion or executable price is invented.
    """
    _instant(received_at, "received_at")
    if not isinstance(body, bytes):
        raise ContextError("Cboe CSV requires bytes")
    if type(max_rows) is not int or max_rows < 1:
        raise ContextError("Cboe max_rows must be positive")
    if isinstance(metrics, str):
        metrics = (metrics,)
    if (
        not metrics
        or len(set(metrics)) != len(metrics)
        or set(metrics) - {"OPEN", "HIGH", "LOW", "CLOSE"}
    ):
        raise ContextError("Choose unique exact Cboe OPEN/HIGH/LOW/CLOSE columns")
    if availability_basis not in ("receipt", "publication", "reconstructed"):
        raise ContextError("Unsupported Cboe availability policy")
    delay = pd.Timedelta(latency)
    if pd.isna(delay) or delay < pd.Timedelta(0):
        raise ContextError("Cboe latency requires a non-negative fixed duration")
    if availability_basis == "reconstructed" and not isinstance(close_schedule, dict):
        raise ContextError("Reconstructed Cboe timing needs an explicit date/aware-close map")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeError:
        raise ContextError("Cboe CSV must be UTF-8") from None
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames
    if (
        not header
        or len(set(header)) != len(header)
        or not {date_column, *metrics}.issubset(header)
    ):
        raise ContextError("Cboe CSV misses exact selected columns or has duplicate headers")
    revision = revision_id or hashlib.sha256(body).hexdigest()
    records = []
    for index, row in enumerate(reader):
        if index >= max_rows or (index + 1) * len(metrics) > max_rows:
            raise ContextLimitError("Cboe selected observations exceed max_rows")
        if None in row or any(value is None for value in row.values()):
            raise ContextError("Malformed Cboe CSV row")
        try:
            day = datetime.strptime(row[date_column], date_format).date().isoformat()
        except (TypeError, ValueError):
            raise ContextError("Cboe DATE does not match the explicit date format") from None
        for metric in metrics:
            record = _observation(
                entity_id=entity_id,
                metric=metric.lower(),
                period_start=day,
                period_end=day,
                period_label=row[date_column],
                value=row[metric],
                unit="index_points",
                source_id="cboe",
                revision_id=revision,
                received_at=received_at,
                published_at=published_at,
                availability_basis="receipt"
                if availability_basis == "reconstructed"
                else availability_basis,
                frequency="daily",
                calendar="provider_index_observation_dates",
            )
            if availability_basis == "reconstructed":
                if day not in close_schedule:
                    raise ContextError("Cboe date missing from the declared close schedule")
                available = _instant(close_schedule[day], "declared_close_at") + delay
                if record["published_at"] is not None:
                    available = max(available, record["published_at"])
                record.update(available_at=available, availability_basis="reconstructed")
            records.append(record)
    return _observations(records)


def collect_cboe_vix(
    transport,
    store,
    *,
    snapshot_id,
    limits=None,
    source_metadata=None,
    offline=None,
    **parser_options,
):
    """Collect only the linked official VIX file; full markets remain unqualified."""
    return _collect_file(
        transport,
        store,
        source_id="cboe",
        dataset="vix_csv_history",
        url=CBOE_VIX_URL,
        snapshot_id=snapshot_id,
        options=parser_options,
        parser=parse_cboe_vix_csv,
        limits=limits,
        source_metadata=source_metadata,
        offline=offline,
    )
