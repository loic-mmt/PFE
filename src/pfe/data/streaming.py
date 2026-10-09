"""Read bounded Parquet batches, projecting only the OHLCV and time fields."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from pfe.data.bars import OHLCV, normalize_bars


def iter_bars(path: str | Path, *, batch_size: int = 100_000):
    """Yield normalized Parquet OHLCV batches without loading the whole file.

    Accept a timestamp column or the single pandas DatetimeIndex stored in
    Parquet metadata. Files must be globally chronological. Batch normalization
    checks OHLCV; iter_features additionally checks ordering across batches.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    path = Path(path)
    if path.suffix.lower() != ".parquet":
        raise ValueError("Batch input must be Parquet")
    with pq.ParquetFile(path) as file:
        fields = file.schema_arrow.names
        metadata = json.loads((file.schema_arrow.metadata or {}).get(b"pandas", b"{}"))
        indices = metadata.get("index_columns", [])
        timestamps = "timestamp" if "timestamp" in fields else None
        if timestamps is None and len(indices) == 1 and isinstance(indices[0], str):
            timestamps = indices[0]
        if timestamps is None or timestamps not in fields:
            raise ValueError("Parquet requires a timestamp column or a stored DatetimeIndex")
        columns = [timestamps, *OHLCV]
        for batch in file.iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            yield normalize_bars(frame, timestamp_column=timestamps, frequency="1ns")
