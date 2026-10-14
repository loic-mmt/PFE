"""No downloads: minute bars, multiple timeframes, global scaler and small shards."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pfe import (
    TimeframeSpec,
    build_timeframe_observations,
    fit_partitioned_scaler,
    iter_partitioned_sequences,
    iter_saved_partitions,
    save_partitioned_dataset,
)

times = pd.date_range("2026-01-01T10:00:00Z", periods=40, freq="min")
prices = 100 + np.arange(len(times), dtype=float)
bars = pd.DataFrame(
    dict(
        open=prices,
        high=prices + 1,
        low=prices - 1,
        close=prices,
        volume=10.0,
        available_at=times + pd.Timedelta("1min"),
    ),
    index=times,
)
features = build_timeframe_observations(
    {"issuer:example": bars},
    [
        TimeframeSpec(
            freq, ("close", "returns.simple"), {"close": "USD", "returns.simple": "fraction"}
        )
        for freq in ("1min", "5min")
    ],
    source_frequency="1min",
    source_id="synthetic",
    revision_id="fixture-1",
)
observations = features.observations.sort_values("available_at", kind="stable")


def partitions():
    batches = (observations.iloc[start : start + 10] for start in range(0, len(observations), 10))
    return iter_partitioned_sequences(
        times + pd.Timedelta("1min"),
        batches,
        entity_ids=["issuer:example"],
        registry=features.registry,
        lookback=4,
        label_feature="tf[1min].close",
        label_horizon=2,
        partition_size=10,
    )


scaler = fit_partitioned_scaler(partitions(), train_end="2026-01-01T10:25:00Z")
with tempfile.TemporaryDirectory(prefix="pfe-reusable-") as temporary:
    output = Path(temporary) / "dataset"
    manifest = save_partitioned_dataset(
        output,
        partitions(),
        scaler=scaler,
        configuration=features.configuration,
        max_archive_bytes=1_048_576,
    )
    sample_count = sum(len(arrays["sequence_x"]) for arrays, _ in iter_saved_partitions(output))
    print(f"{len(manifest['shards'])} shards, {sample_count} samples, no downloads")
