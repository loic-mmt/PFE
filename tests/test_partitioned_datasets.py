"""Out-of-core replay preserves window overlap, labels, masks and training stats."""

import numpy as np
import pandas as pd
import pytest

from pfe.datasets import (
    TrainOnlyScaler,
    build_feature_panel,
    build_sequence_dataset,
    forward_return_labels,
    temporal_split,
)
from pfe.datasets.partitioned import (
    fit_partitioned_scaler,
    iter_parquet_observations,
    iter_partitioned_sequences,
    iter_saved_partitions,
    save_partitioned_dataset,
)


def source(*, outlier=False):
    dates = pd.date_range("2026-01-01T12:00:00Z", periods=20, freq="5min")
    records = []
    for i, stamp in enumerate(dates):
        for entity in ("A", "B"):
            for metric in ("close", "signal"):
                value = 100.0 + i if metric == "close" else float(i)
                if outlier and i >= 12 and metric == "signal":
                    value = 1e8
                if i == 3 and entity == "B" and metric == "signal":
                    value = None
                records.append(
                    dict(
                        entity_id=entity,
                        metric=metric,
                        source_id="fixture",
                        unit="USD" if metric == "close" else "score",
                        value=value,
                        observation_date=stamp.isoformat(),
                        available_at=stamp,
                        published_at=None,
                        revision_id="1",
                        schema_version="1",
                    )
                )
    old = records[1].copy()
    old.update(value=77.0, revision_id="2", available_at=dates[10])
    records.append(old)
    frame = pd.DataFrame(records).sort_values("available_at", kind="stable").reset_index(drop=True)
    registry = [
        dict(
            name=metric,
            metric=metric,
            unit="USD" if metric == "close" else "score",
            source_id="fixture",
        )
        for metric in ("close", "signal")
    ]
    return dates, frame, registry


def partitions(*, size=4, batch=7, outlier=False):
    dates, frame, registry = source(outlier=outlier)
    return iter_partitioned_sequences(
        dates,
        (frame.iloc[i : i + batch] for i in range(0, len(frame), batch)),
        entity_ids=["A", "B"],
        registry=registry,
        lookback=4,
        label_feature="close",
        label_horizon=2,
        stride=2,
        partition_size=size,
    )


@pytest.mark.parametrize("size,batch", [(1, 1), (3, 7), (4, 13), (11, 5)])
def test_partition_windows_labels_and_global_indices_match_full(size, batch):
    dates, frame, registry = source()
    panel = build_feature_panel(dates, frame, entity_ids=["A", "B"], registry=registry)
    full = build_sequence_dataset(
        panel,
        lookback=4,
        labels=forward_return_labels(panel, price_feature="close", horizon=2),
        stride=2,
    )
    parts = list(partitions(size=size, batch=batch))
    for field in (
        "x",
        "mask",
        "y",
        "y_mask",
        "decision_ns",
        "label_end_ns",
        "label_available_at_ns",
    ):
        np.testing.assert_array_equal(
            np.concatenate([getattr(p.sequences, field) for p in parts]), getattr(full, field)
        )
    np.testing.assert_array_equal(
        np.concatenate([p.sequences.end_indices + p.global_offset for p in parts]), full.end_indices
    )


def test_global_training_scaler_matches_full_and_ignores_validation_outlier():
    dates, frame, registry = source()
    panel = build_feature_panel(dates, frame, entity_ids=["A", "B"], registry=registry)
    data = build_sequence_dataset(
        panel,
        lookback=4,
        labels=forward_return_labels(panel, price_feature="close", horizon=2),
        stride=2,
    )
    split = temporal_split(data, train_end=dates[11], validation_end=dates[16])
    expected = TrainOnlyScaler.fit(data, split)
    scaler = fit_partitioned_scaler(partitions(), train_end=dates[11])
    other = fit_partitioned_scaler(partitions(size=3, outlier=True), train_end=dates[11])
    np.testing.assert_allclose(scaler.mean, expected.mean, rtol=1e-13)
    np.testing.assert_allclose(scaler.scale, expected.scale, rtol=1e-13)
    np.testing.assert_array_equal(scaler.count, expected.count)
    np.testing.assert_array_equal(scaler.mean, other.mean)


def test_parquet_batches_shard_replay_and_scaler_lineage(tmp_path):
    dates, frame, registry = source()
    path = tmp_path / "observations.parquet"
    frame.to_parquet(path, row_group_size=8)
    batches = list(iter_parquet_observations([path], batch_rows=3))
    assert len(batches) > 1 and all(len(b) <= 3 for b in batches)
    scaler = fit_partitioned_scaler(partitions(size=3), train_end=dates[11])
    manifest = save_partitioned_dataset(tmp_path / "dataset", partitions(size=5), scaler=scaler)
    assert manifest == save_partitioned_dataset(
        tmp_path / "dataset", partitions(size=5), scaler=scaler
    )
    loaded = list(iter_saved_partitions(tmp_path / "dataset"))
    assert len(loaded) == len(manifest["shards"])
    assert all("sequence_x_scaled" in arrays for arrays, _ in loaded)
    for arrays, _ in loaded:
        assert np.isfinite(arrays["sequence_x_scaled"]).all()
    with pytest.raises(ValueError, match="budget|max_bytes"):
        save_partitioned_dataset(tmp_path / "tiny", partitions(), max_archive_bytes=100)
    assert not (tmp_path / "tiny/manifest.json").exists()


def test_unsorted_stream_refused_before_complete_export():
    dates, frame, registry = source()
    with pytest.raises(ValueError, match="sorted"):
        list(
            iter_partitioned_sequences(
                dates,
                [frame.iloc[20:40], frame.iloc[:20]],
                entity_ids=["A", "B"],
                registry=registry,
                lookback=2,
                partition_size=3,
            )
        )
