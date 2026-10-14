"""Causal windows, purged label horizons, training-only stats and durable provenance."""

import json

import numpy as np
import pandas as pd
import pytest

from pfe.datasets import (
    DatasetLimits,
    FeatureRegistry,
    TrainOnlyScaler,
    build_feature_panel,
    build_sequence_dataset,
    feature_frame_to_observations,
    forward_return_labels,
    iter_sequence_batches,
    load_model_dataset,
    save_model_dataset,
    temporal_split,
)


def fixture_panel(*, future_outlier=False, count=15, limits=None):
    dates = pd.date_range("2026-01-01T22:00:00Z", periods=count, freq="D")
    rows = []
    for i, day in enumerate(dates):
        for entity in ("A", "B"):
            for metric in ("price", "signal"):
                value = 100 + i if metric == "price" else i
                if future_outlier and i >= 8 and metric == "signal":
                    value = 1e10
                if i == 2 and entity == "B" and metric == "signal":
                    value = None
                rows.append(
                    dict(
                        entity_id=entity,
                        metric=metric,
                        unit="USD" if metric == "price" else "score",
                        observation_date=day.isoformat(),
                        published_at=None,
                        available_at=day.isoformat(),
                        value=value,
                        source_id="fixture",
                        revision_id=str(i),
                        schema_version="1",
                        source_document_id="fixture:" + str(i),
                    )
                )
    registry = FeatureRegistry(
        [
            dict(
                name="price",
                metric="price",
                unit="USD",
                source_id="fixture",
                dependencies=["close"],
            ),
            dict(
                name="signal",
                metric="signal",
                unit="score",
                source_id="fixture",
                window="3 observations",
                dependencies=["raw_signal"],
            ),
        ]
    )
    return build_feature_panel(
        dates, pd.DataFrame(rows), entity_ids=["A", "B"], registry=registry, limits=limits
    )


def test_panel_mask_provenance_and_availability():
    panel = fixture_panel()
    assert panel.values.shape == (15, 2, 2)
    assert not panel.mask[2, 1, 1]
    assert panel.provenance_codes[2, 1, 1] >= 0
    assert (panel.available_at_ns <= panel.decisions_ns[:, None, None]).all()
    assert all(p["source_document_id"].startswith("fixture:") for p in panel.provenance)


def test_sequence_order_horizon_labels_and_missing_separate():
    panel = fixture_panel()
    labels = forward_return_labels(panel, price_feature="price", horizon=2)
    data = build_sequence_dataset(panel, lookback=3, labels=labels)
    assert data.x.shape == (26, 3, 2)
    np.testing.assert_array_equal(data.x[0, :, 0], [100, 101, 102])
    assert data.y[0] == pytest.approx(104 / 102 - 1)
    assert not data.mask[1, -1, 1] and data.x[1, -1, 1] == 0
    assert not data.y_mask[-4:].any()
    assert data.x.shape[-1] == 2  # No future target in input features.


@pytest.mark.parametrize("batch_size", [1, 3, 17, 100])
def test_chunk_batch_equivalence(batch_size):
    panel = fixture_panel()
    labels = forward_return_labels(panel, price_feature="price", horizon=1)
    expected = build_sequence_dataset(panel, lookback=4, labels=labels)
    batches = list(iter_sequence_batches(panel, lookback=4, labels=labels, batch_size=batch_size))
    for name in ("x", "mask", "end_indices", "entity_indices", "y", "y_mask", "label_end_ns"):
        np.testing.assert_array_equal(
            np.concatenate([getattr(b, name) for b in batches]), getattr(expected, name)
        )


def test_purge_boundary_and_training_stats_ignore_validation_outlier():
    panel = fixture_panel()
    data = build_sequence_dataset(
        panel, lookback=3, labels=forward_return_labels(panel, price_feature="price", horizon=2)
    )
    split = temporal_split(
        data, train_end="2026-01-08T22:00:00Z", validation_end="2026-01-12T22:00:00Z"
    )
    assert (data.label_end_ns[split.train] < pd.Timestamp("2026-01-08T22:00:00Z").value).all()
    assert (data.label_end_ns[split.validation] < pd.Timestamp("2026-01-12T22:00:00Z").value).all()
    assert set(split.train).isdisjoint(split.validation)
    scaler = TrainOnlyScaler.fit(data, split)
    changed_panel = fixture_panel(future_outlier=True)
    changed = build_sequence_dataset(
        changed_panel,
        lookback=3,
        labels=forward_return_labels(changed_panel, price_feature="price", horizon=2),
    )
    changed_split = temporal_split(
        changed, train_end="2026-01-08T22:00:00Z", validation_end="2026-01-12T22:00:00Z"
    )
    changed_scaler = TrainOnlyScaler.fit(changed, changed_split)
    np.testing.assert_array_equal(scaler.mean, changed_scaler.mean)
    np.testing.assert_array_equal(scaler.scale, changed_scaler.scale)
    np.testing.assert_array_equal(data.x[:12], changed.x[:12])
    scaled, mask = scaler.transform(data)
    assert np.isfinite(scaled).all() and (scaled[~mask] == 0).all()


def test_empty_training_or_foreign_split_rejected():
    panel = fixture_panel()
    data = build_sequence_dataset(panel, lookback=3)
    split = temporal_split(
        data, train_end="2026-01-08T00:00:00Z", validation_end="2026-01-12T00:00:00Z"
    )
    with pytest.raises(ValueError, match="nonempty"):
        TrainOnlyScaler.fit(data, split)
    other = build_sequence_dataset(panel, lookback=2)
    with pytest.raises(ValueError, match="this dataset"):
        TrainOnlyScaler.fit(other, split)


def test_artifact_idempotency_hash_and_no_pickle(tmp_path):
    panel = fixture_panel()
    data = build_sequence_dataset(
        panel, lookback=3, labels=forward_return_labels(panel, price_feature="price")
    )
    split = temporal_split(
        data, train_end="2026-01-08T00:00:00Z", validation_end="2026-01-12T00:00:00Z"
    )
    scaler = TrainOnlyScaler.fit(data, split)
    one = save_model_dataset(tmp_path, panel, sequences=data, split=split, scaler=scaler)
    two = save_model_dataset(tmp_path, panel, sequences=data, split=split, scaler=scaler)
    assert one == two
    arrays, manifest = load_model_dataset(tmp_path)
    assert all(a.dtype != object for a in arrays.values())
    assert manifest["panel_fingerprint"] == panel.fingerprint
    np.testing.assert_array_equal(arrays["sequence_y"], data.y)
    body = bytearray((tmp_path / "arrays.npz").read_bytes())
    body[-1] ^= 1
    (tmp_path / "arrays.npz").write_bytes(body)
    with pytest.raises(ValueError, match="checksum"):
        load_model_dataset(tmp_path)


@pytest.mark.parametrize(
    "limits",
    [DatasetLimits(max_bytes=100), DatasetLimits(max_entities=1), DatasetLimits(max_times=2)],
)
def test_axis_memory_caps_before_allocating(limits):
    with pytest.raises(ValueError):
        fixture_panel(limits=limits)


def test_feature_bridge_does_not_guess_availability():
    panel = fixture_panel()
    index = pd.MultiIndex.from_tuples(
        [("2026-01-01T22:00:00Z", "A")], names=["decision_at", "entity_id"]
    )
    frame = pd.DataFrame({"price": [100.0], "signal": [1.0]}, index=index)
    known = pd.DataFrame(
        {"price": ["2026-01-02T22:00:00Z"], "signal": ["2026-01-02T22:00:00Z"]}, index=index
    )
    points = feature_frame_to_observations(
        frame, panel.registry, available_at=known, revision_id="run-1"
    )
    before = build_feature_panel(
        ["2026-01-01T22:00:00Z"], points, entity_ids=["A"], registry=panel.registry
    )
    assert not before.mask.any()
    known.iloc[0, 0] = None
    with pytest.raises(ValueError, match="needs availability"):
        feature_frame_to_observations(
            frame, panel.registry, available_at=known, revision_id="run-1"
        )


def test_scaler_rejects_train_validation_index_swap():
    from dataclasses import replace

    panel = fixture_panel()
    data = build_sequence_dataset(
        panel, lookback=2, labels=forward_return_labels(panel, price_feature="price")
    )
    split = temporal_split(
        data, train_end="2026-01-07T00:00:00Z", validation_end="2026-01-11T00:00:00Z"
    )
    swapped = replace(split, train=split.validation, validation=split.train)
    with pytest.raises(ValueError, match="chronological"):
        TrainOnlyScaler.fit(data, swapped)


def test_untrained_feature_stays_masked_in_validation():
    from dataclasses import replace

    panel = fixture_panel()
    values, mask = panel.values.copy(), panel.mask.copy()
    values[:8, :, 1], mask[:8, :, 1] = np.nan, False
    panel = replace(panel, values=values, mask=mask)
    data = build_sequence_dataset(
        panel, lookback=2, labels=forward_return_labels(panel, price_feature="price")
    )
    split = temporal_split(
        data, train_end="2026-01-07T00:00:00Z", validation_end="2026-01-11T00:00:00Z"
    )
    scaler = TrainOnlyScaler.fit(data, split)
    assert scaler.count[1] == 0
    scaled, observed = scaler.transform(data)
    assert not observed[:, :, 1].any() and (scaled[:, :, 1] == 0).all()
    _, panel_mask = scaler.transform_panel(panel)
    assert not panel_mask[:, :, 1].any()


def test_stale_repeated_price_is_not_a_new_future_label():
    from dataclasses import replace

    panel = fixture_panel()
    observed = panel.observation_at_ns.copy()
    observed[1, :, 0] = observed[0, :, 0]
    labels = forward_return_labels(
        replace(panel, observation_at_ns=observed), price_feature="price"
    )
    assert not labels.mask[0].any()


def test_artifact_unknown_npy_allocation_rejected(tmp_path):
    import hashlib
    import io
    import zipfile

    save_model_dataset(tmp_path, fixture_panel())
    original = (tmp_path / "arrays.npz").read_bytes()
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as old, zipfile.ZipFile(stream, "w") as new:
        for name in old.namelist():
            if name == "panel_values.npy":
                header = io.BytesIO()
                np.lib.format.write_array_header_1_0(
                    header, dict(descr="<f8", fortran_order=False, shape=(1_000_000_000,))
                )
                new.writestr(name, header.getvalue())
            else:
                new.writestr(name, old.read(name))
    body = stream.getvalue()
    (tmp_path / "arrays.npz").write_bytes(body)
    metadata = json.loads((tmp_path / "manifest.json").read_text())
    metadata["data_sha256"] = hashlib.sha256(body).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="allocation"):
        load_model_dataset(tmp_path)


def test_panel_instrument_ambiguity_requires_explicit_selection():
    row = dict(
        entity_id="issuer",
        metric="price",
        unit="USD",
        source_id="fixture",
        observation_date="2026-01-01",
        available_at="2026-01-02T00:00:00Z",
        published_at=None,
        value=10,
        revision_id="1",
        schema_version="1",
    )
    rows = pd.DataFrame(
        [
            dict(row, instrument_id="share-A"),
            dict(row, instrument_id="share-B", value=100),
        ]
    )
    spec = dict(name="price", metric="price", unit="USD", source_id="fixture")
    options = dict(entity_ids=["issuer"], registry=[spec])
    for frame in (rows, rows.iloc[::-1]):
        with pytest.raises(ValueError, match="instrument_id"):
            build_feature_panel(["2026-01-03T00:00:00Z"], frame, **options)
        selected = build_feature_panel(
            ["2026-01-03T00:00:00Z"],
            frame,
            entity_ids=["issuer"],
            registry=[dict(spec, instrument_id="share-B")],
        )
        assert selected.values[0, 0, 0] == 100
        assert selected.provenance[0]["instrument_id"] == "share-B"


@pytest.mark.parametrize("flag", ["False", 0, 1])
def test_panel_quality_flag_is_not_coerced_by_truthiness(flag):
    registry = fixture_panel(count=1).registry
    index = pd.MultiIndex.from_tuples(
        [("2026-01-01T22:00:00Z", "A")], names=["decision_at", "entity_id"]
    )
    frame = pd.DataFrame({"price": [100.0], "signal": [1.0]}, index=index)
    known = pd.DataFrame("2026-01-01T22:00:00Z", columns=frame.columns, index=index)
    points = feature_frame_to_observations(frame, registry, available_at=known, revision_id="1")
    points["quality.feature_eligible"] = flag
    with pytest.raises(ValueError, match="booleans"):
        build_feature_panel(["2026-01-01T22:00:00Z"], points, entity_ids=["A"], registry=registry)
    points["quality.feature_eligible"] = pd.Series([False, pd.NA], dtype=object)
    panel = build_feature_panel(
        ["2026-01-01T22:00:00Z"], points, entity_ids=["A"], registry=registry
    )
    assert panel.mask[0, 0].tolist() == [False, True]


@pytest.mark.parametrize("field", ["x", "mask", "decision_ns", "label_available_at_ns"])
def test_sequence_fingerprint_covers_inputs_and_clocks(field):
    data = build_sequence_dataset(
        fixture_panel(),
        lookback=2,
        labels=forward_return_labels(fixture_panel(), price_feature="price"),
    )
    split = temporal_split(
        data, train_end="2026-01-07T00:00:00Z", validation_end="2026-01-11T00:00:00Z"
    )
    previous = data.fingerprint
    array = getattr(data, field)
    array.flat[0] = not array.flat[0] if array.dtype == bool else array.flat[0] + 1
    assert data.fingerprint != previous
    with pytest.raises(ValueError, match="this dataset"):
        TrainOnlyScaler.fit(data, split)


def test_export_rejects_object_arrays_before_writing(tmp_path):
    from dataclasses import replace

    panel = fixture_panel()
    with pytest.raises(ValueError, match="pickle"):
        save_model_dataset(tmp_path / "objects", replace(panel, values=panel.values.astype(object)))
    assert not (tmp_path / "objects").exists()


@pytest.mark.parametrize("field", ["x", "mask", "decision_ns", "window_start_ns"])
def test_export_rejects_sequences_detached_from_panel(tmp_path, field):
    panel = fixture_panel()
    data = build_sequence_dataset(panel, lookback=2)
    array = getattr(data, field)
    array.flat[0] = not array.flat[0] if array.dtype == bool else array.flat[0] + 1
    with pytest.raises(ValueError, match="differ from panel"):
        save_model_dataset(tmp_path / "changed", panel, sequences=data)
    assert not (tmp_path / "changed").exists()
