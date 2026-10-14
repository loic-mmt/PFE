"""Synthetic bounded dataset; no network, optional SDK or DL framework."""

from tempfile import TemporaryDirectory

import pandas as pd

import pfe

decisions = pd.date_range("2026-01-01T22:00:00Z", periods=24, freq="D")
rows = [
    dict(
        entity_id=entity,
        metric="close",
        unit="USD",
        observation_date=day.isoformat(),
        published_at=None,
        available_at=day.isoformat(),
        value=100.0 + i,
        source_id="local",
        revision_id=str(i),
        schema_version="1",
        source_document_id="synthetic:" + str(i),
    )
    for i, day in enumerate(decisions)
    for entity in ("example:A", "example:B")
]
registry = pfe.FeatureRegistry(
    [pfe.FeatureSpec("close", "close", "USD", "local", dependencies=("synthetic_close",))]
)
panel = pfe.build_feature_panel(
    decisions, pd.DataFrame(rows), entity_ids=["example:A", "example:B"], registry=registry
)
labels = pfe.forward_return_labels(panel, price_feature="close", horizon=2)
dataset = pfe.build_sequence_dataset(panel, lookback=4, labels=labels)
split = pfe.temporal_split(
    dataset, train_end="2026-01-11T00:00:00Z", validation_end="2026-01-19T00:00:00Z"
)
scaler = pfe.TrainOnlyScaler.fit(dataset, split)
with TemporaryDirectory(prefix="pfe-model-dataset-") as directory:
    pfe.save_model_dataset(
        directory,
        panel,
        sequences=dataset,
        split=split,
        scaler=scaler,
        configuration={"fixture": True, "decision_grid": "synthetic_daily"},
    )
    arrays, manifest = pfe.load_model_dataset(directory)
    assert arrays["sequence_x"].shape == dataset.x.shape
    assert manifest["panel_fingerprint"] == panel.fingerprint
    print(
        f"GRU/TCN: {dataset.x.shape}; train={len(split.train)}, "
        f"validation={len(split.validation)}, test={len(split.test)}, purged={len(split.purged)}"
    )
