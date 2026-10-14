"""Chronological splits, label-horizon purging and train-only scaling."""

from dataclasses import dataclass

import numpy as np

from pfe.data.contracts import _instant
from pfe.datasets.core import NAT, digest_metadata


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    dataset_fingerprint: str
    boundaries: dict


def temporal_split(dataset, *, train_end, validation_end, test_end=None, embargo="0s"):
    """Half-open decision intervals; labels must be known strictly before boundary.

    Training labels ending at/after validation start are purged. Validation is
    purged at test start too. Embargo is applied after each preceding boundary.
    Lookback inputs may include earlier history, as in an online prediction.
    """
    import pandas as pd

    first, second = (
        _instant(train_end, "train_end").value,
        _instant(validation_end, "validation_end").value,
    )
    last = _instant(test_end, "test_end").value if test_end is not None else np.iinfo(np.int64).max
    gap = pd.Timedelta(embargo)
    if not first < second < last or pd.isna(gap) or gap < pd.Timedelta(0):
        raise ValueError("Split requires ordered boundaries and nonnegative embargo")
    known = dataset.y_mask & (dataset.label_end_ns != NAT) & (dataset.label_available_at_ns != NAT)
    now = dataset.decision_ns
    train = (
        known
        & (now < first)
        & (dataset.label_end_ns < first)
        & (dataset.label_available_at_ns < first)
    )
    validation = (
        known
        & (now >= first + gap.value)
        & (now < second)
        & (dataset.label_end_ns < second)
        & (dataset.label_available_at_ns < second)
    )
    test = (
        known
        & (now >= second + gap.value)
        & (now < last)
        & (dataset.label_end_ns < last)
        & (dataset.label_available_at_ns < last)
    )
    assigned = train | validation | test
    return TemporalSplit(
        *(np.flatnonzero(v) for v in (train, validation, test, ~assigned)),
        dataset.fingerprint,
        dict(
            train_end=str(train_end),
            validation_end=str(validation_end),
            test_end=str(test_end) if test_end is not None else None,
            embargo=str(gap),
        ),
    )


@dataclass(frozen=True)
class TrainOnlyScaler:
    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray
    registry_fingerprint: str
    training_fingerprint: str

    @classmethod
    def fit(cls, dataset, split):
        if not isinstance(split, TemporalSplit) or split.dataset_fingerprint != dataset.fingerprint:
            raise ValueError("Scaler requires the split produced for this dataset")
        groups = (split.train, split.validation, split.test, split.purged)
        expected = temporal_split(dataset, **split.boundaries)
        if any(
            not np.array_equal(getattr(expected, name), getattr(split, name))
            for name in ("train", "validation", "test", "purged")
        ):
            raise ValueError("Split indices disagree with chronological/purged boundaries")
        joined = np.concatenate(groups)
        if (
            not len(split.train)
            or len(joined) != len(dataset.x)
            or len(np.unique(joined)) != len(dataset.x)
            or np.any(joined < 0)
            or np.any(joined >= len(dataset.x))
        ):
            raise ValueError("Scaler requires nonempty training and a disjoint complete split")
        # Deduplicate overlapping window cells, preventing stride/window weighting.
        unique, values = {}, []
        for sample in split.train:
            end, entity = dataset.end_indices[sample], dataset.entity_indices[sample]
            for offset in range(dataset.lookback):
                key = (int(end - dataset.lookback + 1 + offset), int(entity))
                if key not in unique:
                    unique[key] = len(values)
                    values.append(
                        np.where(dataset.mask[sample, offset], dataset.x[sample, offset], np.nan)
                    )
        matrix = np.asarray(values)
        mask = np.isfinite(matrix)
        count = mask.sum(axis=0)
        mean = np.divide(
            np.where(mask, matrix, 0).sum(axis=0),
            count,
            out=np.zeros(matrix.shape[1]),
            where=count > 0,
        )
        delta = np.where(mask, matrix - mean, 0)
        variance = np.divide(
            (delta * delta).sum(axis=0), count, out=np.zeros(matrix.shape[1]), where=count > 0
        )
        scale = np.sqrt(variance)
        scale[scale == 0] = 1
        return cls(
            mean,
            scale,
            count,
            dataset.registry_fingerprint,
            digest_metadata({"dataset": dataset.fingerprint, "train": split.train.tolist()}),
        )

    def transform(self, dataset):
        if dataset.registry_fingerprint != self.registry_fingerprint or dataset.x.shape[-1] != len(
            self.mean
        ):
            raise ValueError("Scaler feature registry differs from dataset")
        mask = dataset.mask & (self.count > 0)[None, None, :]
        transformed = np.where(mask, (dataset.x - self.mean) / self.scale, 0.0)
        return transformed, mask

    def to_dict(self):
        return dict(
            mean=self.mean.tolist(),
            scale=self.scale.tolist(),
            count=self.count.tolist(),
            registry_fingerprint=self.registry_fingerprint,
            training_fingerprint=self.training_fingerprint,
            fit_scope="unique_training_window_cells",
        )

    def transform_panel(self, panel):
        """Apply training statistics to GNN node panels with the same registry."""
        if panel.registry.fingerprint != self.registry_fingerprint:
            raise ValueError("Scaler feature registry differs from panel")
        mask = panel.mask & (self.count > 0)[None, None, :]
        return np.where(mask, (panel.values - self.mean) / self.scale, 0.0), mask
