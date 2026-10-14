"""Cross-entity, same-period ranks and known-at sector membership."""

import numpy as np
import pandas as pd
import pytest

from pfe.features.cross_section import compute_cross_sectional_features


def points():
    return pd.DataFrame(
        [
            dict(
                entity_id=entity,
                metric="signal",
                unit="fraction",
                source_id="fixture",
                revision_id="1",
                schema_version="1",
                published_at=None,
                available_at="2026-01-01T10:00:00Z",
                observation_date="2026-01-01T10:00:00Z",
                value=value,
            )
            for entity, value in (("A", 1.0), ("B", 3.0), ("C", 100.0))
        ]
    )


REGISTRY = [dict(name="s", metric="signal", unit="fraction", source_id="fixture")]


def test_cross_section_manual_values_and_future_invariance():
    frame = points()
    options = dict(entity_ids=["A", "B"], registry=REGISTRY)
    result = compute_cross_sectional_features(["2026-01-01T10:00:00Z"], frame, **options)
    assert result["cross_section.s.universe.rank_pct"].tolist() == [0.5, 1.0]
    assert result["cross_section.s.universe.zscore"].tolist() == [-1.0, 1.0]
    future = frame.iloc[:1].assign(
        value=999.0, revision_id="2", available_at="2026-01-02T10:00:00Z"
    )
    pd.testing.assert_frame_equal(
        result,
        compute_cross_sectional_features(
            ["2026-01-01T10:00:00Z"], pd.concat([frame, future]), **options
        ),
    )


def test_groups_known_at_decision_and_minimum_count():
    groups = pd.DataFrame(
        [
            dict(
                mapping_id=entity,
                entity_id=entity,
                group_id="tech",
                source_id="classification",
                revision_id="1",
                valid_from="2026-01-01T00:00:00Z",
                valid_to=None,
                available_at="2026-01-01T11:00:00Z",
            )
            for entity in ("A", "B")
        ]
    )
    result = compute_cross_sectional_features(
        ["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"],
        points(),
        entity_ids=["A", "B", "C"],
        registry=REGISTRY,
        classifications=groups,
    )
    assert not result["cross_section.s.group.mask"].iloc[:3].any()
    assert result["cross_section.s.group.rank_pct"].iloc[3:5].tolist() == [0.5, 1.0]
    assert not result["cross_section.s.group.mask"].iloc[-1]


def test_mixed_periods_and_constant_group_are_not_invented():
    frame = points().iloc[:2].copy()
    frame.loc[1, "observation_date"] = "2025-12-31T10:00:00Z"
    options = dict(entity_ids=["A", "B"], registry=REGISTRY)
    result = compute_cross_sectional_features(["2026-01-01T10:00:00Z"], frame, **options)
    assert not result["cross_section.s.universe.mask"].any()
    frame["observation_date"] = "2026-01-01T10:00:00Z"
    frame["value"] = 1.0
    result = compute_cross_sectional_features(["2026-01-01T10:00:00Z"], frame, **options)
    assert np.isnan(result["cross_section.s.universe.zscore"]).all()
    assert result["cross_section.s.universe.rank_pct"].tolist() == [0.75, 0.75]
    with pytest.raises(ValueError):
        compute_cross_sectional_features(["2026-01-01T10:00:00Z"], frame, min_count=1, **options)
