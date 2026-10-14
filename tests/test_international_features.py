"""Sparse foreign factors, exact periods, known graph links and receipt clocks."""

import numpy as np
import pandas as pd
import pytest

from pfe import (
    EntityRegistry,
    InternationalSeries,
    build_graph_history,
    compute_gdelt_features,
    compute_international_features,
    compute_linked_international_features,
    feature_catalog,
    merge_graph_declarations,
    normalize_observations,
)
from pfe.data.international_sources import _point, parse_gdelt_articles


def observation(period, value, receipt="2026-09-01T12:00:00Z", revision=None):
    return _point(
        (revision or period).encode(),
        received_at=receipt,
        source="tsmc",
        entity_id="tsmc",
        metric="revenue",
        unit="million_TWD",
        period=period,
        frequency="monthly",
        value=value,
        country="TW",
        currency="TWD",
        evidence="fixture",
    )


def series(**kwargs):
    return [
        InternationalSeries("tsmc", "tsmc", "revenue", "million_TWD", "tsmc", "monthly", **kwargs)
    ]


def test_exact_period_yoy_and_later_old_revision_does_not_change_past():
    points = [observation("2025-08", 100), observation("2026-07", 120), observation("2026-08", 150)]
    frame = normalize_observations(pd.DataFrame(points))
    decisions = ["2026-09-01T11:59:59Z", "2026-09-01T12:00:00Z", "2026-09-03T00:00:00Z"]
    before = compute_international_features(decisions, frame, series=series())
    assert np.isnan(before.iloc[0]["international.tsmc.level"])
    assert before.iloc[1]["international.tsmc.pct_change"] == pytest.approx(0.25)
    assert before.iloc[1]["international.tsmc.yoy"] == pytest.approx(0.5)
    future = observation("2026-07", 130, "2026-09-02T12:00:00Z", "revised")
    after = compute_international_features(
        decisions, pd.DataFrame([*points, future]), series=series()
    )
    pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])
    assert after.iloc[2]["international.tsmc.level"] == 150
    assert after.iloc[2]["international.tsmc.pct_change"] == pytest.approx(150 / 130 - 1)


def test_missing_month_never_becomes_previous_month_and_null_revision_masks():
    points = [observation("2026-06", 100), observation("2026-08", 150)]
    decisions = ["2026-09-01T12:00:00Z", "2026-09-03T00:00:00Z"]
    points.append(observation("2026-08", None, "2026-09-02T12:00:00Z", "null_revision"))
    output = compute_international_features(decisions, pd.DataFrame(points), series=series())
    assert np.isnan(output.iloc[0]["international.tsmc.pct_change"])
    assert np.isnan(output.iloc[1]["international.tsmc.level"])
    assert output.iloc[1]["international.tsmc.mask.available"] == 1
    assert output.iloc[1]["international.tsmc.mask.value"] == 0


def test_stale_source_and_economic_age_masks_separate():
    point = observation("2026-01", 100)
    output = compute_international_features(
        ["2026-09-01T12:00:00Z"], pd.DataFrame([point]), series=series(max_observation_age="90D")
    )
    assert output.iloc[0]["international.tsmc.mask.available"] == 1
    assert output.iloc[0]["international.tsmc.mask.fresh"] == 0
    assert np.isnan(output.iloc[0]["international.tsmc.level"])
    assert np.isnan(output.iloc[0]["international.tsmc.publication_age_seconds"])


def test_wrong_units_source_frequency_and_future_economic_period_do_not_join():
    point = observation("2026-08", 100)
    point["unit"] = "thousand_TWD"
    output = compute_international_features(
        ["2026-09-01T12:00:00Z"], pd.DataFrame([point]), series=series()
    )
    assert output.iloc[0]["international.tsmc.mask.available"] == 0
    point = observation("2026-12", 100)
    output = compute_international_features(
        ["2026-09-01T12:00:00Z"], pd.DataFrame([point]), series=series()
    )
    assert output.iloc[0]["international.tsmc.mask.available"] == 0


def test_graph_link_only_after_evidence_known_no_sector_guess():
    registry = EntityRegistry(
        [
            {
                "alias": name,
                "entity_id": name,
                "node_type": kind,
                "available_at": "2026-01-01T00:00:00Z",
            }
            for name, kind in (("us_stock", "company"), ("unlinked", "company"), ("tsmc", "factor"))
        ]
    )
    graph = build_graph_history([], registry)
    declaration = {
        "entity_id": "us_stock",
        "metric": "graph.factor_exposure.tsmc",
        "unit": "relationship",
        "observation_date": "2026-01-01",
        "published_at": None,
        "available_at": "2026-09-02T00:00:00Z",
        "value": {
            "target_entity_id": "tsmc",
            "relation_type": "factor_exposure",
            "active": True,
            "weight_pct": None,
        },
        "source_id": "fixture",
        "revision_id": "1",
        "schema_version": "1",
        "source_document_id": "doc:proof",
        "evidence_quote": "explicit factor exposure",
    }
    graph = merge_graph_declarations(graph, pd.DataFrame([declaration]))
    output = compute_linked_international_features(
        ["2026-09-01T12:00:00Z", "2026-09-02T00:00:00Z"],
        pd.DataFrame([observation("2026-08", 150)]),
        graph,
        series=series(),
        entity_ids=["us_stock", "unlinked"],
    )
    assert output.iloc[0]["international.tsmc.mask.link"] == 0
    assert output.iloc[2]["international.tsmc.level"] == 150
    assert output.iloc[2]["international.tsmc.link_source_documents"] == "doc:proof"
    assert output.iloc[3]["international.tsmc.mask.link"] == 0
    assert np.isnan(output.iloc[3]["international.tsmc.level"])


def test_gdelt_recapture_dedup_and_detection_not_publication():
    body = b'{"articles":[{"url":"https://example.org/a","seendate":"20260901T120000Z","title":"Fixture"}]}'
    kwargs = dict(
        query="fixture", start="2026-09-01T00:00:00Z", end="2026-09-02T00:00:00Z", entity_id="news"
    )
    first = parse_gdelt_articles(body, received_at="2026-09-02T00:00:00Z", **kwargs)
    second = parse_gdelt_articles(body + b" ", received_at="2026-09-04T00:00:00Z", **kwargs)
    observations = pd.concat([first, second], ignore_index=True)
    result = compute_gdelt_features(
        ["2026-09-01T23:00:00Z", "2026-09-04T12:00:00Z"],
        observations,
        entity_id="news",
        windows=("1D", "7D"),
    )
    assert result.iloc[0]["gdelt.article_detections.count_7D"] == 0
    assert result.iloc[0]["gdelt.mask.source_available"] == 0
    assert result.iloc[1]["gdelt.mask.source_available"] == 1
    assert result.iloc[1]["gdelt.article_detections.count_7D"] == 1
    assert result.iloc[1]["gdelt.article_detections.count_1D"] == 0
    assert result.iloc[1]["gdelt.quality.verified_events"] == 0


@pytest.mark.parametrize("age", ["-1D", "NaT"])
def test_invalid_freshness_rejected(age):
    with pytest.raises(ValueError):
        series(max_age=age)


def test_catalog_international_family():
    assert "international" in set(feature_catalog().family)
