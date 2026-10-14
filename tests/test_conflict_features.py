"""Small UCDP-format records: delayed releases, revisions, coverage and graph weights."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_graph_events import make_fact

from pfe import (
    CaptureLimits,
    ConflictFeatureLimits,
    HTTPResponse,
    RawStore,
    UCDPQuery,
    build_graph_history,
    collect_ucdp,
    compute_conflict_features,
    compute_graph_conflict_features,
    context_feature_observations,
    normalize_conflict_observations,
    parse_ucdp,
    run_pipeline,
)
from pfe.datasets import build_feature_panel


def event(identifier=1, conflict=10, country=678, day="2026-01-02", deaths=6, **changes):
    return (
        dict(
            id=identifier,
            conflict_new_id=conflict,
            country_id=country,
            region="Middle East",
            type_of_violence=1,
            date_start=day,
            date_end=day,
            best=deaths,
            low=deaths,
            high=deaths,
        )
        | changes
    )


def payload(rows, total=None, pages=1):
    return json.dumps(
        dict(Result=rows, TotalCount=len(rows) if total is None else total, TotalPages=pages)
    ).encode()


QUERY = UCDPQuery("gedevents", "26.1", "2026-01-01", "2026-03-31")


def test_quarterly_candidate_version_is_supported():
    query = UCDPQuery("gedevents", "26.01.26.06", "2026-01-01", "2026-06-30")
    frame = parse_ucdp(payload([event()]), query=query, received_at="2026-07-01T12:00Z")
    assert frame.provider_version.tolist() == ["26.01.26.06"]


def parse(rows, receipt="2026-01-03T12:00Z", version="26.1"):
    return parse_ucdp(
        payload(rows),
        query=UCDPQuery("gedevents", version, QUERY.start, QUERY.end),
        received_at=receipt,
    )


def test_known_boundary_counts_conflicts_not_events_and_distinct_violence_types():
    frame = parse(
        [
            event(),
            event(2),
            event(3, conflict=20, type_of_violence=2),
            event(4, conflict=30, type_of_violence=3),
        ]
    )
    out = compute_conflict_features(
        ["2026-01-03T11:59:59Z", "2026-01-03T12:00Z"],
        frame,
        countries=(678,),
        regions=("Middle East",),
    )
    name = "conflicts.global.state_based.active_conflicts_30D"
    assert np.isnan(out[name].iloc[0])
    assert out[name].iloc[1] == 1
    assert out["conflicts.global.state_based.events_30D"].iloc[1] == 2
    assert out["conflicts.global.state_based.fatalities_best_30D"].iloc[1] == 12
    assert out["conflicts.global.non_state.active_conflicts_30D"].iloc[1] == 1
    assert out["conflicts.global.one_sided.active_conflicts_30D"].iloc[1] == 1
    assert out["conflicts.country[678].state_based.active_conflicts_30D"].iloc[1] == 1
    assert out["conflicts.quality.complete_global_universe"].eq(0).all()
    assert out["conflicts.annual.wars"].isna().all()
    assert out["conflicts.global.observation_age_days"].iloc[1] == 1
    assert out["conflicts.global.age_seconds"].iloc[1] == 0


def test_future_revision_date_correction_and_withdrawal_never_rewrite_past():
    original = parse([event()])
    corrected = parse(
        [event(day="2026-01-04", deaths=10)], receipt="2026-01-05T12:00Z", version="26.2"
    )
    decisions = ["2026-01-03T12:00Z", "2026-01-05T12:00Z"]
    baseline = compute_conflict_features(decisions, original)
    revised = compute_conflict_features(decisions, pd.concat([original, corrected]))
    pd.testing.assert_series_equal(baseline.iloc[0], revised.iloc[0])
    assert revised["conflicts.global.state_based.events_30D"].iloc[-1] == 1
    assert revised["conflicts.global.state_based.fatalities_best_30D"].iloc[-1] == 10
    withdrawal = corrected.copy()
    withdrawal["available_at"] = pd.Timestamp("2026-01-06T12:00Z")
    withdrawal["ingested_at"] = withdrawal.available_at
    withdrawal["revision_id"] = "withdrawn"
    withdrawal.at[withdrawal.index[0], "value"] = dict(withdrawal.iloc[0].value, active=False)
    out = compute_conflict_features(
        ["2026-01-06T12:00Z"], pd.concat([original, corrected, withdrawal])
    )
    assert out["conflicts.global.state_based.active_conflicts_30D"].iloc[0] == 0


def test_activity_window_changes_and_newly_observed_is_not_actual_onset():
    frame = parse(
        [event(day="2026-01-02"), event(2, conflict=20, day="2026-02-01")],
        receipt="2026-02-02T12:00Z",
    )
    out = compute_conflict_features(["2026-02-02T12:00Z"], frame)
    prefix = "conflicts.global.state_based."
    assert out[prefix + "active_conflicts_30D"].iloc[0] == 1
    assert out[prefix + "events_change_30D"].iloc[0] == 0
    assert out[prefix + "newly_observed_conflicts_90D"].iloc[0] == 2


def test_annual_wars_are_unavailable_until_release_and_do_not_create_event_zeros():
    query = UCDPQuery("ucdpprioconflict", "26.1", "2025-01-01", "2025-12-31")
    frame = parse_ucdp(
        payload(
            [
                dict(conflict_id=10, year=2025, intensity_level=2),
                dict(conflict_id=20, year=2025, intensity_level=1),
            ]
        ),
        query=query,
        received_at="2026-06-01T12:00Z",
        published_at="2026-06-01T10:00Z",
    )
    out = compute_conflict_features(["2025-12-31T23:00Z", "2026-06-01T12:00Z"], frame)
    assert np.isnan(out["conflicts.annual.wars"].iloc[0])
    assert out["conflicts.annual.wars"].iloc[1] == 1
    assert out["conflicts.annual.armed_conflicts"].iloc[1] == 2
    assert out["conflicts.annual.year"].iloc[1] == 2025
    assert out["conflicts.global.state_based.events_30D"].isna().all()


def test_capture_one_page_replay_caps_empty_query_and_token_is_not_persisted(tmp_path):
    calls = []

    class Transport:
        def get(self, url, *, params, headers, max_bytes):
            calls.append(params)
            assert headers == {"x-ucdp-access-token": "test-token"}
            assert params["StartDate"] == QUERY.start and params["pagesize"] == 1
            return HTTPResponse(payload([event()], total=2, pages=2), "2026-01-03T12:00Z")

    store = RawStore(tmp_path / "store")
    first = collect_ucdp(
        QUERY,
        store,
        snapshot_id="sample",
        api_token="test-token",
        transport=Transport(),
        limits=CaptureLimits(max_rows=1),
    )
    assert not first.complete and first.issues == ("ucdp_query_pages_incomplete",)
    replay = collect_ucdp(QUERY, store, snapshot_id="sample", limits=CaptureLimits(max_rows=1))
    assert replay.collection_id == first.collection_id and len(calls) == 1
    assert all(b"test-token" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())
    empty = collect_ucdp(
        QUERY, store, snapshot_id="empty", offline=(payload([]), "2026-01-03T12:00Z")
    )
    out = compute_conflict_features(["2026-01-03T11:00Z", "2026-01-03T12:00Z"], empty)
    assert np.isnan(out["conflicts.global.state_based.events_30D"].iloc[0])
    assert out["conflicts.global.state_based.events_30D"].iloc[1] == 0
    assert out["conflicts.quality.complete_supplied_query"].iloc[1] == 1
    with pytest.raises(ValueError, match="response budget"):
        collect_ucdp(
            QUERY,
            store,
            snapshot_id="oversized",
            offline=(payload([event()]), "2026-01-03T12:00Z"),
            limits=CaptureLimits(max_response_bytes=10),
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(best=-1),
        dict(low=7, high=8),
        dict(type_of_violence=4),
        dict(country_id=0),
        dict(date_end="2026-04-01"),
        dict(date_start="2026-01-03"),
        dict(id=True),
    ],
)
def test_invalid_ucdp_records_rejected(change):
    with pytest.raises(ValueError):
        parse([event(**change)])


def test_conflict_budgets_and_missing_fatalities_remain_unknown():
    frame = parse([event(best=None, low=None, high=None)])
    out = compute_conflict_features(["2026-01-03T12:00Z"], frame)
    assert np.isnan(out["conflicts.global.state_based.fatalities_best_30D"].iloc[0])
    assert out["conflicts.global.state_based.fatalities_coverage_30D"].iloc[0] == 0
    with pytest.raises(ValueError, match="budget"):
        compute_conflict_features(
            ["2026-01-03T12:00Z"], frame, limits=ConflictFeatureLimits(max_output_cells=1)
        )
    with pytest.raises(ValueError, match="whole-day"):
        compute_conflict_features(["2026-01-03T12:00Z"], frame, windows=("1h",))
    with pytest.raises(ValueError, match="simultaneous|Duplicate"):
        normalize_conflict_observations(pd.concat([frame, frame]))


def test_graph_propagates_only_known_country_exposure_and_preserves_partial_weights(tmp_path):
    fact, _ = make_fact(
        tmp_path,
        task="B",
        text="Acme derives 50% of revenue from Yemen in FY2025.",
        record=dict(
            company="Acme",
            country_or_region="Yemen",
            exposure_type="revenue",
            exposure_percentage=50,
            fiscal_period="FY2025",
        ),
    )
    entities = [
        dict(alias="Acme", entity_id="A", node_type="company", available_at="2026-01-01T00:00Z"),
        dict(
            alias="Yemen",
            entity_id="country:YE",
            node_type="country",
            available_at="2026-01-01T00:00Z",
        ),
    ]
    history = build_graph_history([fact], entities)
    out = compute_graph_conflict_features(
        ["2026-01-03T12:00Z"],
        parse([event()]),
        history,
        entity_ids=["A"],
        country_nodes={678: "country:YE"},
    )
    assert out["graph.conflicts.revenue.active_conflicts_weighted_30D"].iloc[0] == 0.5
    assert out["graph.conflicts.revenue.fatalities_best_weighted_30D"].iloc[0] == 3
    assert np.isnan(out["graph.conflicts.production.active_conflicts_weighted_30D"].iloc[0])
    assert out["graph.conflicts.revenue.known_weight_sum"].iloc[0] == 0.5


def test_context_to_dataset_and_offline_pipeline_replay(tmp_path):
    decisions = ["2026-01-03T12:00Z", "2026-01-04T12:00Z"]
    steps = [
        dict(
            name="politics",
            operation="political_features",
            parameters=dict(
                decisions=decisions,
                observation_options=dict(
                    entity_ids=["A", "B"],
                    units={"politics.us.cycle_year": "ordinal"},
                    source_id="local",
                    revision_id="config-1",
                ),
            ),
            outputs={"observations": "macro/politics.parquet"},
        )
    ]
    first = run_pipeline(steps, tmp_path / "run")
    second = run_pipeline(steps, tmp_path / "run")
    assert second["requests"] == 0 and first["manifest"] == second["manifest"]
    observations = first["results"]["politics"].tables["observations"]
    panel = build_feature_panel(
        decisions,
        observations,
        entity_ids=["A", "B"],
        registry=[
            dict(name="cycle", metric="politics.us.cycle_year", unit="ordinal", source_id="local")
        ],
    )
    assert panel.mask.all() and (panel.values == 2).all()
    features = first["results"]["politics"].tables["features"]
    with pytest.raises(ValueError, match="budget"):
        context_feature_observations(
            features,
            units={"politics.us.cycle_year": "ordinal"},
            source_id="local",
            revision_id="1",
            entity_ids=["A"],
            max_cells=1,
        )


def test_documented_example_keeps_historical_occurrence_and_today_receipt_distinct(tmp_path):
    body = (Path(__file__).parent / "fixtures/context/ucdp_api_example.json").read_bytes()
    query = UCDPQuery("gedevents", "26.1", "2010-09-01", "2010-09-30")
    capture = collect_ucdp(
        query, RawStore(tmp_path), snapshot_id="docs-example", offline=(body, "2026-10-14T12:00Z")
    )
    row = capture.observations.iloc[0]
    assert row.observation_date == "2010-09-25"
    assert row.available_at == pd.Timestamp("2026-10-14T12:00Z")
    assert pd.isna(row.published_at) and not row.point_in_time and not capture.complete


def test_capture_embargo_does_not_expose_empty_counts_before_publication(tmp_path):
    capture = collect_ucdp(
        QUERY,
        RawStore(tmp_path),
        snapshot_id="embargo",
        offline=(payload([event()]), "2026-01-03T12:00Z"),
        published_at="2026-01-04T12:00Z",
    )
    out = compute_conflict_features(["2026-01-03T12:00Z", "2026-01-04T12:00Z"], capture)
    assert out["conflicts.mask.source_available"].tolist() == [0, 1]
    assert np.isnan(out["conflicts.global.state_based.events_30D"].iloc[0])
    assert out["conflicts.global.state_based.events_30D"].iloc[1] == 1


def test_registered_ucdp_and_context_pipeline_handlers_with_mocked_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("UCDP_API_TOKEN", "pipeline-test-token")

    class Transport:
        def get(self, url, *, params, headers, max_bytes):
            assert headers["x-ucdp-access-token"] == "pipeline-test-token"
            assert url.endswith("/gedevents/26.1") and params["page"] == 1
            return HTTPResponse(payload([event()]), "2026-01-03T12:00Z")

    decisions = ["2026-01-03T12:00Z"]
    steps = [
        dict(
            name="source",
            operation="collect",
            parameters=dict(
                kind="ucdp",
                options=dict(
                    query=dict(
                        resource="gedevents", version="26.1", start="2026-01-01", end="2026-01-31"
                    ),
                    snapshot_id="mock-transport",
                ),
            ),
        ),
        dict(
            name="conflicts",
            operation="conflict_features",
            needs=["source"],
            parameters=dict(
                decisions=decisions,
                observations="source.observations",
                options=dict(countries=[678]),
            ),
        ),
        dict(name="calendar", operation="calendar_features", parameters=dict(decisions=decisions)),
    ]
    first = run_pipeline(
        steps, tmp_path / "run", allow_network=True, transports={"ucdp": Transport()}
    )
    assert first["requests"] == 1
    assert (
        first["results"]["conflicts"]
        .tables["features"]
        .iloc[0]["conflicts.global.state_based.active_conflicts_30D"]
        == 1
    )
    assert first["results"]["calendar"].tables["features"].iloc[0]["calendar.month"] == 1
    replay = run_pipeline(steps, tmp_path / "run")
    assert replay["requests"] == 0 and replay["manifest"] == first["manifest"]
    assert all(
        b"pipeline-test-token" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    )
