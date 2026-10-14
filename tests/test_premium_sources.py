"""Synthetic COT, licensed exports and prospective chains; no paid/network calls."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfe.data.captures import CaptureLimits, merge_observation_captures
from pfe.data.cot import COT_DATASETS, COTQuery, collect_cot, parse_cot
from pfe.data.estimates import EstimateLicense, LicensedEstimateAdapter
from pfe.data.http import HTTPResponse
from pfe.data.option_archive import archive_option_sample, collect_option_chain, merge_option_chains
from pfe.data.storage import RawStore
from pfe.features.cot import compute_cot_features
from pfe.features.estimates import compute_estimate_features

CONTEXT = dict(
    metric="EPS",
    period_start="2026-01-01",
    period_end="2026-03-31",
    period_kind="quarter",
    currency="USD",
    unit_scale="ones",
    accounting_basis="non_GAAP",
    eps_basis="diluted",
    segment=None,
)


def estimate(value=2.0, snapshot="2026-04-14T12:00:00Z", **changes):
    row = dict(
        CONTEXT,
        entity_id="A",
        statement_type="analyst_estimate",
        statistic="mean",
        analyst_id=None,
        snapshot_at=snapshot,
        published_at=None,
        available_at=snapshot,
        value=value,
        active=True,
        stddev=0.5,
        analyst_count=3,
        announcement_at=None,
    )
    row.update(changes)
    return row


def adapter(pit=True):
    return LicensedEstimateAdapter(
        EstimateLicense(
            "factset", "fixture", "synthetic-license", pit, "fixture-qualification" if pit else None
        )
    )


@pytest.mark.parametrize("report", tuple(COT_DATASETS))
def test_cot_variants_receipt_release_not_tuesday(report, tmp_path):
    query = COTQuery(
        report,
        "001602",
        "wheat",
        "2026-06-01",
        "2026-06-30",
        {"long": "explicit_long", "short": "explicit_short", "open_interest": "open_interest_all"},
        {"2026-06-16": "2026-06-22T15:30:00-04:00"},
    )
    body = json.dumps(
        [
            dict(
                cftc_contract_market_code="001602",
                report_date_as_yyyy_mm_dd="2026-06-16T00:00:00.000",
                explicit_long="30",
                explicit_short="10",
                open_interest_all="100",
            )
        ]
    ).encode()
    result = collect_cot(
        query, RawStore(tmp_path), snapshot_id="cot", offline=(body, "2026-06-22T20:00:00Z")
    )
    assert result.complete
    assert result.observations.available_at.eq(pd.Timestamp("2026-06-22T20:00:00Z")).all()
    assert result.observations.published_at.eq(pd.Timestamp("2026-06-22T19:30:00Z")).all()
    features = compute_cot_features(
        ["2026-06-22T19:59:59Z", "2026-06-22T20:00:00Z"],
        result.observations,
        entity_id="wheat",
        report=report,
        long_metric="long",
        short_metric="short",
    )
    assert np.isnan(features.iloc[0]["cot.net"])
    assert features.iloc[1]["cot.net"] == 20 and features.iloc[1]["cot.net_fraction_oi"] == 0.2


def test_cot_cap_and_online_query_are_bounded(tmp_path):
    query = COTQuery("legacy_futures", "001602", "wheat", "2026-01-01", "2026-01-31")
    body = (
        b'[{"cftc_contract_market_code":"001602",'
        b'"report_date_as_yyyy_mm_dd":"2026-01-06","open_interest_all":"100"}]'
    )

    class Transport:
        def get(self, url, *, params, max_bytes):
            assert params["$limit"] == 1 and "001602" in params["$where"]
            assert "6dca-aqww" in url and max_bytes == 524288
            return HTTPResponse(body, "2026-01-10T00:00:00Z")

    store = RawStore(tmp_path)
    result = collect_cot(
        query,
        store,
        snapshot_id="selected",
        transport=Transport(),
        limits=CaptureLimits(max_rows=1),
    )
    assert not result.complete
    replay = collect_cot(query, store, snapshot_id="selected", limits=CaptureLimits(max_rows=1))
    assert result.collection_id == replay.collection_id


def test_cot_weekly_change_rejects_incompatible_prior_revisions():
    query = COTQuery(
        "legacy_futures",
        "001602",
        "wheat",
        "2026-06-01",
        "2026-06-30",
        {"long": "long", "short": "short", "open_interest": "open_interest_all"},
    )
    body = json.dumps(
        [
            dict(
                cftc_contract_market_code="001602",
                report_date_as_yyyy_mm_dd=day,
                long=value,
                short=10,
                open_interest_all=100,
            )
            for day, value in (("2026-06-09", 30), ("2026-06-16", 40))
        ]
    ).encode()
    frame = parse_cot(body, query=query, received_at="2026-06-20T00:00:00Z")
    kwargs = dict(
        entity_id="wheat", report="legacy_futures", long_metric="long", short_metric="short"
    )
    valid = compute_cot_features(["2026-06-20T00:00:00Z"], frame, **kwargs)
    assert valid.iloc[0]["cot.net_change_1w"] == 10
    previous_short = frame.observation_date.eq("2026-06-09") & frame.metric.str.endswith(".short")
    frame.loc[previous_short, "revision_id"] = "independent-short-revision"
    out = compute_cot_features(["2026-06-20T00:00:00Z"], frame, **kwargs)
    assert out.iloc[0]["cot.net"] == 30
    assert np.isnan(out.iloc[0]["cot.net_change_1w"])


def test_surprise_uses_strict_preannouncement_consensus_and_matching_basis():
    rows = [
        estimate(),
        estimate(99.0, "2026-04-15T15:01:00Z"),
        estimate(
            2.5,
            "2026-04-15T15:00:00Z",
            statement_type="reported_actual",
            statistic="actual",
            stddev=None,
            analyst_count=None,
            published_at="2026-04-15T15:00:00Z",
            announcement_at="2026-04-15T15:00:00Z",
            available_at="2026-04-15T15:02:00Z",
        ),
    ]
    frame = adapter().parse(json.dumps(rows).encode(), received_at="2026-04-17T00:00:00Z")
    out = compute_estimate_features(
        ["2026-04-14T12:00:00Z", "2026-04-15T15:02:00Z"],
        frame,
        entity_id="A",
        source_id="factset",
        context=CONTEXT,
    )
    assert out.iloc[0]["estimates.dispersion"] == 0.25
    assert out.iloc[1]["estimates.mean"] == 99
    assert out.iloc[1]["estimates.surprise"] == 0.5
    assert out.iloc[1]["estimates.surprise_fraction"] == 0.25
    wrong = dict(CONTEXT, eps_basis="basic")
    assert (
        compute_estimate_features(
            ["2026-04-17T00:00:00Z"], frame, entity_id="A", source_id="factset", context=wrong
        ).iloc[0]["estimates.mask"]
        == 0
    )


def test_unqualified_import_cannot_backdate_or_create_earlier_surprise():
    rows = [estimate(), estimate(3.0, "2026-04-15T12:00:00Z")]
    frame = adapter(False).parse(json.dumps(rows).encode(), received_at="2026-04-17T00:00:00Z")
    assert not frame.point_in_time.any()
    out = compute_estimate_features(
        ["2026-04-16T00:00:00Z", "2026-04-17T00:00:00Z"],
        frame,
        entity_id="A",
        source_id="factset",
        context=CONTEXT,
    )
    assert out.iloc[0]["estimates.mask"] == 0 and out.iloc[1]["estimates.mean"] == 3


def test_detail_analyst_updates_and_withdrawals_no_double_count():
    rows = [
        estimate(1.0, statistic="detail", analyst_id="a", stddev=None, analyst_count=None),
        estimate(3.0, statistic="detail", analyst_id="b", stddev=None, analyst_count=None),
        estimate(None, "2026-04-15T12:00:00Z", statistic="detail", analyst_id="a", active=False),
    ]
    frame = adapter().parse(json.dumps(rows).encode(), received_at="2026-04-17T00:00:00Z")
    out = compute_estimate_features(
        ["2026-04-14T12:00:00Z", "2026-04-15T12:00:00Z"],
        frame,
        entity_id="A",
        source_id="factset",
        context=CONTEXT,
    )
    assert out.iloc[0]["estimates.mean"] == 2 and out.iloc[0]["estimates.analyst_count"] == 2
    assert out.iloc[1]["estimates.mean"] == 3 and out.iloc[1]["estimates.analyst_count"] == 1


@pytest.mark.parametrize(
    "change",
    [
        dict(statement_type="management_guidance"),
        dict(statistic="detail"),
        dict(currency=None),
        dict(active="true"),
        dict(analyst_count=2.5),
        dict(available_at="2026-04-13T00:00:00Z"),
    ],
)
def test_invalid_estimate_class_basis_time_and_count_rejected(change):
    with pytest.raises(ValueError):
        adapter().parse(
            json.dumps([estimate(**change)]).encode(), received_at="2026-04-17T00:00:00Z"
        )


def test_licensed_capture_replay_without_client_and_coverage(tmp_path):
    body = json.dumps([estimate()]).encode()
    calls = []

    def fetch(*, max_bytes):
        calls.append(max_bytes)
        return HTTPResponse(body, "2026-04-17T00:00:00Z")

    client, store = adapter(), RawStore(tmp_path)
    first = client.capture(store, snapshot_id="export", query={"entity_ids": ["A"]}, fetch=fetch)
    second = client.capture(store, snapshot_id="export", query={"entity_ids": ["A"]})
    assert first.collection_id == second.collection_id and len(calls) == 1


def option_payload():
    return dict(
        provider="LSE",
        snapshot_at="2026-01-02T15:00:00Z",
        limit=1,
        chain=[
            dict(
                underlying="A",
                ticker="A261218C00100000",
                contract_type="call",
                strike=100.0,
                expiry="2026-12-18",
                last_trade_at="2026-01-02T14:59:00Z",
                last_price=2.0,
                volume_today=5.0,
                premium_today=1000.0,
            )
        ],
        flow=[],
        candles=[],
        limit_reached={"chain": True, "flow": False, "candles": False},
    )


def test_option_archive_atomic_receipt_missing_oi_and_replay(tmp_path):
    store = RawStore(tmp_path)
    capture = archive_option_sample(
        option_payload(), store, snapshot_id="snapshot", received_at="2026-01-02T15:01:00Z"
    )
    replay = archive_option_sample(
        option_payload(), store, snapshot_id="snapshot", received_at="2026-01-02T15:01:00Z"
    )
    assert capture.collection_id == replay.collection_id
    chain = capture.data.chain
    assert chain.available_at.eq(pd.Timestamp("2026-01-02T15:01:00Z")).all()
    assert chain.open_interest.isna().all() and chain.bid.isna().all() and chain.ask.isna().all()
    assert capture.metadata["oi_coverage"] == "unqualified"
    merged = merge_option_chains(capture, replay)
    assert len(merged) == 1 and merged.attrs["limit_reached"]


def test_option_chain_native_bounded_capture_replay_without_sdk(tmp_path):
    calls = []

    class Transport:
        def get(self, endpoint, params, *, max_bytes):
            calls.append(params)
            assert endpoint == "/options/chain" and params["limit"] == 1 and max_bytes == 524288
            return SimpleNamespace(payload=option_payload()["chain"])

    store = RawStore(tmp_path)
    one = collect_option_chain(
        Transport(), store, underlying="A", snapshot_id="prospective", limit=1
    )
    two = collect_option_chain(None, store, underlying="A", snapshot_id="prospective", limit=1)
    assert one.collection_id == two.collection_id and len(calls) == 1


def test_no_license_no_historical_pit_claim():
    with pytest.raises(ValueError, match="qualification"):
        EstimateLicense("ibes", "PIT", "license", True)


def test_imported_options_keep_source_snapshot_age_and_vault_utc_clock(tmp_path):
    sample = option_payload()
    sample["chain"][0]["last_trade_at"] = "2026-01-02 14:59:00"
    capture = archive_option_sample(
        sample, RawStore(tmp_path), snapshot_id="old", received_at="2026-02-01T00:00:00Z"
    )
    assert capture.data.chain.snapshot_at.iloc[0] == pd.Timestamp("2026-01-02T15:00:00Z")
    assert capture.data.chain.available_at.iloc[0] == pd.Timestamp("2026-02-01T00:00:00Z")
    assert capture.data.chain.quote_at.iloc[0] == pd.Timestamp("2026-01-02T14:59:00Z")


def test_late_old_consensus_snapshot_does_not_replace_newer_snapshot():
    rows = [
        estimate(2.0, "2026-04-13T12:00:00Z"),
        estimate(3.0, "2026-04-14T12:00:00Z"),
        estimate(9.0, "2026-04-13T12:00:00Z", available_at="2026-04-15T12:00:00Z"),
    ]
    frame = adapter().parse(json.dumps(rows).encode(), received_at="2026-04-17T00:00:00Z")
    out = compute_estimate_features(
        ["2026-04-14T12:00:00Z", "2026-04-15T12:00:00Z"],
        frame,
        entity_id="A",
        source_id="factset",
        context=CONTEXT,
    )
    assert out["estimates.mean"].tolist() == [3.0, 3.0]


def test_actual_withdrawal_and_missing_announcement_do_not_reuse_old_surprise():
    actual = estimate(
        2.5,
        "2026-04-15T15:00:00Z",
        statement_type="reported_actual",
        statistic="actual",
        announcement_at="2026-04-15T15:00:00Z",
        stddev=None,
        analyst_count=None,
    )
    withdrawal = dict(actual, active=False, value=None, available_at="2026-04-16T00:00:00Z")
    frame = adapter().parse(
        json.dumps([estimate(), actual, withdrawal]).encode(), received_at="2026-04-17T00:00:00Z"
    )
    out = compute_estimate_features(
        ["2026-04-15T15:00:00Z", "2026-04-16T00:00:00Z"],
        frame,
        entity_id="A",
        source_id="factset",
        context=CONTEXT,
    )
    assert out["estimates.surprise_mask"].tolist() == [1.0, 0.0]
    no_time = dict(actual, announcement_at=None)
    frame = adapter().parse(
        json.dumps([estimate(), no_time]).encode(), received_at="2026-04-17T00:00:00Z"
    )
    assert (
        compute_estimate_features(
            ["2026-04-17T00:00:00Z"], frame, entity_id="A", source_id="factset", context=CONTEXT
        ).iloc[0]["estimates.surprise_mask"]
        == 0
    )


def test_export_mapping_and_csv_boolean_values_are_explicit(tmp_path):
    import csv
    import io

    row = estimate()
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    values = {
        "active": {"True": True},
        "analyst_id": {"": None},
        "published_at": {"": None},
        "segment": {"": None},
        "announcement_at": {"": None},
    }
    client = LicensedEstimateAdapter(adapter().license, value_maps=values)
    frame = client.parse(
        stream.getvalue().encode(), received_at="2026-04-17T00:00:00Z", format="csv"
    )
    assert frame.value.iloc[0]["value"] == 2
    capture = client.capture(
        RawStore(tmp_path),
        snapshot_id="authorized-csv",
        query={"entity_ids": ["A"]},
        offline=(stream.getvalue().encode(), "2026-04-17T00:00:00Z"),
        format="csv",
    )
    assert capture.complete and capture.observations.value.iloc[0]["value"] == 2


@pytest.mark.parametrize("online", [False, True])
def test_csv_credentials_rejected_before_raw_persistence(tmp_path, online):
    import csv
    import io

    row = dict(estimate(), api_key="AUDIT_SECRET_SENTINEL")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    body = stream.getvalue().encode()
    options = (
        {"fetch": lambda **kwargs: HTTPResponse(body, "2026-04-17T00:00:00Z")}
        if online
        else {"offline": (body, "2026-04-17T00:00:00Z")}
    )
    with pytest.raises(ValueError, match="Credentials"):
        adapter().capture(
            RawStore(tmp_path),
            snapshot_id="csv-credentials",
            query={"entity_ids": ["A"]},
            format="csv",
            **options,
        )
    assert not any(
        b"AUDIT_SECRET_SENTINEL" in path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )


def test_actual_cannot_be_available_before_announcement_but_receipt_can_follow_it():
    actual = estimate(
        2.5,
        "2026-04-14T12:01:00Z",
        statistic="actual",
        statement_type="reported_actual",
        announcement_at="2026-04-15T15:00:00Z",
        stddev=None,
        analyst_count=None,
    )
    body = json.dumps([actual]).encode()
    with pytest.raises(ValueError, match="before its announcement"):
        adapter().parse(body, received_at="2026-04-17T00:00:00Z")
    imported = adapter(False).parse(body, received_at="2026-04-17T00:00:00Z")
    assert imported.available_at.iloc[0] == pd.Timestamp("2026-04-17T00:00:00Z")
    # An analyst estimate may carry a future scheduled announcement as metadata.
    analyst = estimate(announcement_at="2026-04-18T15:00:00Z")
    assert (
        len(adapter().parse(json.dumps([analyst]).encode(), received_at="2026-04-17T00:00:00Z"))
        == 1
    )


def test_future_announcement_never_uses_estimates_after_decision():
    actual = estimate(
        2.5,
        "2026-04-14T12:01:00Z",
        statistic="actual",
        statement_type="reported_actual",
        announcement_at="2026-04-15T15:00:00Z",
        available_at="2026-04-15T15:00:00Z",
        stddev=None,
        analyst_count=None,
    )
    frame = adapter().parse(
        json.dumps([estimate(), actual, estimate(99, "2026-04-15T14:59:00Z")]).encode(),
        received_at="2026-04-17T00:00:00Z",
    )
    # Direct canonical inputs also need a defensive decision-time check, even
    # when upstream metadata incorrectly makes the actual available early.
    actual_rows = frame.value.map(lambda value: value["statement_type"] == "reported_actual")
    frame.loc[actual_rows, "available_at"] = pd.Timestamp("2026-04-14T12:01:00Z")
    decision = pd.Timestamp("2026-04-14T12:02:00Z")
    base = frame.loc[frame.available_at <= decision]
    kwargs = dict(entity_id="A", source_id="factset", context=CONTEXT)
    original = compute_estimate_features([decision], base, **kwargs)
    appended = compute_estimate_features([decision], frame, **kwargs)
    pd.testing.assert_frame_equal(original, appended)
    assert appended.iloc[0]["estimates.surprise_mask"] == 0
    assert np.isnan(appended.iloc[0]["estimates.surprise"])


def test_capture_failure_resumes_without_client(tmp_path, monkeypatch):
    client, store = adapter(), RawStore(tmp_path)
    body = json.dumps([estimate()]).encode()
    original = store.write_page

    def fail(collection, page, payload, **kwargs):
        if page == "normalized":
            raise OSError("simulated interruption")
        return original(collection, page, payload, **kwargs)

    monkeypatch.setattr(store, "write_page", fail)
    with pytest.raises(OSError):
        client.capture(
            store,
            snapshot_id="resume",
            query={"entity_ids": ["A"]},
            fetch=lambda **kw: HTTPResponse(body, "2026-04-17T00:00:00Z"),
        )
    monkeypatch.setattr(store, "write_page", original)
    assert client.capture(store, snapshot_id="resume", query={"entity_ids": ["A"]}).complete


@pytest.mark.parametrize(
    "limits",
    [
        CaptureLimits(max_response_bytes=10),
        CaptureLimits(max_response_bytes=1000, max_archive_bytes=1000),
    ],
)
def test_source_and_archive_caps(limits, tmp_path):
    with pytest.raises(ValueError, match="budget"):
        adapter().capture(
            RawStore(tmp_path),
            snapshot_id="cap",
            query={"entity_ids": ["A"]},
            offline=(json.dumps([estimate()]).encode(), "2026-04-17T00:00:00Z"),
            limits=limits,
        )


def test_overlapping_capture_history_keeps_first_availability(tmp_path):
    client, store = adapter(False), RawStore(tmp_path)
    body = json.dumps([estimate()]).encode()
    first = client.capture(
        store,
        snapshot_id="first",
        query={"entity_ids": ["A"]},
        offline=(body, "2026-04-17T00:00:00Z"),
    )
    later = client.capture(
        store,
        snapshot_id="later",
        query={"entity_ids": ["A"]},
        offline=(body, "2026-04-18T00:00:00Z"),
    )
    merged = merge_observation_captures(first, later)
    assert len(merged) == 1 and merged.available_at.iloc[0] == pd.Timestamp("2026-04-17T00:00:00Z")
    conflicting = first.observations.copy()
    conflicting.at[0, "value"] = dict(conflicting.value.iloc[0], value=99.0)
    with pytest.raises(ValueError, match="Conflicting"):
        merge_observation_captures(first, conflicting)
