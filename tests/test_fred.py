"""Synthetic native FRED pages, date-level vintage policies and sparse macro bridges."""

import json

import numpy as np
import pandas as pd
import pytest

from pfe.data import storage
from pfe.data.alfred import normalize_alfred_observations
from pfe.data.fred import (
    FREDError,
    collect_fred_series,
    compute_fred_features,
    compute_fred_macro_features,
    fred_to_macro_series,
    normalize_fred_observations,
)
from pfe.data.http import HTTPResponse
from pfe.data.storage import RawStore


def _metadata(series="SYN", **changes):
    return (
        dict(
            id=series,
            title="Synthetic rate",
            units="Percent",
            units_short="%",
            frequency="Daily",
            seasonal_adjustment="Not Seasonally Adjusted",
            last_updated="2026-01-05 09:26:00-05",
            notes="Synthetic fixtures only",
        )
        | changes
    )


def _row(period="2026-01-01", vintage="2026-01-04", value="1.5", end="9999-12-31"):
    return dict(date=period, realtime_start=vintage, realtime_end=end, value=value)


def _adapt(rows, **options):
    return normalize_fred_observations(
        rows, _metadata(), received_at="2026-04-01T12:00:00Z", **options
    )


def test_latest_receipt_does_not_backdate_or_invent_publication():
    out = _adapt([_row(), _row(period="2026-01-02", value=".")])
    assert out.available_at.eq(pd.Timestamp("2026-04-01T12:00:00Z")).all()
    assert out.published_at.isna().all()
    assert out.availability_basis.eq("receipt").all()
    assert out.point_in_time.eq(False).all()
    assert out.metadata_point_in_time.eq(False).all()
    assert out.unit.eq("Percent").all()
    assert out.loc[0, "realtime_end"] == "9999-12-31"
    assert out.loc[1, "value"] is None
    macro = fred_to_macro_series(out)
    assert len(macro) == 1 and macro.value.iloc[0] == 1.5
    assert macro.attrs["units"] == {"SYN": "Percent"}


def test_next_day_convention_not_vintage_midnight_and_respects_dst():
    out = _adapt(
        [_row(vintage="2026-03-07"), _row(period="2026-01-02", vintage="2026-03-08")],
        availability="vintage_next_day",
    )
    assert out.available_at.tolist() == [
        pd.Timestamp("2026-03-08T05:00:00Z"),
        pd.Timestamp("2026-03-09T04:00:00Z"),
    ]
    assert out.availability_basis.eq("reconstructed").all()
    assert out.point_in_time.eq(False).all()
    utc = normalize_alfred_observations(
        [_row()], _metadata(), received_at="2026-04-01T12:00:00Z", vintage_timezone="UTC"
    )
    assert utc.available_at.iloc[0] == pd.Timestamp("2026-01-05T00:00:00Z")


@pytest.mark.parametrize(
    "row",
    [
        _row(value="NaN"),
        _row(value="inf"),
        _row(value=""),
        _row(value=1.5),
        _row(value=True),
        _row(vintage="2026-02-30"),
        _row(end="2020-01-01"),
    ],
)
def test_invalid_native_values_and_dates_rejected(row):
    with pytest.raises(ValueError):
        _adapt([row])


def test_ambiguous_historical_vintages_at_single_receipt_rejected():
    with pytest.raises(ValueError, match="simultaneous"):
        _adapt([_row(), _row(vintage="2026-02-04", value="1.6")])


def test_older_period_revision_preserves_latest_period_and_sparse_changes():
    rows = [
        _row(value="100"),
        _row(period="2026-02-01", vintage="2026-03-04", value="110"),
        _row(vintage="2026-03-06", value="102"),
    ]
    out = _adapt(rows, availability="vintage_next_day")
    macro = fred_to_macro_series(out)
    assert len(macro) == 3
    bars = pd.DatetimeIndex(
        ["2026-01-05T04:59:00Z", "2026-03-05T04:59:00Z", "2026-03-07T04:59:00Z"]
    )
    result = compute_fred_macro_features(
        bars,
        observations=out,
        names=[
            "macro.SYN.level",
            "macro.SYN.change",
            "macro.SYN.is_revision",
        ],
    )
    assert result["macro.SYN.level"].tolist() == [100, 110, 110]
    assert np.isnan(result["macro.SYN.change"].iloc[0])
    assert result["macro.SYN.change"].iloc[-1] == 8
    assert result["macro.SYN.is_revision"].iloc[-1] == 1


def test_tombstone_does_not_silently_resurrect_an_old_numeric_value():
    out = _adapt([_row(), _row(vintage="2026-02-04", value=".")], availability="vintage_next_day")
    macro = fred_to_macro_series(out)
    assert len(macro) == 2 and pd.isna(macro.value.iloc[-1])
    before, after = macro.available_at
    features = compute_fred_features(
        pd.DatetimeIndex([before, after]), observations=out, names=["macro.SYN.level"]
    )
    assert features.iloc[0, 0] == 1.5 and pd.isna(features.iloc[1, 0])


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("frequency", "Monthly"),
        ("seasonal_adjustment", "Seasonally Adjusted"),
    ],
)
def test_single_series_rolling_rejects_changed_metadata_across_periods_and_revisions(
    field, new_value
):
    frame = _adapt(
        [
            _row(value="100"),
            _row(period="2026-02-01", vintage="2026-03-04", value="110"),
            _row(vintage="2026-03-06", value="102"),
        ],
        availability="vintage_next_day",
    )
    # Raw observations legitimately retain each source declaration. The numeric
    # bridge must not mix monthly/daily or SA/NSA values in one rolling window.
    frame.loc[frame.observation_date.eq("2026-02-01"), field] = new_value
    with pytest.raises(FREDError, match=field):
        fred_to_macro_series(frame)
    with pytest.raises(FREDError, match=field):
        compute_fred_features(
            pd.DatetimeIndex(["2026-04-01T12:00:00Z"]),
            observations=frame,
            names=["macro.SYN.mean_w3"],
        )


def test_bridge_retains_consistent_frequency_and_seasonal_metadata():
    frame = fred_to_macro_series(_adapt([_row()]))
    assert frame.attrs["frequency"] == {"SYN": "Daily"}
    assert frame.attrs["seasonal_adjustment"] == {"SYN": "Not Seasonally Adjusted"}


def test_common_input_cannot_merge_two_entities_under_one_macro_metric():
    frame = _adapt([_row(), _row(period="2026-01-02")])
    frame.loc[1, "entity_id"] = "OTHER"
    with pytest.raises(FREDError, match="Multiple entities"):
        fred_to_macro_series(frame)


def test_rate_spread_preserves_percentage_points_and_rejects_incompatible_units():
    first = normalize_fred_observations(
        [_row(value="4")], _metadata("LONG"), received_at="2026-04-01T12:00:00Z"
    )
    second = normalize_fred_observations(
        [_row(value="2")], _metadata("SHORT"), received_at="2026-04-01T12:00:00Z"
    )
    frame = pd.concat([first, second], ignore_index=True)
    bars = pd.DatetimeIndex(["2026-04-01T11:59:00Z", "2026-04-01T12:00:00Z"])
    result = compute_fred_macro_features(
        bars, observations=frame, pairs=[("LONG", "SHORT")], groups=["spreads"]
    )
    assert result["macro.pair[LONG,SHORT].spread"].iloc[-1] == 2
    assert result.attrs["spread_units"]["macro.pair[LONG,SHORT].spread"] == "percentage_points"
    second["unit"] = "Index"
    with pytest.raises(FREDError, match="compatible"):
        compute_fred_macro_features(
            bars,
            observations=pd.concat([first, second]),
            pairs=[("LONG", "SHORT")],
            groups=["spreads"],
        )


def test_explicit_decisions_have_no_bar_offset_and_preserve_entity_index_order():
    observations = _adapt([_row()])
    decisions = pd.DataFrame(
        {
            "entity_id": ["second", "first", "third"],
            "decision_at": ["2026-04-01T12:00:00Z", "2026-04-01T11:59:59Z", "2026-04-01T12:00:00Z"],
        },
        index=[9, 4, 9],
    )
    result = compute_fred_features(decisions, observations=observations, names=["macro.SYN.level"])
    assert result.index.equals(decisions.index)
    assert result.entity_id.tolist() == decisions.entity_id.tolist()
    assert result["macro.SYN.level"].iloc[0] == 1.5
    assert np.isnan(result["macro.SYN.level"].iloc[1])
    assert result["macro.SYN.level"].iloc[2] == 1.5
    assert result.attrs["timestamp_semantics"] == "explicit_decision"
    index = pd.DatetimeIndex(["2026-04-01T11:59:59Z", "2026-04-01T12:00:00Z"])
    direct = compute_fred_features(index, observations=observations, names=["macro.SYN.level"])
    assert direct.index.equals(index)
    assert np.isnan(direct.iloc[0, 0]) and direct.iloc[1, 0] == 1.5


@pytest.mark.parametrize(
    "decisions",
    [
        pd.DatetimeIndex(["2026-04-01"]),
        pd.DataFrame({"entity_id": [7], "decision_at": ["2026-04-01T12:00:00Z"]}),
        pd.DataFrame({"entity_id": ["company"], "decision_at": ["2026-04-01"]}),
    ],
)
def test_explicit_decisions_require_typed_ids_and_aware_instants(decisions):
    with pytest.raises(ValueError):
        compute_fred_features(decisions, observations=_adapt([_row()]))


def _page(key, rows, *, offset=0, limit=2, count=None):
    return {
        key: rows,
        "count": len(rows) if count is None else count,
        "offset": offset,
        "limit": limit,
    }


class FakeFRED:
    def __init__(self, rows=None, *, count_delta=False, empty_second=False):
        self.rows = (
            [
                _row(period=f"2026-01-{i:02}", vintage="2026-04-01", value=str(i))
                for i in range(1, 4)
            ]
            if rows is None
            else rows
        )
        self.calls = []
        self.count_delta, self.empty_second = count_delta, empty_second

    def get(self, url, *, params, max_bytes):
        endpoint = url.removeprefix("https://api.stlouisfed.org/fred")
        self.calls.append((endpoint, params.copy()))
        if endpoint == "/series":
            payload = {"seriess": [_metadata()], "api_key": "synthetic-secret"}
        elif endpoint == "/series/release":
            payload = {
                "releases": [
                    {"id": 82, "name": "Synthetic release", "link": "https://example.test/release"}
                ]
            }
        else:
            key, rows = {
                "/series/observations": ("observations", self.rows),
                "/series/vintagedates": ("vintage_dates", ["2026-04-01"]),
                "/release/dates": ("release_dates", [{"release_id": 82, "date": "2026-04-01"}]),
            }[endpoint]
            offset, limit = params["offset"], params["limit"]
            batch = rows[offset : offset + limit]
            if endpoint == "/series/observations":
                batch = [row | {"realtime_end": params["realtime_end"]} for row in batch]
            if self.empty_second and endpoint == "/series/observations" and offset:
                batch = []
            count = len(rows) + (1 if self.count_delta and offset else 0)
            payload = _page(key, batch, offset=offset, limit=limit, count=count)
        body = json.dumps(payload).encode()
        if len(body) > max_bytes:
            raise FREDError("Synthetic byte cap")
        return HTTPResponse(body, "2026-04-01T12:00:00Z", "application/json")


def _collect(transport, store, **options):
    return collect_fred_series(
        transport,
        store,
        series_id="SYN",
        start="2026-01-01",
        end="2026-01-31",
        snapshot_id="one",
        realtime_start="2026-04-01",
        realtime_end="2026-04-01",
        api_key="synthetic-secret" if transport else None,
        page_size=2,
        **options,
    )


def test_paginated_collection_cache_release_dates_and_secret_free_store(tmp_path):
    transport, store = FakeFRED(), RawStore(tmp_path)
    collection_id, frame, summary = _collect(transport, store)
    assert frame.value.tolist() == [1.0, 2.0, 3.0]
    assert summary["pagination_complete"] is True
    assert summary["source_rows"] == 7
    assert summary["release_dates"] == [{"release_id": 82, "date": "2026-04-01"}]
    assert summary["publication_timing"] == "unqualified"
    assert store.manifest(collection_id)["complete"] is True
    original = store.manifest(collection_id)
    replay_id, replay, replay_summary = _collect(None, store)
    assert replay_id == collection_id and replay_summary == summary
    pd.testing.assert_frame_equal(frame, replay)
    assert store.manifest(collection_id) == original
    assert len(transport.calls) == 6
    for path in tmp_path.rglob("*.bin"):
        assert b"synthetic-secret" not in path.read_bytes()
    assert "synthetic-secret" not in json.dumps(original)


def test_row_cap_counts_missing_and_metadata_does_not_infer_absence(tmp_path):
    transport = FakeFRED(rows=[_row(vintage="2026-04-01", value=".")])
    store = RawStore(tmp_path)
    collection_id, frame, summary = _collect(transport, store, max_rows=3)
    assert len(frame) == 1 and frame.value.iloc[0] is None
    assert summary["source_rows"] == 3 and summary["missing_rows"] == 1
    assert summary["pagination_complete"] is False
    assert summary["absence_scope"] == "unqualified"
    assert store.manifest(collection_id)["complete"] is False
    assert store.resume_cursor(collection_id) is not None


def test_request_cap_can_resume_without_refetching_original_pages(tmp_path):
    transport, store = FakeFRED(), RawStore(tmp_path)
    collection_id, first, summary = _collect(transport, store, max_requests=3)
    assert len(first) == 2 and summary["pagination_complete"] is False
    first_page = store.read_page(collection_id, "observations-0")
    resumed_id, resumed, summary = _collect(transport, store, max_requests=3)
    assert resumed_id == collection_id and len(resumed) == 3
    assert summary["pagination_complete"] is True
    assert store.read_page(collection_id, "observations-0") == first_page
    assert len(transport.calls) == 6
    assert store.manifest(collection_id)["complete"] is True


@pytest.mark.parametrize("option", [{"count_delta": True}, {"empty_second": True}])
def test_changed_counts_or_nonadvancing_pages_rejected(tmp_path, option):
    with pytest.raises(FREDError):
        _collect(FakeFRED(**option), RawStore(tmp_path))
    (manifest_path,) = tmp_path.glob("*/manifest.json")
    assert json.loads(manifest_path.read_bytes())["complete"] is False


def test_manifest_crash_recovers_original_http_receipt_and_skips_refetch(tmp_path, monkeypatch):
    publish = storage._atomic_write
    attempts = 0

    def fail_once(path, body, *, replace):
        nonlocal attempts
        if path.name == "manifest.json" and replace:
            attempts += 1
            if attempts == 1:
                raise OSError("interrupt")
        return publish(path, body, replace=replace)

    monkeypatch.setattr(storage, "_atomic_write", fail_once)
    transport, store = FakeFRED(), RawStore(tmp_path)
    with pytest.raises(OSError, match="interrupt"):
        _collect(transport, store)
    collection_id, frame, summary = _collect(transport, store)
    assert len(transport.calls) == 6
    assert frame.available_at.eq(pd.Timestamp("2026-04-01T12:00:00Z")).all()
    assert summary["pagination_complete"] is True
    assert store.manifest(collection_id)["complete"] is True


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("frequency", "Quarterly"),
        ("seasonal_adjustment", "Seasonally Adjusted"),
        ("entity_id", "another-factor"),
    ],
)
def test_bridge_rejects_changed_series_conventions_or_colliding_entities(field, new_value):
    first = _adapt([_row()])
    second = _adapt([_row(period="2026-03-01")])
    second[field] = new_value
    pattern = "Multiple entities|entity_id changed" if field == "entity_id" else f"{field} changed"
    with pytest.raises(FREDError, match=pattern):
        fred_to_macro_series(pd.concat([first, second], ignore_index=True))
