"""Economic periods, immutable vintages and justified availability."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe.data.contracts import normalize_observations, observations_asof


def observation(**changes):
    row = dict(
        entity_id="us:cpi",
        metric="cpi_yoy",
        observation_date="2025-01-01",
        published_at="2025-02-01T08:30:00-05:00",
        available_at="2025-02-01T13:31:00Z",
        value=3.0,
        unit="percentage_points",
        source_id="bls",
        revision_id="initial",
        schema_version="1",
    )
    return row | changes


def test_late_old_period_revision_preserves_other_periods_and_past_decisions():
    initial = observation()
    newer = observation(
        observation_date="2025-02-01",
        published_at="2025-03-01T13:30:00Z",
        available_at="2025-03-01T13:31:00Z",
        value=4.0,
    )
    correction = observation(
        revision_id="revised",
        published_at="2025-03-10T13:30:00Z",
        available_at="2025-03-10T13:31:00Z",
        value=3.5,
    )
    before = pd.DataFrame([initial, newer])
    full = pd.DataFrame([correction, newer, initial])
    cutoff = "2025-03-01T14:00:00Z"
    assert_frame_equal(observations_asof(before, cutoff), observations_asof(full, cutoff))
    current = observations_asof(full, "2025-03-10T13:31:00Z")
    assert dict(zip(current.observation_date, current.value, strict=True)) == {
        "2025-01-01": 3.5,
        "2025-02-01": 4.0,
    }
    assert len(normalize_observations(full)) == 3


def test_simultaneous_different_periods_sources_and_instruments_all_survive():
    rows = [
        observation(),
        observation(observation_date="2025-02-01"),
        observation(source_id="other"),
        observation(instrument_id="security:1"),
        observation(instrument_id="security:2"),
    ]
    assert len(observations_asof(pd.DataFrame(rows), "2025-02-01T13:31:00Z")) == 5
    assert observations_asof(pd.DataFrame(rows), "2025-02-01T13:30:59Z").empty


def test_economic_dates_preserve_precision_future_guidance_and_parquet(tmp_path):
    rows = [
        observation(observation_date=date(2026, 12, 31)),
        observation(observation_date="2026-12-31T18:30:00+05:00"),
    ]
    out = normalize_observations(pd.DataFrame(rows))
    assert out.observation_date.tolist() == ["2026-12-31", "2026-12-31T13:30:00+00:00"]
    assert str(out.published_at.dtype) == "datetime64[ns, UTC]"
    assert out.published_at.iloc[0] == pd.Timestamp("2025-02-01T13:30:00Z")
    assert len(observations_asof(out, "2025-02-01T13:31:00Z")) == 2
    assert_frame_equal(normalize_observations(out), out)
    path = tmp_path / "observations.parquet"
    out.to_parquet(path)
    assert_frame_equal(normalize_observations(pd.read_parquet(path)), out)


def test_missing_publication_and_explicit_missing_do_not_assert_point_in_time():
    out = normalize_observations(pd.DataFrame([observation(published_at=None, value=None)]))
    assert pd.isna(out.published_at.iloc[0])
    assert out.value.iloc[0] is None
    assert not out.point_in_time.iloc[0]
    assert out.availability_basis.iloc[0] == "unknown"
    assert pd.isna(out.ingested_at.iloc[0])
    nulls = pd.DataFrame([observation(value=np.nan), observation(metric="other", value=pd.NA)])
    assert normalize_observations(nulls).value.tolist() == [None, None]


def test_nested_structures_are_detached_without_mutating_input():
    value = {"range": [1, 2], "currency": "USD", "missing": None, "confirmed": True}
    raw = pd.DataFrame([observation(value=value)])
    out = normalize_observations(raw)
    out.value.iloc[0]["range"].append(3)
    assert raw.value.iloc[0]["range"] == [1, 2]
    assert raw.published_at.iloc[0] == "2025-02-01T08:30:00-05:00"


def test_cyclic_structures_rejected_and_numpy_json_scalars_normalized():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="acyclic"):
        normalize_observations(pd.DataFrame([observation(value=cyclic)]))
    value = {"confirmed": np.bool_(True), "count": np.int64(2), "ratio": np.float64(0.5)}
    out = normalize_observations(pd.DataFrame([observation(value=value)]))
    assert out.value.iloc[0] == {"confirmed": True, "count": 2, "ratio": 0.5}


def test_receipt_and_reconstruction_have_distinct_ingestion_semantics():
    receipt = observation(
        availability_basis="receipt", ingested_at="2025-02-01T13:30:30Z", point_in_time=False
    )
    assert len(normalize_observations(pd.DataFrame([receipt]))) == 1
    historical = observation(
        availability_basis="reconstructed",
        ingested_at="2026-10-11T12:00:00Z",
        point_in_time=True,
    )
    out = normalize_observations(pd.DataFrame([historical]))
    assert out.available_at.iloc[0] < out.ingested_at.iloc[0]
    assert out.point_in_time.iloc[0]
    with pytest.raises(ValueError, match="ingested_at"):
        normalize_observations(pd.DataFrame([receipt | {"ingested_at": None}]))
    with pytest.raises(ValueError, match="precede ingested_at"):
        normalize_observations(pd.DataFrame([receipt | {"ingested_at": "2025-02-02T00:00:00Z"}]))


@pytest.mark.parametrize(
    "change,message",
    [
        ({"available_at": "2025-02-01"}, "timezone-aware"),
        ({"available_at": 1738416660}, "epoch"),
        ({"published_at": "2025-02-01"}, "timezone-aware"),
        ({"available_at": None}, "timezone-aware"),
        ({"ingested_at": "2025-02-01"}, "timezone-aware"),
        ({"available_at": "2025-02-01T13:29:00Z"}, "precede published_at"),
        ({"observation_date": "2025-02-30"}, "valid date"),
        ({"observation_date": pd.Timestamp("2025-01-01")}, "timezone-aware"),
        ({"observation_date": "2025-01-01T01:00:00"}, "timezone-aware"),
        ({"observation_date": None}, "timezone-aware"),
        ({"value": float("inf")}, "finite"),
        ({"value": {"number": float("nan")}}, "finite"),
        ({"value": {1: "bad key"}}, "JSON-compatible"),
        ({"value": {"when": date(2025, 1, 1)}}, "JSON-compatible"),
        ({"value": True}, "finite numbers"),
        ({"value": "3.0"}, "finite numbers"),
        ({"entity_id": " company"}, "whitespace"),
        ({"source_id": None}, "non-empty"),
        ({"revision_id": ""}, "non-empty"),
        ({"unit": ""}, "non-empty"),
        ({"schema_version": 1}, "non-empty"),
        ({"currency": 1}, "non-empty"),
        ({"point_in_time": "yes"}, "booleans"),
        ({"availability_basis": "assumed"}, "availability_basis"),
        ({"availability_basis": "publication", "published_at": None}, "published_at"),
    ],
)
def test_invalid_fields_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        normalize_observations(pd.DataFrame([observation(**change)]))


def test_duplicates_immutable_revision_ids_and_conflicting_simultaneous_vintages():
    row = observation()
    with pytest.raises(ValueError, match="simultaneous"):
        normalize_observations(pd.DataFrame([row, row]))
    with pytest.raises(ValueError, match="simultaneous"):
        normalize_observations(pd.DataFrame([row, row | {"value": 4, "revision_id": "other"}]))
    late = row | {"available_at": "2025-03-01T00:00:00Z"}
    with pytest.raises(ValueError, match="immutable"):
        normalize_observations(pd.DataFrame([row, late]))
    with pytest.raises(ValueError, match="unit cannot change"):
        normalize_observations(
            pd.DataFrame([row, late | {"revision_id": "new", "unit": "fraction"}])
        )


def test_empty_schema_and_invalid_decision_are_explicit():
    empty = pd.DataFrame(columns=observation())
    out = normalize_observations(empty)
    assert out.empty
    assert observations_asof(out, "2025-01-01T00:00:00Z").empty
    with pytest.raises(ValueError, match="timezone-aware"):
        observations_asof(empty, "2025-01-01")
    with pytest.raises(ValueError, match="requires columns"):
        normalize_observations(pd.DataFrame([observation()]).drop(columns="published_at"))
