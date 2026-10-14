"""Historical knowledge, symbol changes, delistings and identity conflicts."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe.data.security_master import normalize_security_master, security_master_asof


def mapping(**changes):
    row = dict(
        mapping_id="listing:old",
        entity_id="company:1",
        instrument_id="security:1",
        ticker="OLD",
        venue="XNAS",
        currency="USD",
        status="active",
        valid_from="2025-01-01T14:30:00Z",
        valid_to=None,
        available_at="2025-01-01T14:30:00Z",
        source_id="exchange",
        revision_id="initial",
        schema_version="1",
        cik="1234",
        share_class="common",
    )
    return row | changes


def history():
    switch = "2025-02-01T14:30:00Z"
    notice = "2025-01-20T10:00:00Z"
    corrected = "2025-01-28T14:30:00Z"
    return pd.DataFrame(
        [
            mapping(),
            mapping(valid_to=switch, available_at=notice, revision_id="end_announced"),
            mapping(mapping_id="listing:new", ticker="NEW", valid_from=switch, available_at=notice),
            mapping(
                valid_to=corrected, available_at="2025-02-10T10:00:00Z", revision_id="end_corrected"
            ),
            mapping(
                mapping_id="listing:new",
                ticker="NEW",
                valid_from=corrected,
                available_at="2025-02-10T10:00:00Z",
                revision_id="start_corrected",
            ),
        ]
    )


def test_symbol_revisions_are_bitemporal_and_simultaneous_changes_atomic():
    full = history()
    initial = full.iloc[:1]
    past = "2025-01-15T15:00:00Z"
    assert_frame_equal(security_master_asof(initial, past), security_master_asof(full, past))
    assert security_master_asof(full, "2025-02-02T15:00:00Z").ticker.tolist() == ["NEW"]
    # Same effective date, different knowledge date: the correction must not leak.
    effective = "2025-01-30T15:00:00Z"
    assert security_master_asof(full, "2025-02-05T15:00:00Z", effective).ticker.tolist() == ["OLD"]
    assert security_master_asof(full, "2025-02-10T10:00:00Z", effective).ticker.tolist() == ["NEW"]
    assert len(normalize_security_master(full)) == 5


def test_latest_revision_resolved_before_economic_interval_filter():
    original = mapping()
    closed = mapping(
        valid_to="2025-01-10T00:00:00Z",
        available_at="2025-01-15T00:00:00Z",
        revision_id="closed",
    )
    full = pd.DataFrame([original, closed])
    assert security_master_asof(full, "2025-01-16T00:00:00Z", "2025-01-12T00:00:00Z").empty
    assert len(security_master_asof(full, "2025-01-14T00:00:00Z")) == 1


def test_delisting_preserves_old_universe_and_keeps_status_explicit():
    original = mapping()
    announcement = "2025-02-15T10:00:00Z"
    end = "2025-02-20T21:00:00Z"
    rows = [
        original,
        mapping(valid_to=end, available_at=announcement, revision_id="closed"),
        mapping(
            mapping_id="delisting",
            valid_from=end,
            available_at=announcement,
            status="delisted",
        ),
    ]
    full = pd.DataFrame(rows)
    assert security_master_asof(full, "2025-02-14T10:00:00Z").status.tolist() == ["active"]
    assert security_master_asof(full, end).status.tolist() == ["delisted"]
    assert security_master_asof(full, "2025-02-16T10:00:00Z", end).status.tolist() == ["delisted"]
    assert len(normalize_security_master(full)) == 3


def test_current_snapshot_cannot_create_a_universe_known_in_the_past():
    current = pd.DataFrame([mapping(available_at="2026-10-11T12:00:00Z")])
    assert security_master_asof(current, "2025-01-10T12:00:00Z").empty
    assert len(security_master_asof(current, "2026-10-11T12:00:00Z", "2025-01-10T12:00:00Z")) == 1
    assert not normalize_security_master(current).point_in_time.any()


def test_entity_and_instrument_correction_preserves_prior_snapshot():
    correction = mapping(
        entity_id="company:2",
        instrument_id="security:2",
        available_at="2025-01-10T12:00:00Z",
        revision_id="corrected",
    )
    full = pd.DataFrame([mapping(), correction])
    assert security_master_asof(full, "2025-01-05T00:00:00Z").entity_id.tolist() == ["company:1"]
    assert security_master_asof(full, "2025-01-10T12:00:00Z").instrument_id.tolist() == [
        "security:2"
    ]


@pytest.mark.parametrize(
    "other",
    [
        mapping(mapping_id="other", ticker="OTHER"),
        mapping(mapping_id="other", instrument_id="security:2"),
        mapping(mapping_id="other", source_id="other", instrument_id="security:2"),
        mapping(mapping_id="other", source_id="other", entity_id="company:2", ticker="OTHER"),
    ],
)
def test_overlapping_instrument_ticker_and_entity_mappings_rejected(other):
    with pytest.raises(ValueError, match="Overlapping or ambiguous"):
        normalize_security_master(pd.DataFrame([mapping(), other]))


def test_ticker_reuse_after_end_and_independent_agreeing_sources():
    end = "2025-02-01T00:00:00Z"
    rows = [
        mapping(valid_to=end),
        mapping(source_id="other", valid_to=end),
        mapping(mapping_id="reused", instrument_id="security:2", valid_from=end),
    ]
    out = normalize_security_master(pd.DataFrame(rows))
    assert len(security_master_asof(out, "2025-01-15T00:00:00Z")) == 2
    assert security_master_asof(out, end).instrument_id.tolist() == ["security:2"]


def test_overlap_existing_in_old_state_cannot_be_hidden_by_future_correction():
    invalid = mapping(mapping_id="other", ticker="OTHER")
    later = invalid | {
        "valid_from": "2025-02-01T00:00:00Z",
        "available_at": "2025-01-20T00:00:00Z",
        "revision_id": "fixed",
    }
    with pytest.raises(ValueError, match="Overlapping"):
        normalize_security_master(pd.DataFrame([mapping(), invalid, later]))


@pytest.mark.parametrize(
    "change,message",
    [
        ({"valid_from": "2025-01-01"}, "timezone-aware"),
        ({"valid_to": "2025-02-01"}, "timezone-aware"),
        ({"available_at": "2025-01-01"}, "timezone-aware"),
        ({"valid_from": 1735741800}, "epoch"),
        ({"valid_to": "2025-01-01T14:30:00Z"}, "later than"),
        ({"valid_to": "2024-12-31T00:00:00Z"}, "later than"),
        ({"published_at": "2025-01-02T00:00:00Z"}, "precede published_at"),
        ({"ingested_at": "2025-01-01"}, "timezone-aware"),
        ({"mapping_id": None}, "non-empty"),
        ({"ticker": " OLD"}, "whitespace"),
        ({"venue": ""}, "non-empty"),
        ({"currency": ""}, "non-empty"),
        ({"cik": 1234}, "non-empty"),
        ({"cik": "abc"}, "1 to 10 digits"),
        ({"status": "unknown"}, "status"),
        ({"point_in_time": 1}, "booleans"),
    ],
)
def test_invalid_mapping_fields_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        normalize_security_master(pd.DataFrame([mapping(**change)]))


def test_vintage_conflicts_missing_optional_fields_and_idempotence(tmp_path):
    raw = pd.DataFrame([mapping()])
    expected = normalize_security_master(raw)
    assert expected.cik.iloc[0] == "0000001234"
    assert raw.cik.iloc[0] == "1234"
    assert_frame_equal(normalize_security_master(expected), expected)
    path = tmp_path / "security_master.parquet"
    expected.to_parquet(path)
    assert_frame_equal(normalize_security_master(pd.read_parquet(path)), expected)
    optional = normalize_security_master(raw.drop(columns=["cik", "share_class"]))
    assert optional.cik.iloc[0] is None
    assert optional.share_class.iloc[0] is None
    with pytest.raises(ValueError, match="simultaneous"):
        normalize_security_master(pd.concat([raw, raw]))
    with pytest.raises(ValueError, match="immutable"):
        normalize_security_master(
            pd.DataFrame([mapping(), mapping(available_at="2025-01-10T00:00:00Z")])
        )


def test_empty_schema_and_naive_query_times_rejected():
    empty = pd.DataFrame(columns=mapping())
    assert normalize_security_master(empty).empty
    assert security_master_asof(empty, "2025-01-01T00:00:00Z").empty
    with pytest.raises(ValueError, match="timezone-aware"):
        security_master_asof(empty, "2025-01-01")
    with pytest.raises(ValueError, match="timezone-aware"):
        security_master_asof(empty, "2025-01-01T00:00:00Z", "2025-01-01")
