"""Small directory snapshots: stable IDs, conservative changes and replay."""

import json

import pandas as pd
import pytest

from pfe.data.security_sources import (
    collect_security_master,
    parse_security_snapshot,
    security_universe_asof,
    update_security_master,
)
from pfe.data.storage import RawStore

IDENTITY = dict(entity_id="issuer:1", instrument_id="security:1", venue="XNAS", currency="USD")


def sec(ticker):
    return json.dumps({"0": dict(ticker=ticker, cik_str=1, title="Fixture")}).encode()


def test_snapshot_changes_are_receipt_based_and_absence_is_not_delisting(tmp_path):
    store = RawStore(tmp_path)
    first = collect_security_master(
        store,
        source="sec",
        identities={"OLD": IDENTITY},
        snapshot_id="1",
        offline=(sec("OLD"), "2026-01-01T00:00:00Z"),
    )
    second = collect_security_master(
        store,
        source="sec",
        identities={"NEW": IDENTITY},
        snapshot_id="2",
        previous=first.history,
        offline=(sec("NEW"), "2026-02-01T00:00:00Z"),
    )
    assert security_universe_asof(second.history, "2025-12-31T00:00:00Z").empty
    assert security_universe_asof(second.history, "2026-01-02T00:00:00Z").ticker.tolist() == ["OLD"]
    assert security_universe_asof(second.history, "2026-02-02T00:00:00Z").ticker.tolist() == ["NEW"]
    replay = collect_security_master(
        store,
        source="sec",
        identities={"OLD": IDENTITY},
        snapshot_id="1",
        previous=second.history,
        offline=(sec("OLD"), "2026-01-01T00:00:00Z"),
    )
    pd.testing.assert_frame_equal(replay.history, second.history)
    missing = parse_security_snapshot(
        sec("OTHER"), source="sec", identities={"NEW": IDENTITY}, received_at="2026-03-01T00:00:00Z"
    )
    unchanged = update_security_master(second.history, missing)
    assert security_universe_asof(unchanged, "2026-03-02T00:00:00Z").ticker.tolist() == ["NEW"]


def test_explicit_delisting_keeps_old_universe_and_date_precision():
    original = parse_security_snapshot(
        sec("A"), source="sec", identities={"A": IDENTITY}, received_at="2026-01-01T00:00:00Z"
    )
    # Different sources remain independent; use a first AV receipt and its later status.
    csv = b"symbol,status,ipoDate,delistingDate\nA,Active,2000-01-01,null\n"
    one = parse_security_snapshot(
        csv, source="alpha_vantage", identities={"A": IDENTITY}, received_at="2026-01-01T00:00:00Z"
    )
    two = parse_security_snapshot(
        csv.replace(b"Active", b"Delisted").replace(b"null", b"2026-01-02"),
        source="alpha_vantage",
        identities={"A": IDENTITY},
        received_at="2026-01-03T00:00:00Z",
    )
    history = update_security_master(one, two)
    assert len(original) == 1 and len(security_universe_asof(history, "2026-01-02T00:00:00Z")) == 1
    assert security_universe_asof(history, "2026-01-04T00:00:00Z").empty
    assert two.delisting_date.iloc[0] == "2026-01-02"


def test_nasdaq_complete_footer_and_secret_csv_preflight(tmp_path):
    body = b"Symbol|Security Name|Test Issue\nA|Fixture|N\nFile Creation Time: 1014202612:00||\n"
    rows = parse_security_snapshot(
        body, source="nasdaq", identities={"A": IDENTITY}, received_at="2026-10-14T12:00:00Z"
    )
    assert len(rows) == 1
    with pytest.raises(ValueError, match="footer"):
        parse_security_snapshot(
            body.split(b"File Creation")[0],
            source="nasdaq",
            identities={"A": IDENTITY},
            received_at="2026-10-14T12:00:00Z",
        )
    with pytest.raises(ValueError):
        collect_security_master(
            RawStore(tmp_path),
            source="alpha_vantage",
            identities={"A": IDENTITY},
            snapshot_id="secret",
            offline=(b"symbol,status,api_key\nA,Active,SECRET_SENTINEL\n", "2026-01-01T00:00:00Z"),
        )
    assert all(b"SECRET_SENTINEL" not in path.read_bytes() for path in tmp_path.rglob("*.bin"))


def test_sec_contact_is_forwarded_to_injected_transport(tmp_path):
    from pfe.data.http import HTTPResponse

    class Fake:
        def get(self, url, *, params, headers, max_bytes):
            assert headers["User-Agent"] == "Feature library contact@example.com"
            return HTTPResponse(sec("A"), "2026-01-01T00:00:00Z")

    result = collect_security_master(
        RawStore(tmp_path),
        source="sec",
        identities={"A": IDENTITY},
        snapshot_id="test",
        user_agent="Feature library contact@example.com",
        transport=Fake(),
    )
    assert result.complete and len(result.history) == 1


def test_universe_rejects_conflicting_source_status_before_filtering():
    current = parse_security_snapshot(
        sec("A"), source="sec", identities={"A": IDENTITY}, received_at="2026-01-01T00:00:00Z"
    )
    delisted = parse_security_snapshot(
        b"symbol,status\nA,Delisted\n",
        source="alpha_vantage",
        identities={"A": IDENTITY},
        received_at="2026-01-02T00:00:00Z",
    )
    history = update_security_master(current, delisted)
    with pytest.raises(ValueError, match="Conflicting"):
        security_universe_asof(history, "2026-01-03T00:00:00Z")
    assert security_universe_asof(history, "2026-01-03T00:00:00Z", source_id="alpha_vantage").empty
