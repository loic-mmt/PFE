"""Source rights are explicit and declarations cannot silently change across runs."""

from datetime import UTC, datetime

import pytest

from pfe.data.sources import SourceRecord, SourceRegistry, default_source_registry


def record(**overrides):
    return SourceRecord(
        **{
            "source_id": "test",
            "revision_id": "v1",
            "name": "Test provider",
            "url": "https://example.com/data",
            **overrides,
        }
    )


def test_unknown_rights_stay_unknown_and_json_roundtrip():
    registry = SourceRegistry((record(),))
    source = registry.get("test")
    assert source.allowed_uses == ()
    assert source.redistribution_allowed is None
    assert source.availability == "unknown"
    assert SourceRegistry.from_dict(registry.to_dict()).get("test") == source


def test_local_source_can_have_no_public_url():
    assert record(access="local", url=None).url is None
    with pytest.raises(ValueError, match="url"):
        record(url=None)


def test_immutable_revision_idempotent_replay_and_explicit_version_access():
    older = record(verified_at="2026-01-01T01:00:00+01:00")
    newer = record(revision_id="v2", verified_at="2026-02-01T00:00:00Z", access="licensed")
    registry = SourceRegistry((older,))
    assert registry.register(older) == older
    with pytest.raises(ValueError, match="different content"):
        registry.register(record(access="free"))
    registry.register(newer)
    assert registry.get("test") == newer
    assert registry.get("test", "v1").verified_at == datetime(2026, 1, 1, tzinfo=UTC)


def test_latest_requires_known_unambiguous_audit_order():
    registry = SourceRegistry((record(), record(revision_id="v2")))
    with pytest.raises(ValueError, match="audit order"):
        registry.get("test")
    timestamp = "2026-01-01T00:00:00Z"
    registry = SourceRegistry(
        (record(verified_at=timestamp), record(revision_id="v2", verified_at=timestamp))
    )
    with pytest.raises(ValueError, match="simultaneous"):
        registry.get("test")


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_id": " padded "},
        {"verified_at": "2026-01-01"},
        {"url": "https://user:secret@example.com/data"},
        {"url": "https://example.com/data?api_key=secret"},
        {"license_url": "file:///secret"},
        {"access": "guaranteed"},
        {"availability": "date"},
        {"allowed_uses": "research"},
        {"allowed_uses": ["research", "research"]},
        {"redistribution_allowed": 1},
    ],
)
def test_bad_declarations_rejected(overrides):
    with pytest.raises(ValueError):
        record(**overrides)


def test_default_lse_metadata_separates_sdk_data_license():
    source = default_source_registry().get("lse")
    assert source.source_id == "lse"
    assert source.redistribution_allowed is False
    assert source.availability == "receipt"
    assert "MIT" in source.license_notes
    assert "commercial" in source.allowed_uses


def test_bounded_registry_load(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text('{"schema_version":"1","sources":[]}')
    assert SourceRegistry.load(path).to_dict()["sources"] == []
    with pytest.raises(ValueError, match="max_bytes"):
        SourceRegistry.load(path, max_bytes=8)
