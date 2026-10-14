"""Metadata qualification tests are synthetic and never contact LSE."""

import io
import json
import urllib.error
from dataclasses import asdict

import pytest

from pfe.data.qualification import (
    DISCOVERY_ENDPOINTS,
    REQUIREMENTS,
    BoundedLSETransport,
    DiscoveryError,
    DiscoveryLimitError,
    DiscoveryLimits,
    DiscoveryResponse,
    SymbolRequest,
    collect_lse_discovery,
    qualify_lse_snapshot,
    sanitize_metadata,
)


@pytest.fixture
def discovery_snapshot():
    return {
        "provider": "LSE",
        "audit_id": "synthetic-audit",
        "mode": "offline",
        "audited_at": "2026-10-11T12:00:00Z",
        "catalog_complete": True,
        "catalog_authoritative": True,
        "reference_complete": True,
        "reference_authoritative": True,
        "catalog": [
            {
                "symbol": "QQQ",
                "dataset": "etf",
                "country": "US",
                "unit": "USD",
                "first_tick": "2020-01-01",
                "last_tick": "2026-10-10",
                "ticks": 42,
            },
            {"symbol": "AAPL", "dataset": "stocks", "country": "US"},
        ],
        "reference": [{"dataset": "stock_splits"}, {"dataset": "dividends"}],
        "usage": {"remaining": 10},
        "meta": {},
    }


def test_every_current_requirement_is_explicit_typed_and_json_compatible(discovery_snapshot):
    assert len(REQUIREMENTS) == 34
    assert len({item.requirement_id for item in REQUIREMENTS}) == len(REQUIREMENTS)
    assert all(item.priority in ("P0", "P1", "P2", "P3") and item.models for item in REQUIREMENTS)
    report = qualify_lse_snapshot(discovery_snapshot)
    assert len(report["requirements"]) == 34
    assert all(
        row["status"] in ("available", "absent", "unqualified", "local", "not-applicable")
        for row in report["requirements"]
    )
    assert all(isinstance(row["models"], list) for row in report["requirements"])
    assert all(row["entitlement"] == "unknown" for row in report["requirements"])
    assert json.loads(json.dumps(report)) == report


def test_exact_listed_metadata_does_not_certify_pit_units_or_entitlement(discovery_snapshot):
    report = qualify_lse_snapshot(discovery_snapshot, symbols=("QQQ", "SPY"))
    assert [item["status"] for item in report["requested_symbols"]] == ["available", "absent"]
    qqq, aapl = report["matrix"]
    assert qqq["first"] == "2020-01-01"
    assert qqq["adjustment"]["candles"] == "split_adjusted"
    assert qqq["adjustment"]["dividends"] == "unqualified"
    assert qqq["point_in_time"] == "unqualified"
    assert aapl["unit"] is None and aapl["unit_status"] == "unqualified"
    requirements = {row["requirement_id"]: row for row in report["requirements"]}
    assert requirements["corporate_actions"]["status"] == "available"
    assert requirements["corporate_actions"]["implementation"] == "partial"
    assert report["implementation_baseline"]["revision_id"] == "lot2-2026-10-12"
    assert requirements["benchmark_returns"]["status"] == "unqualified"
    assert requirements["security_master"]["status"] == "unqualified"
    assert requirements["technical"]["status"] == "local"


@pytest.mark.parametrize(
    "change",
    [
        {"catalog_complete": False},
        {"catalog_authoritative": False},
        {"catalog_complete": "true"},
        {"catalog_issues": ["provider_truncated"]},
    ],
)
def test_missing_symbols_not_absent_without_complete_authoritative_catalogue(
    discovery_snapshot, change
):
    report = qualify_lse_snapshot({**discovery_snapshot, **change}, symbols=("MISSING",))
    assert report["requested_symbols"][0]["status"] == "unqualified"


@pytest.mark.parametrize("audited_at", [None, "2026-10-11", "2026-10-11T12:00:00", "broken"])
def test_absence_requires_dated_audit(discovery_snapshot, audited_at):
    with pytest.raises(DiscoveryError, match="timezone-aware timestamp"):
        qualify_lse_snapshot({**discovery_snapshot, "audited_at": audited_at}, symbols=("MISSING",))


def test_codes_case_and_aliases_are_not_guessed(discovery_snapshot):
    report = qualify_lse_snapshot(discovery_snapshot, symbols=("qqq", "Invesco QQQ"))
    assert all(item["status"] == "absent" for item in report["requested_symbols"])
    assert all(item["matches"] == [] for item in report["requested_symbols"])
    assert report["catalogue_coverage"]["absent_scope"] == "exact_provider_code_only"


def test_multiple_datasets_require_pin_and_duplicate_identity_stays_ambiguous(discovery_snapshot):
    discovery_snapshot["catalog"].append({"symbol": "QQQ", "dataset": "options", "country": "US"})
    report = qualify_lse_snapshot(discovery_snapshot, symbols=("QQQ", SymbolRequest("QQQ", "etf")))
    assert [row["status"] for row in report["requested_symbols"]] == ["unqualified", "available"]
    discovery_snapshot["catalog"].append({"symbol": "QQQ", "dataset": "etf", "country": "US"})
    report = qualify_lse_snapshot(discovery_snapshot, symbols=(SymbolRequest("QQQ", "etf"),))
    assert report["requested_symbols"][0]["reason"] == "ambiguous_catalogue_match"


def test_invalid_or_capped_catalogue_never_proves_absence(discovery_snapshot):
    discovery_snapshot["catalog"].append({"symbol": "INVALID"})
    report = qualify_lse_snapshot(discovery_snapshot, symbols=("MISSING",))
    assert report["catalogue_coverage"]["invalid_rows"] == 1
    assert report["requested_symbols"][0]["status"] == "unqualified"
    report = qualify_lse_snapshot(
        discovery_snapshot, symbols=("MISSING",), limits=DiscoveryLimits(max_rows=1)
    )
    assert len(report["matrix"]) == 1
    assert report["catalogue_coverage"]["row_cap_reached"]
    assert report["requested_symbols"][0]["status"] == "unqualified"


def test_country_not_inferred_from_stock_class(discovery_snapshot):
    discovery_snapshot["catalog"][1].pop("country")
    report = qualify_lse_snapshot(discovery_snapshot)
    us = next(row for row in report["requirements"] if row["requirement_id"] == "us_ohlcv")
    assert us["status"] == "unqualified"


@pytest.mark.parametrize(
    "rows,issue",
    [
        ([{"dataset": "dividends"}, {"name": "dividends"}], "duplicate_reference_entry"),
        ([{"dataset": "dividends"}, {"dataset": 3}], "invalid_reference_entry"),
        (["dividends"], "invalid_reference_entry"),
    ],
)
def test_invalid_or_duplicate_reference_rows_override_completeness(discovery_snapshot, rows, issue):
    report = qualify_lse_snapshot({**discovery_snapshot, "reference": rows})
    assert report["reference_coverage"]["complete"] is False
    assert issue in report["reference_coverage"]["issues"]


def test_reference_cap_is_independent_of_instrument_catalogue(discovery_snapshot):
    discovery_snapshot["catalog"] = []
    report = qualify_lse_snapshot(discovery_snapshot, limits=DiscoveryLimits(max_rows=2))
    assert report["catalogue_coverage"]["complete"] is True
    assert report["reference_coverage"]["complete"] is False
    assert report["reference_coverage"]["row_cap_reached"] is True


def test_duplicate_instrument_identity_blocks_other_absence_claims(discovery_snapshot):
    discovery_snapshot["catalog"].append(dict(discovery_snapshot["catalog"][0]))
    report = qualify_lse_snapshot(discovery_snapshot, symbols=("MISSING",))
    assert report["catalogue_coverage"]["complete"] is False
    assert report["catalogue_coverage"]["duplicate_rows"] == 1
    assert report["requested_symbols"][0]["status"] == "unqualified"


def test_malformed_provider_details_do_not_become_qualified_fields(discovery_snapshot):
    discovery_snapshot["catalog"][0].update(
        unit={"unknown": True},
        first_tick="broken",
        last_tick="2026-10-11T12:00:00",
        ticks=True,
        currency="US$",
        timezone="Mars/Missing",
    )
    row = qualify_lse_snapshot(discovery_snapshot)["matrix"][0]
    assert row["unit_status"] == "malformed"
    assert row["first_status"] == "malformed"
    assert row["last_status"] == "unqualified_timezone"
    assert row["ticks_status"] == "malformed"
    assert row["currency_status"] == "malformed"
    assert row["timezone_status"] == "malformed"
    assert row["unit"] == {"unknown": True}  # Original provider claim remains inspectable.
    discovery_snapshot["catalog"][0].update(currency="USD", timezone="Europe/Paris", ticks=3)
    row = qualify_lse_snapshot(discovery_snapshot)["matrix"][0]
    assert (
        row["currency_status"]
        == row["timezone_status"]
        == row["ticks_status"]
        == "provider_reported"
    )


def test_snapshot_byte_limit_and_nonfinite_metadata(discovery_snapshot):
    with pytest.raises(DiscoveryLimitError, match="max_bytes"):
        qualify_lse_snapshot(discovery_snapshot, limits=DiscoveryLimits(max_bytes=20))
    with pytest.raises(DiscoveryError, match="finite"):
        qualify_lse_snapshot({**discovery_snapshot, "usage": {"value": float("nan")}})


class FakeTransport:
    def __init__(self, catalog=None, *, failure=None):
        self.calls = []
        self.failure = failure
        self.responses = {
            "/meta": {"datasets": ["stocks"]},
            "/catalog": catalog
            if catalog is not None
            else [{"symbol": "AAPL", "dataset": "stocks", "country": "US"}],
            "/reference": [{"dataset": "stock_splits"}],
            "/usage": {"remaining": 2},
        }

    def get(self, endpoint, *, max_bytes):
        self.calls.append(endpoint)
        if endpoint == self.failure:
            raise DiscoveryError("Synthetic metadata failure")
        payload = self.responses[endpoint]
        count = len(json.dumps(payload).encode())
        if count > max_bytes:
            raise DiscoveryLimitError("Synthetic response exceeds max_bytes")
        return DiscoveryResponse(payload, count)


def test_metadata_only_collection_and_resume_keep_original_endpoint_pages():
    first = FakeTransport(failure="/reference")
    pages = {}
    with pytest.raises(DiscoveryError, match="Synthetic metadata failure"):
        collect_lse_discovery(
            first,
            audit_id="test-1",
            audited_at="2026-10-11T12:00:00Z",
            on_page=lambda name, page: pages.update({name: page}),
        )
    assert set(pages) == {"meta", "catalog"}
    original = json.dumps(pages, sort_keys=True)
    resumed = FakeTransport()
    snapshot = collect_lse_discovery(
        resumed, audit_id="test-1", audited_at="2026-10-11T12:00:00Z", cached_pages=pages
    )
    assert resumed.calls == ["/reference", "/usage"]
    assert snapshot["catalog_complete"] is True
    assert snapshot["catalog_authoritative"] is True
    assert snapshot["requests_this_run"] == 2
    assert json.dumps(pages, sort_keys=True) == original
    assert set(first.calls + resumed.calls) <= set(DISCOVERY_ENDPOINTS)


@pytest.mark.parametrize(
    "catalog, issue",
    [
        (
            {"rows": [], "complete": True, "authoritative": True, "next_cursor": "page2"},
            "pagination_not_followed",
        ),
        (
            {"rows": [], "complete": True, "authoritative": True, "next_page_token": "page2"},
            "pagination_not_followed",
        ),
        (
            {"rows": [], "complete": True, "authoritative": True, "total": 2},
            "provider_count_mismatch",
        ),
        (
            {"rows": [], "complete": True, "authoritative": True, "truncated": True},
            "provider_truncated",
        ),
    ],
)
def test_unknown_pagination_never_followed_or_marked_complete(catalog, issue):
    fake = FakeTransport(catalog=catalog)
    snapshot = collect_lse_discovery(fake, audit_id="test", audited_at="2026-10-11T12:00:00Z")
    assert snapshot["catalog_complete"] is False
    assert issue in snapshot["catalog_issues"]
    assert fake.calls == list(DISCOVERY_ENDPOINTS)


def test_row_cap_keeps_partial_catalogue_and_not_absent_after_resume():
    pages = {}
    snapshot = collect_lse_discovery(
        FakeTransport(),
        audit_id="cap",
        audited_at="2026-10-11T12:00:00Z",
        limits=DiscoveryLimits(max_rows=1),
        on_page=lambda name, page: pages.update({name: page}),
    )
    assert snapshot["catalog_complete"] is False
    resumed = collect_lse_discovery(
        FakeTransport(),
        audit_id="cap",
        audited_at="2026-10-11T12:00:00Z",
        limits=DiscoveryLimits(max_rows=1),
        cached_pages=pages,
    )
    assert resumed["catalog_complete"] is False
    report = qualify_lse_snapshot(resumed, symbols=("MISSING",))
    assert report["requested_symbols"][0]["status"] == "unqualified"


@pytest.mark.parametrize("field,value", [("captured_at", None), ("coverage", None)])
def test_invalid_cached_page_fails_without_refetch(field, value):
    pages = {}
    collect_lse_discovery(
        FakeTransport(),
        audit_id="cached",
        audited_at="2026-10-11T12:00:00Z",
        on_page=lambda name, page: pages.update({name: page}),
    )
    pages["catalog"][field] = value
    fake = FakeTransport()
    with pytest.raises(DiscoveryError):
        collect_lse_discovery(
            fake,
            audit_id="cached",
            audited_at="2026-10-11T12:00:00Z",
            cached_pages=pages,
        )
    assert fake.calls == []


def test_request_and_aggregate_byte_budgets_leave_earlier_pages_resumable():
    pages = {}
    with pytest.raises(DiscoveryLimitError, match="max_requests"):
        collect_lse_discovery(
            FakeTransport(),
            audit_id="budget",
            audited_at="2026-10-11T12:00:00Z",
            limits=DiscoveryLimits(max_requests=1),
            on_page=lambda name, page: pages.update({name: page}),
        )
    assert set(pages) == {"meta"}
    with pytest.raises(DiscoveryLimitError, match="max_bytes"):
        collect_lse_discovery(
            FakeTransport(),
            audit_id="budget",
            audited_at="2026-10-11T12:00:00Z",
            limits=DiscoveryLimits(max_bytes=25),
        )


@pytest.mark.parametrize(
    "field,value", [("max_rows", 0), ("max_bytes", True), ("max_requests", -1)]
)
def test_invalid_budgets(field, value):
    with pytest.raises(ValueError, match=field):
        DiscoveryLimits(**{field: value})


def test_http_transport_caps_before_decode_and_redacts_known_keys():
    calls = []

    class FakeOpener:
        def open(self, request, *, timeout):
            calls.append(request)
            assert timeout == 30
            return io.BytesIO(
                json.dumps(
                    {
                        "api_key": "synthetic-secret",
                        "remaining": 3,
                        "echo": "synthetic-secret",
                        "quota_for_synthetic-secret-owner": 1,
                    }
                ).encode()
            )

    transport = BoundedLSETransport("synthetic-secret", opener=FakeOpener())
    result = transport.get("/usage", max_bytes=200)
    assert "synthetic-secret" not in json.dumps(result.payload)
    assert result.payload == {
        "remaining": 3,
        "echo": "[redacted]",
        "quota_for_[redacted]-owner": 1,
    }
    assert calls[0].full_url.endswith("/vault/usage")
    assert calls[0].get_method() == "GET"
    with pytest.raises(DiscoveryLimitError, match="max_bytes"):
        transport.get("/catalog", max_bytes=5)
    with pytest.raises(DiscoveryError, match="Only LSE discovery"):
        transport.get("/export", max_bytes=200)
    assert len(calls) == 2
    assert "synthetic-secret" not in repr(transport)


def test_http_errors_do_not_echo_provider_body_or_key():
    class BadOpener:
        def open(self, request, *, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 403, "synthetic-secret", {}, io.BytesIO(b"synthetic-secret")
            )

    with pytest.raises(DiscoveryError) as error:
        BoundedLSETransport("synthetic-secret", opener=BadOpener()).get("/usage", max_bytes=200)
    assert str(error.value) == "LSE metadata HTTP 403"


def test_header_validation_never_echoes_a_malformed_key():
    with pytest.raises(ValueError) as error:
        BoundedLSETransport("synthetic-secret\ninvalid")
    assert str(error.value) == "LSE_API_KEY must not contain control characters"


def test_credential_sanitizer_preserves_opaque_pagination_only():
    result = sanitize_metadata(
        {
            "API-Key": "one",
            "nested": {"clientSecret": "two"},
            "next_page_token": "opaque",
            "remaining": 1,
        }
    )
    assert result == {"nested": {}, "next_page_token": "opaque", "remaining": 1}
    assert asdict(SymbolRequest("USD/JPY", "forex")) == {"symbol": "USD/JPY", "dataset": "forex"}
