"""SEC acceptance, bounded injected transport and immutable snapshot replay."""

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pfe.data.http import HTTPResponse
from pfe.data.sec import (
    SECClient,
    merge_company_facts,
    normalize_company_facts,
    normalize_sec_filings,
)
from pfe.data.storage import RawStore

ACCN = "0000001234-25-000001"
NEW = "0000001234-25-000002"
RECEIPT = "2026-10-12T10:00:00Z"


def submissions(acceptance="2025-05-01T16:00:00-04:00", accession=ACCN, history=False):
    recent = dict(
        accessionNumber=[accession],
        filingDate=["2025-05-01"],
        reportDate=["2025-03-31"],
        acceptanceDateTime=[acceptance],
        form=["10-Q"],
        primaryDocument=["report.htm"],
    )
    return dict(
        cik=1234,
        filings=dict(
            recent=recent,
            files=[dict(name="CIK0000001234-submissions-001.json")] if history else [],
        ),
    )


def companyfacts(records=None):
    fact = dict(
        start="2025-01-01",
        end="2025-03-31",
        val=100.0,
        accn=ACCN,
        filed="2025-05-01",
        fy=2025,
        fp="Q1",
        form="10-Q",
    )
    return dict(
        cik=1234,
        facts={"us-gaap": {"Revenues": {"units": {"USD": [fact] if records is None else records}}}},
    )


class Fake:
    def __init__(self, *, history=False, fail_facts=False):
        self.history, self.fail_facts, self.calls = history, fail_facts, []

    def get(self, url, *, max_bytes, headers):
        self.calls.append((url, max_bytes, headers))
        if url.endswith("companyfacts/CIK0000001234.json"):
            if self.fail_facts:
                self.fail_facts = False
                raise ValueError("temporary fixture failure")
            payload = companyfacts()
        elif url.endswith("-submissions-001.json"):
            payload = submissions(acceptance=None, accession=NEW)["filings"]["recent"]
        else:
            payload = submissions(history=self.history)
        return HTTPResponse(json.dumps(payload).encode(), RECEIPT, "application/json")


def test_acceptance_timezone_distinct_from_publication_period_and_filed_date():
    filings = normalize_sec_filings(submissions(), received_at=RECEIPT)
    assert filings.accepted_at.iloc[0] == pd.Timestamp("2025-05-01T20:00:00Z")
    assert filings.published_at.isna().all()
    assert filings.available_at.iloc[0] == pd.Timestamp(RECEIPT)
    assert filings.reportDate.iloc[0] == "2025-03-31"
    assert not filings.point_in_time.any()
    facts = normalize_company_facts(
        companyfacts(),
        received_at=RECEIPT,
        filings=filings,
        availability="publication",
        latency="60s",
    )
    assert facts.available_at.iloc[0] == pd.Timestamp("2025-05-01T20:01:00Z")
    assert facts.accepted_at.iloc[0] == pd.Timestamp("2025-05-01T20:00:00Z")
    assert facts.published_at.isna().all()
    assert facts["quality.missing_publication"].all()
    assert facts.ingested_at.iloc[0] == pd.Timestamp(RECEIPT)
    assert facts.availability_basis.iloc[0] == "reconstructed"
    assert not facts.point_in_time.any()
    assert facts.period_kind.iloc[0] == "quarter"
    assert facts.filing_fiscal_year.iloc[0] == 2025
    assert facts.value.iloc[0] == 100
    assert facts.unit.iloc[0] == "USD"


@pytest.mark.parametrize("acceptance", [None, "2025-05-01", "2025-05-01T16:00:00"])
def test_missing_or_naive_acceptance_defaults_to_receipt(acceptance):
    filings = normalize_sec_filings(
        submissions(acceptance), received_at=RECEIPT, availability="publication"
    )
    assert filings.published_at.isna().all()
    assert filings.accepted_at.isna().all()
    assert filings.available_at.eq(pd.Timestamp(RECEIPT)).all()
    facts = normalize_company_facts(
        companyfacts(), received_at=RECEIPT, filings=filings, availability="publication"
    )
    assert facts.published_at.isna().all()
    assert facts.accepted_at.isna().all()
    assert facts.available_at.eq(pd.Timestamp(RECEIPT)).all()
    assert facts["quality.missing_publication"].all()


def test_after_cutoff_acceptance_does_not_invent_publication_or_next_business_day():
    filings = normalize_sec_filings(
        submissions("2025-05-02T18:00:00-04:00"),
        received_at=RECEIPT,
        availability="publication",
        latency="60s",
    )
    assert filings.accepted_at.iloc[0] == pd.Timestamp("2025-05-02T22:00:00Z")
    assert filings.published_at.isna().all()
    assert filings.available_at.iloc[0] == pd.Timestamp("2025-05-02T22:01:00Z")
    assert filings.availability_basis.iloc[0] == "reconstructed"
    assert not filings.point_in_time.any()


def test_verified_publication_is_explicit_and_distinct_from_acceptance():
    filings = normalize_sec_filings(
        submissions("2025-05-02T18:00:00-04:00"),
        received_at=RECEIPT,
        publication_times={ACCN: "2025-05-05T10:00:00-04:00"},
    )
    assert filings.accepted_at.iloc[0] == pd.Timestamp("2025-05-02T22:00:00Z")
    assert filings.published_at.iloc[0] == pd.Timestamp("2025-05-05T14:00:00Z")
    facts = normalize_company_facts(
        companyfacts(), received_at=RECEIPT, filings=filings, availability="publication"
    )
    assert facts.published_at.iloc[0] == pd.Timestamp("2025-05-05T14:00:00Z")
    assert facts.available_at.iloc[0] == pd.Timestamp("2025-05-05T14:00:00Z")
    assert facts.accepted_at.iloc[0] == pd.Timestamp("2025-05-02T22:00:00Z")
    assert not facts["quality.missing_publication"].any()
    with pytest.raises(ValueError, match="timezone"):
        normalize_sec_filings(
            submissions(), received_at=RECEIPT, publication_times={ACCN: "2025-05-01"}
        )


def test_restatement_versions_publication_mode_and_receipt_latest_snapshot():
    old = companyfacts()["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]
    revised = old | dict(val=105, accn=NEW, filed="2025-06-01")
    filings = pd.concat(
        [
            normalize_sec_filings(submissions(), received_at=RECEIPT),
            normalize_sec_filings(submissions("2025-06-01T20:00:00Z", NEW), received_at=RECEIPT),
        ],
        ignore_index=True,
    )
    records = companyfacts([old, revised, old.copy()])
    historical = normalize_company_facts(
        records, received_at=RECEIPT, filings=filings, availability="publication"
    )
    assert historical.value.tolist() == [100, 105]
    assert historical.revision_id.nunique() == 2
    current = normalize_company_facts(records, received_at=RECEIPT, filings=filings)
    assert current.value.tolist() == [105]
    assert current.available_at.iloc[0] == pd.Timestamp(RECEIPT)


def test_context_units_annual_ytd_eps_and_segments_remain_distinct():
    payload = companyfacts()
    old = payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]
    payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"] += [
        old | dict(start="2024-04-01", end="2025-03-31", val=400),
        old | dict(start="2024-10-01", val=200),
        old | dict(segment="Europe", val=20),
    ]
    payload["facts"]["us-gaap"]["Revenues"]["units"]["EUR"] = [old | dict(val=90)]
    payload["facts"]["us-gaap"]["EarningsPerShareBasic"] = dict(
        units={"USD/shares": [old | dict(val=1)]}
    )
    payload["facts"]["us-gaap"]["EarningsPerShareDiluted"] = dict(
        units={"USD/shares": [old | dict(val=0.9)]}
    )
    out = normalize_company_facts(payload, received_at=RECEIPT)
    assert {"quarter", "annual", "ytd"}.issubset(set(out.period_kind))
    assert {"basic", "diluted"}.issubset(set(out.eps_basis))
    assert out.loc[out.entity_scope == "segment", "context_json"].str.contains("Europe").all()
    assert set(out.currency) == {"USD", "EUR"}
    assert not out.coverage_complete.any()
    assert len(out) == 7


def test_incompatible_values_and_invalid_shapes_rejected():
    old = companyfacts()["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]
    for records, match in [
        ([old, old | dict(val=200)], "Conflicting"),
        ([old | dict(val=True)], "finite"),
        ([old | dict(start="2025-04-01")], "precede"),
        ([old | dict(accn="bad")], "accession"),
    ]:
        with pytest.raises(ValueError, match=match):
            normalize_company_facts(companyfacts(records), received_at=RECEIPT)
    bad = submissions()
    bad["filings"]["recent"]["form"] = []
    with pytest.raises(ValueError, match="equal lengths"):
        normalize_sec_filings(bad, received_at=RECEIPT)


def test_injected_download_raw_bytes_cache_history_and_caps(tmp_path):
    fake = Fake(history=True)
    client = SECClient("PFE research contact@example.com", transport=fake, store=tmp_path)
    data = client.download("1234", audit_id="fixture", max_pages=3)
    assert data.complete
    assert len(data.filings) == 2
    assert len(fake.calls) == 3
    assert fake.calls[0][2]["User-Agent"].endswith("contact@example.com")
    store = RawStore(tmp_path)
    assert json.loads(store.read_page(data.collection_id, "companyfacts")) == companyfacts()
    original = store.manifest(data.collection_id)
    replay = client.download("1234", audit_id="fixture", max_pages=3)
    assert len(fake.calls) == 3
    assert_frame_equal(replay.facts, data.facts)
    assert store.manifest(data.collection_id) == original
    capped = client.download("1234", audit_id="capped", max_pages=2)
    assert not capped.complete
    assert not store.manifest(capped.collection_id)["complete"]


def test_interrupted_download_resumes_cached_pages_without_duplicate_http(tmp_path):
    fake = Fake(fail_facts=True)
    client = SECClient("PFE contact@example.com", transport=fake, store=tmp_path)
    with pytest.raises(ValueError, match="temporary"):
        client.download(1234, audit_id="recovery")
    recovered = client.download(1234, audit_id="recovery")
    assert recovered.complete
    assert len(fake.calls) == 3
    assert sum(url.endswith("submissions/CIK0000001234.json") for url, _, _ in fake.calls) == 1


def test_throttle_is_strictly_below_ten_per_second(tmp_path, monkeypatch):
    import pfe.data.sec as module

    clock, waits = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def sleep(delay):
        waits.append(delay)
        clock[0] += delay

    monkeypatch.setattr(module.time, "sleep", sleep)
    SECClient("PFE contact@example.com", transport=Fake(), store=tmp_path).download(
        1234, audit_id="rate"
    )
    assert waits == [pytest.approx(0.125)]


@pytest.mark.parametrize(
    "options",
    [
        dict(user_agent="PFE"),
        dict(user_agent="PFE c@example.com\n"),
        dict(user_agent="PFE c@example.com", max_requests_per_second=10),
        dict(user_agent="PFE c@example.com", max_requests_per_second="8"),
    ],
)
def test_invalid_access_configuration(options, tmp_path):
    with pytest.raises(ValueError):
        SECClient(**options, transport=Fake(), store=tmp_path)


def test_byte_cap_and_history_path_guard(tmp_path):
    client = SECClient("PFE c@example.com", transport=Fake(), store=tmp_path, max_bytes=10)
    with pytest.raises(ValueError, match="max_bytes"):
        client.download(1234, audit_id="cap")

    class Malicious(Fake):
        def get(self, url, **options):
            payload = submissions(history=True)
            payload["filings"]["files"][0]["name"] = "../../other.json"
            return HTTPResponse(json.dumps(payload).encode(), RECEIPT)

    with pytest.raises(ValueError, match="filename"):
        SECClient("PFE c@example.com", transport=Malicious(), store=tmp_path).download(
            1234, audit_id="path"
        )


def test_merge_audits_keeps_original_receipt_and_new_value_revision():
    first = normalize_company_facts(companyfacts(), received_at="2025-05-02T10:00:00Z")
    later_payload = companyfacts()
    original = later_payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]
    later_payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
        original | dict(val=105, accn=NEW, filed="2025-06-01")
    )
    later = normalize_company_facts(later_payload, received_at="2025-06-02T10:00:00Z")
    repeated = normalize_company_facts(companyfacts(), received_at="2025-06-02T10:00:00Z")
    merged = merge_company_facts(first, repeated, later)
    assert merged.value.tolist() == [100, 105]
    assert merged.available_at.tolist() == [
        pd.Timestamp("2025-05-02T10:00:00Z"),
        pd.Timestamp("2025-06-02T10:00:00Z"),
    ]
    assert_frame_equal(merge_company_facts(first, repeated), first)
    from pfe.data.contracts import observations_asof

    before = observations_asof(merged, "2025-05-31T23:59:59Z")
    assert before.value.tolist() == [100]
    assert observations_asof(merged, "2025-06-02T10:00:00Z").value.tolist() == [105]


def test_merge_policies_and_conflicting_revision_content_rejected():
    original = normalize_company_facts(companyfacts(), received_at=RECEIPT)
    changed = original.copy()
    changed["value"] = 200
    with pytest.raises(ValueError, match="immutable value/context"):
        merge_company_facts(original, changed)
    filings = normalize_sec_filings(submissions(), received_at=RECEIPT)
    reconstructed = normalize_company_facts(
        companyfacts(), received_at=RECEIPT, filings=filings, availability="publication"
    )
    with pytest.raises(ValueError, match="Cannot mix"):
        merge_company_facts(original, reconstructed)
    changed_acceptance = reconstructed.copy()
    changed_acceptance["accepted_at"] += pd.Timedelta("1s")
    with pytest.raises(ValueError, match="immutable value/context"):
        merge_company_facts(reconstructed, changed_acceptance)


def test_empty_explicit_identity_and_bad_mode_fail_without_http(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        normalize_sec_filings(submissions(), received_at=RECEIPT, entity_id="")
    with pytest.raises(ValueError, match="non-empty"):
        normalize_company_facts(companyfacts(), received_at=RECEIPT, entity_id="")
    fake = Fake()
    with pytest.raises(ValueError, match="availability"):
        SECClient("PFE c@example.com", transport=fake, store=tmp_path).download(
            1234, audit_id="invalid-policy", availability="unknown"
        )
    assert fake.calls == []
