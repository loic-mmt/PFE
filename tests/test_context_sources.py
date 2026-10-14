"""Synthetic EIA, VIX and Pink Sheet source contracts; no real HTTP calls."""

import io
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from pfe.data.cboe import collect_cboe_vix, parse_cboe_vix_csv
from pfe.data.contracts import observations_asof
from pfe.data.eia import (
    ContextError,
    ContextLimitError,
    ContextLimits,
    EIAQuery,
    collect_eia,
    parse_eia_page,
)
from pfe.data.storage import RawStore
from pfe.data.world_bank import (
    CommodityColumn,
    collect_pink_sheet,
    parse_pink_sheet,
    parse_pink_sheet_xlsx,
    pink_sheet_values_asof,
)


def _eia(rows, total=None, frequency="monthly"):
    return {
        "apiVersion": "2.1.12",
        "response": {
            "data": rows,
            "frequency": frequency,
            "total": str(len(rows) if total is None else total),
        },
    }


@pytest.fixture
def eia_rows():
    return [
        {"period": month, "value": value, "value-units": "Dollars per Barrel"}
        for month, value in (("2026-06", "70.5"), ("2026-07", "-2.3"), ("2026-08", "NA"))
    ]


def test_eia_period_units_numeric_strings_and_receipt_not_release(eia_rows):
    query = EIAQuery("petroleum/pri/spt", "declared_wti", frequency="monthly")
    rows = parse_eia_page(_eia(eia_rows), query=query, received_at="2026-09-03T12:00:00Z")
    assert list(rows.observation_date) == ["2026-06-01", "2026-07-01", "2026-08-01"]
    assert rows.period_end.iloc[0] == "2026-06-30"
    assert list(rows.value[:2]) == [70.5, -2.3] and rows.value.iloc[2] is None
    assert rows.unit.eq("Dollars per Barrel").all()
    assert rows.published_at.isna().all()
    assert rows.available_at.eq(pd.Timestamp("2026-09-03T12:00:00Z")).all()
    assert not rows.point_in_time.any()
    assert observations_asof(rows, "2026-09-03T11:59:59Z").empty


def test_eia_explicit_release_time_vs_date_only(eia_rows):
    query = EIAQuery(
        "petroleum/pri/spt", "energy", frequency="monthly", published_column="released"
    )
    payload = _eia([{**eia_rows[0], "released": "2026-07-01"}])
    result = parse_eia_page(payload, query=query, received_at="2026-09-03T12:00:00Z")
    assert result.published_at.isna().all() and result.publication_date.iloc[0] == "2026-07-01"
    assert result.available_at.iloc[0] == pd.Timestamp("2026-09-03T12:00:00Z")
    publication = EIAQuery(
        "petroleum/pri/spt",
        "energy",
        frequency="monthly",
        published_column="released",
        availability_basis="publication",
    )
    with pytest.raises(ContextError, match="aware release"):
        parse_eia_page(payload, query=publication, received_at="2026-09-03T12:00:00Z")
    payload["response"]["data"][0]["released"] = "2026-07-01T14:30:00Z"
    result = parse_eia_page(payload, query=publication, received_at="2026-09-03T12:00:00Z")
    assert result.available_at.iloc[0] == pd.Timestamp("2026-07-01T14:30:00Z")
    assert not result.point_in_time.any()  # Latest values never certify that original vintage.


@pytest.mark.parametrize("value", [True, float("inf"), "12,5", "$42", "bad"])
def test_eia_invalid_value_no_decimal_currency_or_unit_guess(value):
    with pytest.raises(ContextError):
        parse_eia_page(
            _eia([{"period": "2026-06", "value": value}]),
            query=EIAQuery("energy", "factor", frequency="monthly"),
            received_at="2026-09-03T12:00:00Z",
        )


def test_eia_pagination_resumption_immutable_receipts_and_secret_redaction(eia_rows, tmp_path):
    calls, fail = [], True

    class FakeHTTP:
        def get(self, url, *, params, max_bytes):
            calls.append(params["offset"])
            assert params["api_key"] == "synthetic-secret" and params["length"] == 2
            if params["offset"] == 2 and fail:
                raise OSError("Synthetic interrupted response")
            payload = _eia(eia_rows[params["offset"] : params["offset"] + 2], total=3)
            payload["request"] = {"params": {"api_key": "synthetic-secret"}}
            payload["echo"] = "synthetic-secret"
            return SimpleNamespace(
                body=json.dumps(payload).encode(),
                received_at=f"2026-09-03T1{2 + params['offset']}:00:00Z",
            )

    store = RawStore(tmp_path, max_page_bytes=1_048_576)
    query = EIAQuery(
        "petroleum/pri/spt", "declared_energy", frequency="monthly", facets={"series": ["RWTC"]}
    )
    with pytest.raises(OSError):
        collect_eia(
            FakeHTTP(),
            store,
            query=query,
            snapshot_id="first",
            api_key="synthetic-secret",
            limits=ContextLimits(page_size=2),
        )
    fail = False
    result = collect_eia(
        FakeHTTP(),
        store,
        query=query,
        snapshot_id="first",
        api_key="synthetic-secret",
        limits=ContextLimits(page_size=2),
    )
    assert calls == [0, 2, 2] and result.complete and len(result.observations) == 3
    manifest = store.manifest(result.collection_id)
    replay = collect_eia(
        None, store, query=query, snapshot_id="first", limits=ContextLimits(page_size=2)
    )
    assert replay.complete and store.manifest(result.collection_id) == manifest
    assert len(replay.observations) == 3
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"synthetic-secret" not in path.read_bytes()


def test_eia_request_caps_are_explicit_and_resume_pagination(eia_rows, tmp_path):
    calls = []

    class FakeHTTP:
        def get(self, url, *, params, max_bytes):
            calls.append(params["offset"])
            payload = _eia(eia_rows[params["offset"] : params["offset"] + 2], total=3)
            return SimpleNamespace(
                body=json.dumps(payload).encode(), received_at="2026-09-03T12:00:00Z"
            )

    store = RawStore(tmp_path)
    query = EIAQuery("energy", "factor", frequency="monthly")
    limits = ContextLimits(page_size=2, max_requests=1)
    first = collect_eia(
        FakeHTTP(), store, query=query, snapshot_id="bounded", api_key="fake", limits=limits
    )
    assert not first.complete and first.issues == ("request_cap_reached",)
    assert not store.manifest(first.collection_id)["complete"]
    second = collect_eia(
        FakeHTTP(), store, query=query, snapshot_id="bounded", api_key="fake", limits=limits
    )
    assert second.complete and calls == [0, 2]


def test_eia_changing_provider_total_never_marks_complete(eia_rows, tmp_path):
    class FakeHTTP:
        def get(self, url, *, params, max_bytes):
            offset = params["offset"]
            payload = _eia(eia_rows[offset : offset + 2], total=3 if offset == 0 else 2)
            return SimpleNamespace(
                body=json.dumps(payload).encode(), received_at="2026-09-03T12:00:00Z"
            )

    result = collect_eia(
        FakeHTTP(),
        RawStore(tmp_path),
        query=EIAQuery("energy", "factor", frequency="monthly"),
        snapshot_id="changed",
        api_key="fake",
        limits=ContextLimits(page_size=2),
    )
    assert not result.complete and "provider_total_changed" in result.issues


def test_eia_offline_pages_are_bounded_and_replay_without_network(eia_rows, tmp_path):
    pages = [{"payload": _eia(eia_rows[:2], total=3), "received_at": "2026-09-03T12:00:00Z"}]
    result = collect_eia(
        None,
        RawStore(tmp_path),
        query=EIAQuery("energy", "factor", frequency="monthly"),
        snapshot_id="offline",
        offline_pages=pages,
    )
    assert not result.complete and "offline_pages_incomplete" in result.issues


def test_eia_multiple_facet_codes_or_disjoint_series_cannot_merge():
    with pytest.raises(ContextError, match="one explicit code"):
        EIAQuery("energy", "factor", facets={"series": ["RWTC", "RBRTE"]})
    payload = _eia(
        [
            {"period": "2026-06", "value": "70", "series": "RWTC"},
            {"period": "2026-07", "value": "80", "series": "RBRTE"},
        ]
    )
    with pytest.raises(ContextError, match="multiple dimensional"):
        parse_eia_page(
            payload,
            query=EIAQuery("energy", "factor", frequency="monthly"),
            received_at="2026-09-03T12:00:00Z",
        )
    with pytest.raises(ContextError, match="facet metadata"):
        parse_eia_page(
            payload,
            query=EIAQuery("energy", "factor", frequency="monthly", facets={"series": ["RWTC"]}),
            received_at="2026-09-03T12:00:00Z",
        )


def test_eia_disjoint_series_on_separate_pages_cannot_merge(tmp_path):
    pages = [
        {
            "payload": _eia([{"period": "2026-06", "value": "70", "series": "RWTC"}], total=2),
            "received_at": "2026-09-03T12:00:00Z",
        },
        {
            "payload": _eia([{"period": "2026-07", "value": "80", "series": "RBRTE"}], total=2),
            "received_at": "2026-09-03T13:00:00Z",
        },
    ]
    with pytest.raises(ContextError, match="changes between pages"):
        collect_eia(
            None,
            RawStore(tmp_path),
            query=EIAQuery("energy", "factor", frequency="monthly"),
            snapshot_id="disjoint",
            offline_pages=pages,
            limits=ContextLimits(page_size=1),
        )


@pytest.mark.parametrize(
    "frequency,start,end,before,after",
    [
        ("daily", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-04"),
        ("weekly", "2026-01-09", "2026-01-16", "2026-01-02", "2026-01-23"),
        ("monthly", "2026-01", "2026-02", "2025-12", "2026-03"),
        ("quarterly", "2026-Q1", "2026-Q2", "2025-Q4", "2026-Q3"),
        ("annual", "2025", "2026", "2024", "2027"),
    ],
)
def test_eia_query_bounds_are_inclusive_and_provider_rows_cannot_escape(
    frequency, start, end, before, after
):
    query = EIAQuery("energy", "factor", frequency=frequency, start=start, end=end)
    accepted = parse_eia_page(
        _eia([{"period": start, "value": "1"}, {"period": end, "value": "2"}], frequency=frequency),
        query=query,
        received_at="2026-10-12T12:00:00Z",
    )
    assert list(accepted.observation_period) == [start, end]
    for period in (before, after):
        with pytest.raises(ContextError, match="inclusive query bounds"):
            parse_eia_page(
                _eia([{"period": period, "value": "1"}], frequency=frequency),
                query=query,
                received_at="2026-10-12T12:00:00Z",
            )


def test_eia_out_of_range_page_never_completes_collection(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(ContextError, match="inclusive query bounds"):
        collect_eia(
            None,
            store,
            query=EIAQuery("energy", "factor", start="2026-01-01", end="2026-01-31"),
            snapshot_id="outside-window",
            offline_pages=[
                {
                    "payload": _eia([{"period": "2025-01-01", "value": "1"}], frequency="daily"),
                    "received_at": "2026-10-12T12:00:00Z",
                }
            ],
        )
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    assert not json.loads(manifests[0].read_bytes())["complete"]


VIX = b"DATE,OPEN,HIGH,LOW,CLOSE\n09/03/2026,16,18,15,17\n09/04/2026,17,19,16,18\n"


def test_vix_economic_date_daily_index_points_and_receipt():
    result = parse_cboe_vix_csv(VIX, received_at="2026-09-05T12:00:00Z")
    assert list(result.observation_date) == ["2026-09-03", "2026-09-04"]
    assert result.unit.eq("index_points").all() and result.frequency.eq("daily").all()
    assert result.currency.isna().all() and result.published_at.isna().all()
    assert result.available_at.eq(pd.Timestamp("2026-09-05T12:00:00Z")).all()
    assert not result.point_in_time.any()
    assert observations_asof(result, "2026-09-04T23:59:59Z").empty


def test_vix_reconstructed_close_requires_explicit_schedule_and_never_pit():
    with pytest.raises(ContextError, match="explicit date"):
        parse_cboe_vix_csv(
            VIX, received_at="2026-09-05T12:00:00Z", availability_basis="reconstructed"
        )
    result = parse_cboe_vix_csv(
        VIX,
        received_at="2026-09-05T12:00:00Z",
        availability_basis="reconstructed",
        close_schedule={"2026-09-03": "2026-09-03T20:15:00Z", "2026-09-04": "2026-09-04T20:15:00Z"},
    )
    assert result.available_at.iloc[0] == pd.Timestamp("2026-09-03T20:15:00Z")
    assert result.published_at.isna().all() and not result.point_in_time.any()


def test_vix_caps_and_schema_changes_fail_without_guessing():
    with pytest.raises(ContextLimitError):
        parse_cboe_vix_csv(VIX, received_at="2026-09-05T12:00:00Z", max_rows=1)
    with pytest.raises(ContextError):
        parse_cboe_vix_csv(VIX.replace(b"CLOSE", b"closing"), received_at="2026-09-05T12:00:00Z")


def test_vix_collector_raw_receipt_and_idempotent_replay(tmp_path):
    store = RawStore(tmp_path)
    result = collect_cboe_vix(
        None,
        store,
        snapshot_id="file",
        offline=(VIX, "2026-09-05T12:00:00Z"),
        metrics=("CLOSE",),
    )
    manifest = store.manifest(result.collection_id)
    assert result.complete and store.read_page(result.collection_id, "raw-file") == VIX
    replay = collect_cboe_vix(
        None,
        store,
        snapshot_id="file",
        offline=(VIX, "2026-09-05T12:00:00Z"),
        metrics=["CLOSE"],
    )
    assert replay.complete and store.manifest(result.collection_id) == manifest


@pytest.fixture
def monthly():
    return pd.DataFrame(
        {"Month": ["2026M07", "2026M08"], "Copper": [9500, 9600], "Gold": [None, 3400]}
    )


def test_monthly_units_no_daily_or_release_inference_and_asof(monthly):
    specs = {
        "Copper": CommodityColumn("copper", "USD_per_metric_ton", currency="USD"),
        "Gold": CommodityColumn("gold", "USD_per_troy_ounce", currency="USD"),
    }
    frame = parse_pink_sheet(
        monthly,
        period_column="Month",
        column_specs=specs,
        received_at="2026-09-03T12:00:00Z",
        revision_id="file-v1",
        period_format="YYYYMmm",
    )
    assert frame.frequency.eq("monthly").all() and frame.published_at.isna().all()
    assert frame.loc[frame.entity_id.eq("copper"), "unit"].eq("USD_per_metric_ton").all()
    assert frame.loc[frame.entity_id.eq("gold"), "unit"].eq("USD_per_troy_ounce").all()
    assert pink_sheet_values_asof(frame, "2026-08-31T23:59:59Z").empty
    selected = pink_sheet_values_asof(frame, "2026-09-03T12:00:00Z")
    assert selected.observation_date.eq("2026-08-01").all() and len(selected) == 2
    assert not frame.point_in_time.any()


def test_monthly_publication_is_file_version_time_not_month_end(monthly):
    kwargs = dict(
        period_column="Month",
        column_specs={"Copper": CommodityColumn("copper", "USD/ton")},
        received_at="2026-09-10T12:00:00Z",
        revision_id="file-v1",
        period_format="YYYYMmm",
        availability_basis="publication",
    )
    with pytest.raises(ContextError, match="aware release"):
        parse_pink_sheet(monthly, published_at="2026-09-03", **kwargs)
    frame = parse_pink_sheet(monthly, published_at="2026-09-03T10:00:00Z", **kwargs)
    assert pink_sheet_values_asof(frame, "2026-09-03T09:59:59Z").empty
    assert len(pink_sheet_values_asof(frame, "2026-09-03T10:00:00Z")) == 1
    assert not frame.point_in_time.any()


def _workbook():
    excel = pytest.importorskip("openpyxl")
    book = excel.Workbook()
    sheet = book.active
    sheet.title = "DeclaredMonthly"
    sheet.append([None, "Copper", "Gold", None])
    sheet.append([None, "USD/ton", "USD/oz", None])
    sheet.append(["2026M07", 9500, 3300, None])
    sheet.append(["2026M08", 9600, 3400, None])
    body = io.BytesIO()
    book.save(body)
    book.close()
    return body.getvalue()


def test_xlsx_explicit_sheet_column_index_units_row_and_zip_limits():
    body = _workbook()
    options = dict(
        sheet_name="DeclaredMonthly",
        header_row=0,
        data_start_row=2,
        period_column=0,
        column_specs={"Copper": {"entity_id": "copper", "unit": "USD/ton"}},
        period_format="YYYYMmm",
        received_at="2026-09-03T12:00:00Z",
    )
    frame = parse_pink_sheet_xlsx(body, **options)
    assert len(frame) == 2 and list(frame.value) == [9500, 9600]
    with pytest.raises(ContextLimitError, match="expanded ZIP"):
        parse_pink_sheet_xlsx(body, max_expanded_bytes=10, **options)
    with pytest.raises(ContextError, match="sheet_name"):
        parse_pink_sheet_xlsx(body, **{**options, "sheet_name": "GuessedSheet"})
    with pytest.raises(ContextLimitError):
        parse_pink_sheet_xlsx(body, max_rows=1, **options)


def test_pink_sheet_xlsx_collector_replay_preserves_file_version(tmp_path):
    body = _workbook()
    options = dict(
        sheet_name="DeclaredMonthly",
        header_row=0,
        data_start_row=2,
        period_column=0,
        column_specs={"Copper": {"entity_id": "copper", "unit": "USD/ton"}},
        period_format="YYYYMmm",
    )
    store = RawStore(tmp_path)
    first = collect_pink_sheet(
        None, store, snapshot_id="file-v1", offline=(body, "2026-09-03T12:00:00Z"), **options
    )
    manifest = store.manifest(first.collection_id)
    replay = collect_pink_sheet(
        None, store, snapshot_id="file-v1", offline=(body, "2026-09-03T12:00:00Z"), **options
    )
    assert replay.complete and store.manifest(first.collection_id) == manifest
    assert store.read_page(first.collection_id, "raw-file") == body


@pytest.mark.parametrize("kwargs", [{"max_rows": 0}, {"page_size": 5001}, {"max_bytes": True}])
def test_context_limits_reject_invalid_budgets(kwargs):
    with pytest.raises(ContextError):
        ContextLimits(**kwargs)
