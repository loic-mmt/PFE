"""Small synthetic provider contracts: no real collection or downloaded fixtures."""

import json

import numpy as np
import pandas as pd
import pytest

from pfe import (
    InternationalLimits,
    InternationalSeries,
    RawStore,
    collect_international,
    compute_international_features,
    default_source_registry,
    merge_international_observations,
    observations_asof,
)
from pfe.data.eia import ContextError, ContextLimitError
from pfe.data.http import HTTPResponse
from pfe.data.international_sources import (
    _number,
    parse_boj,
    parse_eurostat,
    parse_gdelt_events,
    parse_mops_revenue,
    parse_opendart_accounts,
    parse_tsmc_revenue,
    parse_twse_daily,
    roc_date,
)

RECEIPT = "2026-10-12T12:00:00Z"


def test_tsmc_missing_marker_preserves_null_revision_for_completed_month(tmp_path):
    template = (
        "<h2>2026 Monthly Revenue</h2><p>In Millions of New Taiwan Dollars</p><table>"
        "<tr><td>Jan.</td><td>{}</td><td>-</td></tr>"
        "<tr><td>Feb.</td><td></td><td></td></tr>"
        "<tr><td>Dec.</td><td>-</td><td>-</td></tr></table>"
    )
    store = RawStore(tmp_path)
    config = {"year": 2026, "entity_id": "tsmc"}
    first = collect_international(
        "tsmc",
        config,
        store,
        snapshot_id="first",
        offline=(template.format("100").encode(), "2026-02-15T00:00:00Z"),
    )
    withdrawn = collect_international(
        "tsmc",
        config,
        store,
        snapshot_id="withdrawn",
        offline=(template.format("-").encode(), "2026-03-01T00:00:00Z"),
    )
    assert len(withdrawn.observations) == 2
    assert withdrawn.observations.observation_date.eq("2026-01-01").all()
    assert withdrawn.observations.value.isna().all()
    merged = merge_international_observations(first, withdrawn)
    features = compute_international_features(
        ["2026-02-16T00:00:00Z", "2026-03-01T00:00:00Z"],
        merged,
        series=[InternationalSeries("tsmc", "tsmc", "revenue", "million_TWD", "tsmc", "monthly")],
    )
    assert features.iloc[0]["international.tsmc.level"] == 100
    assert np.isnan(features.iloc[1]["international.tsmc.level"])
    assert features.iloc[1]["international.tsmc.mask.value"] == 0


@pytest.mark.parametrize(
    "received,expected_rows",
    [("2026-01-31T15:59:59Z", 0), ("2026-01-31T16:00:00Z", 2)],
)
def test_tsmc_missing_month_only_observed_after_taiwan_month_ends(received, expected_rows):
    body = (
        b"<h2>2026 Monthly Revenue</h2><p>In Millions of New Taiwan Dollars</p><table>"
        b"<tr><td>Jan.</td><td>-</td><td>-</td></tr></table>"
    )
    frame = parse_tsmc_revenue(body, received_at=received, year=2026, entity_id="tsmc")
    assert len(frame) == expected_rows


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode()


def samples():
    tsmc = (
        b"<h2>2026 Monthly Revenue</h2><p>In Millions of New Taiwan Dollars</p><table>"
        b"<tr><th>Month</th><th>Net Revenue</th><th>YoY Change</th></tr>"
        b"<tr><td>Jan.</td><td>1,200</td><td>20.0%</td></tr>"
        b"<tr><td>Feb.</td><td></td><td></td></tr>"
        b"<tr><td>Total</td><td>1200</td><td>20%</td></tr></table>"
    )
    mops = [
        {
            "公司代號": "2330",
            "資料年月": "11509",
            "出表日期": "1151012",
            "營業收入-當月營收": "1,000",
            "營業收入-去年同月增減(%)": "10.5",
        }
    ]
    twse = {
        "stat": "OK",
        "date": "20260901",
        "title": "115年09月 2330 台積電",
        "fields": ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價"],
        "data": [["115/09/01", "1,000", "100,000", "100", "110", "95", "105"]],
    }
    account = {
        "account_nm": "Revenue",
        "fs_div": "CFS",
        "sj_div": "IS",
        "amount_field": "thstrm_add_amount",
        "metric": "revenue_ytd",
        "period_start": "2026-01-01",
        "period_end": "2026-06-30",
        "period_kind": "YTD",
        "currency": "KRW",
    }
    dart = {
        "status": "000",
        "list": [
            {
                "account_nm": "Revenue",
                "fs_div": "CFS",
                "sj_div": "IS",
                "bsns_year": "2026",
                "reprt_code": "11012",
                "rcept_no": "20260814000001",
                "thstrm_amount": "150",
                "thstrm_add_amount": "1,500",
                "thstrm_dt": "2026.01.01 ~ 2026.06.30",
                "currency": "KRW",
                "ord": "1",
            }
        ],
    }
    dimensions = {"freq": "M", "unit": "I15", "geo": "DE"}
    euro = {
        "class": "dataset",
        "id": ["freq", "unit", "geo", "time"],
        "size": [1, 1, 1, 2],
        "dimension": {k: {"category": {"index": {v: 0}}} for k, v in dimensions.items()},
        "value": {"0": 110},
        "status": {"1": "c"},
        "updated": "2026-09-01T10:00:00Z",
    }
    euro["dimension"]["time"] = {"category": {"index": {"2026-07": 0, "2026-08": 1}}}
    boj = {
        "STATUS": 200,
        "NEXTPOSITION": None,
        "RESULTSET": [
            {
                "SERIES_CODE": "EXPLICIT_CODE",
                "FREQUENCY": "QUARTERLY",
                "UNIT": "percentage points",
                "LAST_UPDATE": 20261001,
                "VALUES": {"SURVEY_DATES": [202601, 202602], "VALUES": [10, None]},
            }
        ],
    }
    estat = {
        "GET_STATS_DATA": {
            "RESULT": {"STATUS": 0},
            "STATISTICAL_DATA": {
                "TABLE_INF": {"@id": "0001"},
                "RESULT_INF": {},
                "DATA_INF": {
                    "VALUE": [
                        {
                            "@cat01": "100",
                            "@area": "00000",
                            "@time": "2026000800",
                            "@unit": "Index",
                            "$": "99.5",
                        }
                    ]
                },
            },
        }
    }
    nbs = (
        b"<table><tr><th>Month</th><th>Manufacturing PMI</th></tr>"
        b"<tr><td>2026-May</td><td>50.0</td></tr></table>"
    )
    gdelt = {
        "articles": [
            {
                "url": "https://example.org/a",
                "title": "Example",
                "seendate": "20261011T120000Z",
                "language": "Chinese",
                "sourcecountry": "China",
            }
        ]
    }
    event = [""] * 61
    for idx, value in {
        0: "123",
        1: "20261011",
        6: "ENTITY A",
        7: "CHN",
        16: "ENTITY B",
        17: "USA",
        27: "0251",
        29: "02",
        30: "1",
        31: "-2",
        32: "1",
        33: "1",
        34: "1",
        35: "-1.2",
        53: "CH",
        59: "20261011120000",
        60: "https://example.org/a",
    }.items():
        event[idx] = value
    return {
        "tsmc": ({"year": 2026, "entity_id": "tsmc"}, tsmc),
        "mops": ({"companies": {"2330": "tsmc"}}, encoded(mops)),
        "twse": ({"month": "2026-09", "stock_no": "2330", "entity_id": "tsmc"}, encoded(twse)),
        "opendart": (
            {
                "corp_code": "00126380",
                "year": 2026,
                "report_code": "11012",
                "entity_id": "samsung",
                "accounts": [account],
            },
            encoded(dart),
        ),
        "eurostat": (
            {
                "dataset": "test",
                "dimensions": dimensions,
                "start": "2026-07",
                "end": "2026-08",
                "entity_id": "de_industry",
                "metric": "production",
            },
            encoded(euro),
        ),
        "boj": (
            {
                "db": "CO",
                "code": "EXPLICIT_CODE",
                "start": "202601",
                "end": "202602",
                "entity_id": "jp_tankan",
                "metric": "level",
                "unit": "percentage points",
                "frequency": "quarterly",
            },
            encoded(boj),
        ),
        "estat": (
            {
                "table_id": "0001",
                "dimensions": {"cat01": "100", "area": "00000"},
                "start_code": "2026000800",
                "end_code": "2026000800",
                "period_map": {"2026000800": "2026-08"},
                "entity_id": "jp_ip",
                "metric": "production",
                "unit": "Index",
                "frequency": "monthly",
            },
            encoded(estat),
        ),
        "nbs": (
            {
                "url": "https://www.stats.gov.cn/english/PressRelease/example.html",
                "entity_id": "cn_pmi",
                "table_index": 0,
                "columns": {"manufacturing_pmi": {"index": 1, "unit": "percentage_points"}},
                "period_map": {"2026-May": "2026-05"},
                "header_evidence": "Manufacturing PMI",
            },
            nbs,
        ),
        "usgs": (
            {
                "url": "https://www.sciencebase.gov/catalog/file/get/selected",
                "material_id": "copper",
                "countries": {"Chile": "cl"},
                "country_column": "Country",
                "period_columns": {"2025": "2025e"},
                "unit": "thousand_tonnes",
                "value_suffixes": ["e"],
            },
            b'Country,2025e\nChile,"5,300e"\nOther,100\n',
        ),
        "gdelt": (
            {
                "query": '"semiconductors"',
                "start": "2026-10-11T00:00:00Z",
                "end": "2026-10-12T00:00:00Z",
                "entity_id": "semiconductor_news",
                "max_records": 10,
            },
            encoded(gdelt),
        ),
        "gdelt_events": (
            {"entity_id": "cn_events", "country_codes": ["CHN"], "root_codes": ["02"]},
            "\t".join(event).encode(),
        ),
    }


@pytest.mark.parametrize("source", tuple(samples()))
def test_every_source_capture_replay_and_contract(source, tmp_path):
    config, body = samples()[source]
    store = RawStore(tmp_path, max_page_bytes=4_194_304)
    result = collect_international(
        source, config, store, snapshot_id="fixture", offline=(body, RECEIPT)
    )
    replay = collect_international(
        source, config, store, snapshot_id="fixture", offline=(body, RECEIPT)
    )
    pd.testing.assert_frame_equal(result.observations, replay.observations)
    assert result.collection_id == replay.collection_id
    assert result.observations.available_at.eq(pd.Timestamp(RECEIPT)).all()
    assert result.observations.published_at.isna().all()
    assert not result.observations.point_in_time.any()
    assert observations_asof(result.observations, "2026-10-12T11:59:59Z").empty
    assert result.observations.evidence_quote.notna().all()
    assert default_source_registry().get(source).availability == "receipt"
    assert result.complete == (source != "gdelt_events")
    assert store.manifest(result.collection_id)["complete"] == result.complete


def test_tsmc_units_future_blank_total_and_explicit_release():
    config, body = samples()["tsmc"]
    frame = parse_tsmc_revenue(
        body, received_at=RECEIPT, publication_times={"2026-01": "2026-02-10T05:30:00Z"}, **config
    )
    assert list(frame.value) == [1200.0, 20.0]
    assert frame.unit.tolist() == ["million_TWD", "percentage_points"]
    assert frame.observation_date.eq("2026-01-01").all()
    assert frame.period_end.eq("2026-01-31").all()
    assert frame.available_at.eq(pd.Timestamp(RECEIPT)).all()
    with pytest.raises(ContextError, match="year/title"):
        parse_tsmc_revenue(body, received_at=RECEIPT, year=2025, entity_id="tsmc")


def test_roc_and_twse_shares_close_calendar():
    assert roc_date("115/09/01") == "2026-09-01"
    config, body = samples()["twse"]
    frame = parse_twse_daily(
        body,
        received_at=RECEIPT,
        close_schedule={"2026-09-01": "2026-09-01T13:30:00+08:00"},
        **config,
    )
    assert frame.loc[frame.metric.eq("volume"), "value"].iloc[0] == 1000
    assert frame.observed_at.iloc[0] == pd.Timestamp("2026-09-01T05:30:00Z")
    assert frame.price_adjustment.eq("unknown").all()
    with pytest.raises(ContextError, match="close schedule"):
        parse_twse_daily(body, received_at=RECEIPT, close_schedule={}, **config)
    wrong = json.loads(body)
    wrong["title"] = "2317 Other"
    with pytest.raises(ContextError, match="identity"):
        parse_twse_daily(encoded(wrong), received_at=RECEIPT, **config)


def test_mops_csv_chinese_units_selection_and_no_table_date_release():
    body = (
        "公司代號,資料年月,出表日期,營業收入-當月營收,營業收入-去年同月增減(%)\n"
        "2330,11509,1151012,1000,-5.2\n2317,11509,1151012,50,10\n"
    ).encode()
    frame = parse_mops_revenue(body, received_at=RECEIPT, companies={"2330": "tsmc"})
    assert frame.value.tolist() == [1000, -5.2]
    assert frame.table_date.eq("1151012").all() and frame.published_at.isna().all()


def test_dart_ytd_and_unknown_account_are_never_inferred():
    config, body = samples()["opendart"]
    frame = parse_opendart_accounts(body, received_at=RECEIPT, **config)
    assert frame.value.iloc[0] == 1500 and frame.period_kind.iloc[0] == "YTD"
    assert frame.frequency.iloc[0] == "fiscal"
    assert frame.period_end.iloc[0] == "2026-06-30" and frame.published_at.isna().all()
    wrong = json.loads(body)
    wrong["list"][0]["currency"] = "USD"
    with pytest.raises(ContextError, match="currency"):
        parse_opendart_accounts(encoded(wrong), received_at=RECEIPT, **config)
    config["accounts"][0]["account_nm"] = "Absent"
    assert parse_opendart_accounts(body, received_at=RECEIPT, **config).empty


@pytest.mark.parametrize("noncalendar", [False, True])
def test_dart_annual_comparisons_require_calendar_aligned_periods(noncalendar):
    periods = [
        (2025, "2025-01-01", "2025-12-31", 100),
        (
            2026,
            "2025-04-01" if noncalendar else "2026-01-01",
            "2026-03-31" if noncalendar else "2026-12-31",
            150,
        ),
    ]
    frames = []
    for year, start, end, value in periods:
        config, body = samples()["opendart"]
        config.update(year=year, report_code="11011")
        config["accounts"][0].update(
            amount_field="thstrm_amount",
            metric="revenue",
            period_start=start,
            period_end=end,
            period_kind="annual",
        )
        payload = json.loads(body)
        payload["list"][0].update(
            bsns_year=str(year),
            reprt_code="11011",
            thstrm_amount=str(value),
            thstrm_dt=start + " ~ " + end,
        )
        frames.append(
            parse_opendart_accounts(encoded(payload), received_at="2027-02-01T00:00:00Z", **config)
        )
    frequency = "fiscal" if noncalendar else "annual"
    assert frames[0].frequency.iloc[0] == "annual"
    assert frames[1].frequency.iloc[0] == frequency
    observations = merge_international_observations(*frames)
    features = compute_international_features(
        ["2027-02-01T00:00:00Z"],
        observations,
        series=[InternationalSeries("dart", "samsung", "revenue", "KRW", "opendart", frequency)],
    )
    assert features.iloc[0]["international.dart.level"] == 150
    if noncalendar:
        assert np.isnan(features.iloc[0]["international.dart.yoy"])
        assert np.isnan(features.iloc[0]["international.dart.change"])
    else:
        assert features.iloc[0]["international.dart.yoy"] == 0.5


def test_eurostat_missing_cell_status_and_multiseries_rejection():
    config, body = samples()["eurostat"]
    frame = parse_eurostat(body, received_at=RECEIPT, **config)
    assert frame.value.tolist() == [110.0, None] and frame.provider_status.iloc[1] == "c"
    assert frame.published_at.isna().all()
    bad = json.loads(body)
    bad["dimension"]["geo"]["category"]["index"] = {"FR": 0}
    with pytest.raises(ContextError, match="dimensional"):
        parse_eurostat(encoded(bad), received_at=RECEIPT, **config)


@pytest.mark.parametrize("source,field", [("boj", "NEXTPOSITION"), ("estat", "NEXT_KEY")])
def test_continuation_is_partial_not_silent_complete(source, field, tmp_path):
    config, body = samples()[source]
    payload = json.loads(body)
    if source == "boj":
        payload[field] = 2
    else:
        payload["GET_STATS_DATA"]["STATISTICAL_DATA"]["RESULT_INF"][field] = 2
    result = collect_international(
        source,
        config,
        RawStore(tmp_path),
        snapshot_id="partial",
        offline=(encoded(payload), RECEIPT),
    )
    assert not result.complete and "continuation" in result.issues[0]


def test_gdelt_cap_and_event_codes_not_stock_predictions(tmp_path):
    config, body = samples()["gdelt"]
    config["max_records"] = 1
    result = collect_international(
        "gdelt", config, RawStore(tmp_path), snapshot_id="cap", offline=(body, RECEIPT)
    )
    assert not result.complete
    point = result.observations.iloc[0]
    assert point.value["first_known_publication_time"] is None and point.detected_at != RECEIPT
    config, body = samples()["gdelt_events"]
    events = parse_gdelt_events(body, received_at=RECEIPT, **config)
    assert events.value.iloc[0]["event_code"] == "0251"
    assert events.value.iloc[0]["is_model_output"] is True


def test_live_injected_key_redaction_and_keyless_cache(tmp_path):
    config, body = samples()["opendart"]
    payload = json.loads(body)
    payload["request"] = {"crtfc_key": "SECRET"}
    calls = []

    class Transport:
        def get(self, url, *, params, max_bytes):
            calls.append(params)
            assert params["crtfc_key"] == "SECRET"
            return HTTPResponse(encoded(payload), RECEIPT)

    store = RawStore(tmp_path)
    result = collect_international(
        "opendart", config, store, snapshot_id="online", transport=Transport(), credential="SECRET"
    )
    again = collect_international("opendart", config, store, snapshot_id="online")
    assert len(calls) == 1 and result.collection_id == again.collection_id
    for file in tmp_path.rglob("*"):
        if file.is_file():
            assert b"SECRET" not in file.read_bytes() and b"crtfc_key" not in file.read_bytes()


@pytest.mark.parametrize("value", [True, "12,5", "$1", "1,00", float("inf"), "12x"])
def test_no_numeric_locale_guess(value):
    with pytest.raises(ContextError):
        _number(value)


def test_budget_failure_before_transfer_and_archive_bound(tmp_path):
    config, body = samples()["tsmc"]
    with pytest.raises(ContextLimitError, match="response"):
        collect_international(
            "tsmc",
            config,
            RawStore(tmp_path),
            snapshot_id="small",
            offline=(body, RECEIPT),
            limits=InternationalLimits(max_response_bytes=20),
        )
    with pytest.raises(ContextLimitError, match="archive"):
        collect_international(
            "tsmc",
            config,
            RawStore(tmp_path),
            snapshot_id="archive",
            offline=(body, RECEIPT),
            limits=InternationalLimits(max_response_bytes=512, max_archive_bytes=600),
        )


def test_merge_preserves_receipt_and_revisions(tmp_path):
    config, body = samples()["tsmc"]
    one = collect_international(
        "tsmc", config, RawStore(tmp_path), snapshot_id="a", offline=(body, RECEIPT)
    )
    late = one.observations.copy()
    late["available_at"] = pd.Timestamp("2026-10-13T12:00:00Z")
    merged = merge_international_observations(one, late)
    assert len(merged) == 2 and merged.available_at.eq(pd.Timestamp(RECEIPT)).all()
    late.loc[0, "value"] = 99
    with pytest.raises(ContextError, match="conflicting"):
        merge_international_observations(one, late)


def test_boj_wrong_unit_or_series_rejected():
    config, body = samples()["boj"]
    config["unit"] = "JPY"
    with pytest.raises(ContextError, match="unit"):
        parse_boj(body, received_at=RECEIPT, **config)


@pytest.mark.parametrize("source", ("nbs", "usgs"))
@pytest.mark.parametrize(
    "url",
    ("http://www.stats.gov.cn/a", "https://example.org/a", "https://www.stats.gov.cn/a?secret=x"),
)
def test_unauthorized_hosts_or_query_urls_rejected_before_transport(source, url, tmp_path):
    config, body = samples()[source]
    config["url"] = url
    with pytest.raises(ContextError, match="official HTTPS"):
        collect_international(
            source, config, RawStore(tmp_path), snapshot_id="bad", offline=(body, RECEIPT)
        )


def test_dart_period_end_never_confuses_filing_date():
    config, body = samples()["opendart"]
    config["accounts"][0]["period_end"] = "2026-08-14"
    with pytest.raises(ContextError, match="period end"):
        parse_opendart_accounts(body, received_at=RECEIPT, **config)


def test_usgs_one_csv_filename_bound_before_transfer(tmp_path):
    config, body = samples()["usgs"]
    config["file_name"] = "world.zip"
    with pytest.raises(ContextError, match="plain CSV"):
        collect_international(
            "usgs", config, RawStore(tmp_path), snapshot_id="bad", offline=(body, RECEIPT)
        )
    config["file_name"] = "world.csv"
    result = collect_international(
        "usgs", config, RawStore(tmp_path), snapshot_id="csv", offline=(body, RECEIPT)
    )
    assert result.metadata["request"]["params"] == {"name": "world.csv"}
    assert result.observations.value.iloc[0] == 5300
    assert result.observations.provider_status.iloc[0] == "e"


def test_parser_config_must_be_complete_before_network(tmp_path):
    config = {"year": 2026}
    with pytest.raises(ContextError, match="parser configuration"):
        collect_international("tsmc", config, RawStore(tmp_path), snapshot_id="bad")


def test_missing_credentials_require_no_bulk_corporation_codes(tmp_path):
    config, _ = samples()["opendart"]
    with pytest.raises(ContextError, match="credential required"):
        collect_international("opendart", config, RawStore(tmp_path), snapshot_id="key")


def test_no_gdelt_zip_or_model_download(tmp_path):
    config, _ = samples()["gdelt_events"]
    with pytest.raises(ContextError, match="no ZIP"):
        collect_international("gdelt_events", config, RawStore(tmp_path), snapshot_id="zip")


def test_online_request_for_each_supported_endpoint_is_bounded(tmp_path):
    for source, (config, body) in samples().items():
        if source == "gdelt_events":
            continue
        calls = []

        class Transport:
            def __init__(self, response_body, captured_calls):
                self.response_body = response_body
                self.captured_calls = captured_calls

            def get(self, url, *, params, max_bytes):
                self.captured_calls.append((url, params, max_bytes))
                return HTTPResponse(self.response_body, RECEIPT)

        result = collect_international(
            source,
            config,
            RawStore(tmp_path / source),
            snapshot_id="online",
            transport=Transport(body, calls),
            credential="fixture-secret",
        )
        assert result.complete and len(calls) == 1
        assert calls[0][0].startswith("https://") and calls[0][2] == 524_288


def test_source_row_cap_rejects_instead_of_silent_truncation(tmp_path):
    config, body = samples()["twse"]
    with pytest.raises(ContextLimitError, match="max_rows"):
        collect_international(
            "twse",
            config,
            RawStore(tmp_path),
            snapshot_id="cap",
            offline=(body, RECEIPT),
            limits=InternationalLimits(max_rows=1),
        )


def test_resume_after_normalized_write_failure_without_key_or_transfer(tmp_path, monkeypatch):
    config, body = samples()["opendart"]

    class Transport:
        def get(self, url, *, params, max_bytes):
            return HTTPResponse(body, RECEIPT)

    store = RawStore(tmp_path)
    original = store.write_page

    def fail_normalized(collection, page, payload, **kwargs):
        if page == "normalized":
            raise OSError("simulated output failure")
        return original(collection, page, payload, **kwargs)

    monkeypatch.setattr(store, "write_page", fail_normalized)
    with pytest.raises(OSError, match="simulated"):
        collect_international(
            "opendart",
            config,
            store,
            snapshot_id="resume",
            transport=Transport(),
            credential="secret",
        )
    monkeypatch.setattr(store, "write_page", original)
    result = collect_international("opendart", config, store, snapshot_id="resume")
    assert result.complete and result.observations.available_at.eq(pd.Timestamp(RECEIPT)).all()
