"""Opt-in tiny API snapshots and installed Ollama smoke checks; never pulls models."""

import argparse
import json
import os
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path

from pfe import (
    ContextLimits,
    DocumentStore,
    EIAQuery,
    ExtractionLimits,
    Extractor,
    OllamaBackend,
    PipelineContext,
    PipelineLimits,
    collect_eia,
    collect_fred_series,
    collect_international,
)
from pfe.data.http import HTTPTransportError
from pfe.data.qualification import sanitize_metadata
from pfe.data.storage import _json_bytes

KEYS = ("FRED_API_KEY", "EIA_API_KEY", "OPENDART_API_KEY", "ESTAT_APP_ID")


def load_environment(path):
    """Read literal values for the four selected keys; no shell or interpolation."""
    with Path(path).open("rb") as stream:
        body = stream.read(65_537)
    if len(body) > 65_536:
        raise ValueError("Environment file exceeds byte budget")
    for line in body.decode().splitlines():
        line = line.strip().removeprefix("export ")
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or name not in KEYS:
            continue
        parts = shlex.split(value, comments=True)
        if len(parts) > 1:
            raise ValueError("Selected environment value must be a literal scalar")
        if parts and parts[0]:
            os.environ.setdefault(name, parts[0])


def _fred(context, snapshot):
    now = datetime.now(UTC).date().isoformat()
    args = dict(
        series_id="DGS2",
        start="2025-01-02",
        end="2025-01-03",
        snapshot_id=snapshot,
        realtime_start=now,
        realtime_end=now,
        api_key=os.environ["FRED_API_KEY"],
        max_rows=30,
        max_bytes=131_072,
        max_requests=6,
        page_size=5,
    )
    collection, data, summary = collect_fred_series(
        context.transport("fred"), context.store, **args
    )
    replay = collect_fred_series(None, context.store, **{**args, "api_key": None})
    if not data.equals(replay[1]):
        raise ValueError("FRED replay differs")
    return dict(
        status="passed",
        rows=len(data),
        replay=True,
        complete=summary["pagination_complete"],
        collection_id=collection,
    )


def _eia(context, snapshot):
    options = dict(
        query=EIAQuery(
            "petroleum/pri/spt",
            "WTI",
            facets={"series": "RWTC"},
            start="2025-01-02",
            end="2025-01-03",
            unit="Dollars per Barrel",
        ),
        snapshot_id=snapshot,
        limits=ContextLimits(max_rows=3, max_requests=2, max_bytes=65_536, page_size=3),
    )
    result = collect_eia(
        context.transport("eia"), context.store, api_key=os.environ["EIA_API_KEY"], **options
    )
    replay = collect_eia(None, context.store, **options)
    if not result.observations.equals(replay.observations):
        raise ValueError("EIA replay differs")
    return dict(
        status="passed",
        rows=len(result.observations),
        replay=True,
        complete=result.complete,
        collection_id=result.collection_id,
    )


def _dart(context, snapshot):
    # One English major-accounts report, one explicitly selected annual revenue.
    options = dict(
        snapshot_id=snapshot,
        config=dict(
            corp_code="00126380",
            year=2024,
            report_code="11011",
            entity_id="samsung",
            accounts=[
                dict(
                    account_nm="Sales",
                    fs_div="CFS",
                    sj_div="IS",
                    amount_field="thstrm_amount",
                    metric="revenue",
                    period_start="2024-01-01",
                    period_end="2024-12-31",
                    period_kind="annual",
                    currency="KRW",
                )
            ],
        ),
    )
    result = collect_international(
        "opendart",
        store=context.store,
        credential=os.environ["OPENDART_API_KEY"],
        transport=context.transport("opendart", hosts={"engopendart.fss.or.kr"}),
        **options,
    )
    replay = collect_international("opendart", store=context.store, **options)
    if not result.observations.equals(replay.observations):
        raise ValueError("OpenDART replay differs")
    return dict(
        status="passed" if len(result.observations) else "empty_selection",
        rows=len(result.observations),
        replay=True,
        complete=result.complete,
        collection_id=result.collection_id,
    )


def _estat(context, snapshot):
    """One catalogue result plus one cell. Explicitly a transport/schema smoke."""
    key = os.environ["ESTAT_APP_ID"]
    transport = context.transport("estat", hosts={"api.e-stat.go.jp"})
    collection = context.store.begin("estat", "integration_smoke", {"snapshot_id": snapshot})

    def get(page, endpoint, params):
        pages = {p["page_id"] for p in context.store.recover(collection)["pages"]}
        if page in pages:
            return json.loads(context.store.read_page(collection, page))
        response = transport.get(
            "https://api.e-stat.go.jp/rest/3.0/app/json/" + endpoint,
            params={**params, "appId": key},
            max_bytes=65_536,
        )
        payload = sanitize_metadata(json.loads(response.body), secrets=(key,))
        context.store.write_page(
            collection, page, payload, metadata={"received_at": response.received_at}
        )
        return payload

    listing = get("catalogue", "getStatsList", dict(searchWord="消費者物価指数", limit=1))
    root = listing.get("GET_STATS_LIST", {})
    if str(root.get("RESULT", {}).get("STATUS")) != "0":
        raise ValueError("e-Stat unsuccessful catalogue status")
    tables = root.get("DATALIST_INF", {}).get("TABLE_INF", [])
    tables = [tables] if isinstance(tables, dict) else tables
    if len(tables) != 1 or not re.fullmatch(r"[0-9]+", str(tables[0].get("@id"))):
        raise ValueError("e-Stat expected one table")
    table = tables[0]["@id"]
    data = get("cell", "getStatsData", dict(statsDataId=table, limit=1, metaGetFlg="N"))
    root = data.get("GET_STATS_DATA", {})
    if str(root.get("RESULT", {}).get("STATUS")) != "0":
        raise ValueError("e-Stat unsuccessful cell status")
    cells = root.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE", [])
    cells = [cells] if isinstance(cells, dict) else cells
    if len(cells) != 1 or not isinstance(cells[0], dict) or "$" not in cells[0]:
        raise ValueError("e-Stat expected one statistical cell")
    return dict(
        status="passed",
        rows=1,
        scope="authentication_and_one_cell_schema",
        table_id=table,
        collection_id=collection,
        economic_mapping_qualified=False,
    )


def _ollama(context, model):
    # Probe tags only; use the existing backend to reject remote/cloud aliases.
    probe = OllamaBackend(
        model, model_revision="0" * 64, options={"num_ctx": 8192, "num_predict": 768}, timeout=60
    )
    tags, _ = probe._request("/api/tags")
    installed = [row for row in tags.get("models", []) if row.get("name") == model]
    if len(installed) != 1:
        raise ValueError("Selected local model is not installed")
    backend = OllamaBackend(
        model,
        model_revision=installed[0]["digest"],
        options={"num_ctx": 8192, "num_predict": 768},
        timeout=60,
    )
    text = (
        b"Synthetic fixture. Acme Corp buys silicon wafers from Beta Inc. "
        b"Japan represented 30% of Acme Corp revenue in fiscal 2024. "
        b"On 2025-02-03, Acme Corp reported fiscal 2024 revenue of USD 100 million. "
        b"On 2025-02-03, Acme Corp announced a new chip product."
    )
    document = DocumentStore(context.root / "documents").import_bytes(
        text,
        logical_id="tiny-integration-fixture",
        source_id="synthetic",
        rights="authored_fixture",
    )
    extractor = Extractor(
        context.root / "extraction",
        backend=backend,
        limits=ExtractionLimits(
            chunk_chars=1024,
            overlap_chars=0,
            max_chunks=1,
            max_requests=4,
            max_response_bytes=16_384,
            max_total_bytes=1_048_576,
        ),
    )
    result = extractor.extract(document)
    replay = extractor.extract(document)
    if result.to_dict() != replay.to_dict():
        raise ValueError("Ollama replay differs")
    context.check_disk()
    return dict(
        status="passed" if result.complete else "incomplete",
        model=model,
        model_revision=backend.model_revision,
        complete=result.complete,
        accepted=len(result.accepted),
        rejected=len(result.rejected),
        tasks=["A", "B", "C", "D"],
        replay=True,
        quality_qualified=False,
        scope="synthetic_integration_only",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--services",
        nargs="+",
        choices=["fred", "eia", "opendart", "estat", "ollama"],
        default=["fred", "eia", "opendart", "estat", "ollama"],
    )
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    args = parser.parse_args(argv)
    if args.env_file:
        load_environment(args.env_file)
    context = PipelineContext(
        args.output,
        PipelineLimits(
            max_requests=12,
            max_response_bytes=65_536,
            max_download_bytes=524_288,
            max_bytes=2_097_152,
        ),
        True,
        None,
    )
    checks = dict(fred=_fred, eia=_eia, opendart=_dart, estat=_estat, ollama=_ollama)
    results = {}
    for service in dict.fromkeys(args.services):
        if service != "ollama" and not os.environ.get(
            KEYS[["fred", "eia", "opendart", "estat"].index(service)]
        ):
            results[service] = dict(status="missing_credential")
            continue
        try:
            results[service] = checks[service](
                context, args.model if service == "ollama" else args.snapshot_id
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            # Exceptions may contain remote input: record type only, never URLs/keys/body.
            results[service] = dict(status="failed", error_type=type(exc).__name__)
            if isinstance(exc, HTTPTransportError):
                results[service]["transport_error"] = str(exc)
            elif service == "ollama" and type(exc) is ValueError:
                # OllamaBackend emits local, fixed diagnostic strings.
                results[service]["local_error"] = str(exc)
        print(json.dumps({service: results[service]}), flush=True)
    report = dict(
        services=results,
        requests=context.requests,
        download_bytes=context.download_bytes,
        archive_bytes=sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
        bulk_download=False,
        model_download=False,
    )
    body = _json_bytes(report)
    context.check_disk(len(body))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_bytes(body)
    return 0 if all(r["status"] == "passed" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
