"""Offline Lot 0 demonstration: synthetic vintages, raw replay and LSE metadata."""

from tempfile import TemporaryDirectory

import pandas as pd

from pfe import (
    RawStore,
    SymbolRequest,
    default_source_registry,
    normalize_observations,
    observations_asof,
    qualify_lse_snapshot,
)


def main():
    # Synthetic economic periods and publications. The late revision stays invisible
    # to January decisions; different periods available simultaneously remain separate.
    rows = [
        ("2025-11-01", "2026-01-02T13:30:00Z", 3.0, "v1"),
        ("2025-12-01", "2026-01-02T13:30:00Z", 3.1, "v1"),
        ("2025-11-01", "2026-02-02T13:30:00Z", 2.9, "v2"),
    ]
    observations = normalize_observations(
        pd.DataFrame(
            [
                {
                    "entity_id": "factor:synthetic-cpi",
                    "metric": "cpi_yoy",
                    "observation_date": period,
                    "published_at": published,
                    "available_at": published,
                    "value": value,
                    "unit": "percentage_point",
                    "source_id": "local",
                    "revision_id": revision,
                    "schema_version": "1",
                    "availability_basis": "publication",
                }
                for period, published, value, revision in rows
            ]
        )
    )
    for decision in ("2026-01-02T13:29:59Z", "2026-01-02T13:30:00Z", "2026-02-02T13:30:00Z"):
        known = observations_asof(observations, decision)
        print(f"{decision}: {list(zip(known.observation_date, known.value, strict=True))}")

    with TemporaryDirectory(prefix="pfe-lot0-") as directory:
        store = RawStore(directory)
        request = {"start": "2025-11-01", "end": "2026-01-01", "snapshot_id": "synthetic-1"}
        source = default_source_registry().get("local").to_dict()
        collection = store.begin("local", "synthetic", request, source_metadata=source)
        payload = {"synthetic": True, "values": [3.0, 3.1]}
        page = store.write_page(collection, "page-0", payload, row_count=2, complete=True)
        assert store.begin("local", "synthetic", request, source_metadata=source) == collection
        assert store.write_page(collection, "page-0", payload, row_count=2, complete=True) == page
        print(f"Raw replay: pages={store.manifest(collection)['page_count']}, rows=2")

    # Authority/completeness are assertions for this synthetic fixture ONLY. An
    # arbitrary snapshot without these flags cannot prove an exact code absent.
    snapshot = {
        "provider": "LSE",
        "audited_at": "2026-10-11T12:00:00Z",
        "catalog": [{"symbol": "QQQ", "dataset": "etf", "country": "US"}],
        "reference": [{"dataset": "dividends"}, {"dataset": "stock_splits"}],
        "catalog_complete": True,
        "catalog_authoritative": True,
        "reference_complete": True,
        "reference_authoritative": True,
    }
    report = qualify_lse_snapshot(snapshot, symbols=(SymbolRequest("QQQ", "etf"), "MISSING"))
    print(f"Requirements: {len(report['requirements'])}; entitlement remains unknown")
    print([(row["symbol"], row["status"]) for row in report["requested_symbols"]])


if __name__ == "__main__":
    main()
