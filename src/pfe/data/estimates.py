"""Optional licensed consensus/detail exports and caller-owned authenticated clients.

No proprietary API names, entitlements, historical PIT rights or field mappings
are guessed. FactSet/I/B/E/S files require explicit export column/value mappings.
"""

import csv
import hashlib
import io
import json
from dataclasses import dataclass

import pandas as pd

from pfe.data.captures import CaptureLimits, capture_observations
from pfe.data.contracts import _instant, normalize_observations
from pfe.data.eia import _observations, _period
from pfe.data.international_sources import _json, _number, _point
from pfe.data.sources import SourceRecord
from pfe.data.storage import _reject_secrets

ESTIMATE_CONTEXT = (
    "metric",
    "period_start",
    "period_end",
    "period_kind",
    "currency",
    "unit_scale",
    "accounting_basis",
    "eps_basis",
    "segment",
)
ESTIMATE_REQUIRED = (
    *ESTIMATE_CONTEXT,
    "entity_id",
    "statement_type",
    "statistic",
    "analyst_id",
    "snapshot_at",
    "published_at",
    "value",
    "active",
)


@dataclass(frozen=True)
class EstimateLicense:
    provider: str
    dataset: str
    license_reference: str
    qualified_historical_pit: bool = False
    qualification_reference: str | None = None

    def __post_init__(self):
        if self.provider not in ("factset", "ibes", "licensed_export"):
            raise ValueError("Choose a declared licensed estimates provider")
        if any(
            not isinstance(v, str) or not v.strip() for v in (self.dataset, self.license_reference)
        ):
            raise ValueError("Dataset and non-secret license reference are required")
        if (
            type(self.qualified_historical_pit) is not bool
            or self.qualified_historical_pit
            and not self.qualification_reference
        ):
            raise ValueError("Historical PIT requires an explicit dataset qualification reference")

    def source_record(self):
        from dataclasses import asdict

        return SourceRecord(
            self.provider,
            "lot5-2026-10-14",
            self.dataset,
            "https://www.factset.com/"
            if self.provider == "factset"
            else "https://www.lseg.com/"
            if self.provider == "ibes"
            else None,
            access="licensed" if self.provider != "licensed_export" else "local",
            license_notes=json.dumps(asdict(self), sort_keys=True),
            availability="historical_pit" if self.qualified_historical_pit else "receipt",
        )


class LicensedEstimateAdapter:
    def __init__(self, license, *, column_map=None, value_maps=None):
        self.license = EstimateLicense(**license) if isinstance(license, dict) else license
        if not isinstance(self.license, EstimateLicense):
            raise ValueError("Licensed adapter requires EstimateLicense")
        self.column_map = column_map or {
            key: key
            for key in (
                *ESTIMATE_REQUIRED,
                "available_at",
                "stddev",
                "analyst_count",
                "announcement_at",
            )
        }
        if not set(ESTIMATE_REQUIRED) <= set(self.column_map) or any(
            not isinstance(v, str) or not v for v in self.column_map.values()
        ):
            raise ValueError("Export mapping must identify all required consensus fields")
        self.value_maps = value_maps or {}

    @staticmethod
    def _export_rows(body, *, max_rows, format):
        if format == "json":
            rows = _json(body)
        elif format == "csv":
            reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("Estimate CSV headers must be unique")
            rows = list(reader)
        else:
            raise ValueError("Estimate export format must be json or csv")
        if not isinstance(rows, list) or len(rows) > max_rows:
            raise ValueError("Estimate export must be a bounded row array")
        _reject_secrets(rows)
        return rows

    def parse(self, body, *, received_at, max_rows=1000, format="json"):
        rows = self._export_rows(body, max_rows=max_rows, format=format)
        records = []
        for raw in rows:
            if not isinstance(raw, dict) or not {
                self.column_map[k] for k in ESTIMATE_REQUIRED
            } <= set(raw):
                raise ValueError("Estimate row misses explicitly mapped fields")
            row = {k: raw.get(v) for k, v in self.column_map.items()}
            for key, mapping in self.value_maps.items():
                if row.get(key) not in mapping:
                    raise ValueError("Unmapped provider estimate category")
                row[key] = mapping[row[key]]
            for key in ESTIMATE_CONTEXT:
                if key == "segment" and row[key] is None:
                    continue
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError("Estimate period/unit/accounting context must be explicit")
            for key in ("period_start", "period_end"):
                _period(row[key], "daily")
            if row["period_start"] > row["period_end"] or row["period_kind"] not in (
                "quarter",
                "annual",
                "YTD",
                "instant",
            ):
                raise ValueError("Invalid estimate fiscal period")
            if row["statistic"] not in ("mean", "median", "detail", "actual") or row[
                "statement_type"
            ] not in ("analyst_estimate", "reported_actual"):
                raise ValueError("Guidance is not analyst consensus or reported actual")
            if (row["statistic"] == "actual") != (row["statement_type"] == "reported_actual"):
                raise ValueError("Estimate statistic and statement type disagree")
            if row["statistic"] == "detail" and (
                not isinstance(row["analyst_id"], str) or not row["analyst_id"]
            ):
                raise ValueError("Detail estimate requires analyst identity")
            if row["statistic"] != "detail" and row["analyst_id"] is not None:
                raise ValueError("Consensus/actual cannot carry a single analyst identity")
            if type(row["active"]) is not bool:
                raise ValueError(
                    "Estimate active must be explicit boolean, map CSV categories first"
                )
            snapshot = _instant(row["snapshot_at"], "snapshot_at")
            receipt = _instant(received_at, "received_at")
            publication = (
                _instant(row["published_at"], "published_at")
                if row["published_at"] is not None
                else None
            )
            if snapshot > receipt:
                raise ValueError("Estimate snapshot cannot follow receipt")
            value = _number(row["value"])
            row["value"] = value
            row["snapshot_at"] = snapshot.isoformat()
            for key in ("stddev", "analyst_count"):
                row[key] = _number(row.get(key))
                if row[key] is not None and (
                    row[key] < 0 or key == "analyst_count" and row[key] % 1
                ):
                    raise ValueError(
                        "Estimate dispersion/count must be nonnegative with integer count"
                    )
            if row.get("announcement_at") is not None:
                row["announcement_at"] = _instant(
                    row["announcement_at"], "announcement_at"
                ).isoformat()
            context = {k: row[k] for k in ESTIMATE_CONTEXT}
            context_id = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[
                :24
            ]
            component = row["statistic"] + (
                ":" + row["analyst_id"] if row["statistic"] == "detail" else ""
            )
            point = _point(
                body,
                received_at=received_at,
                source=self.license.provider,
                entity_id=row["entity_id"],
                metric="estimates." + context_id + "." + component + "." + str(snapshot.value),
                unit=row["currency"],
                period=row["period_start"],
                frequency="daily",
                value=None,
                country=None,
                currency=row["currency"],
                published_at=publication,
                evidence=json.dumps(raw, sort_keys=True, ensure_ascii=False),
            )
            point.update(
                value=row,
                revision_id=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
                period_end=row["period_end"],
                frequency="fiscal",
                context_id=context_id,
                **{"quality.value_known": value is not None},
            )
            if self.license.qualified_historical_pit:
                if row.get("available_at") is None:
                    raise ValueError(
                        "Qualified historical export needs actual available_at per version"
                    )
                at = _instant(row["available_at"], "available_at")
                if at < snapshot or publication is not None and at < publication:
                    raise ValueError("Estimate availability precedes snapshot/publication")
                point.update(
                    available_at=at,
                    availability_basis="publication" if publication is not None else "unknown",
                    point_in_time=True,
                )
            else:
                point["available_at"] = max(receipt, snapshot, publication or receipt)
            if (
                row["statement_type"] == "reported_actual"
                and row.get("announcement_at") is not None
                and _instant(row["announcement_at"], "announcement_at") > point["available_at"]
            ):
                raise ValueError("Reported actual cannot be available before its announcement")
            records.append(point)
        return normalize_observations(pd.DataFrame(records)) if records else _observations([])

    def capture(
        self, store, *, snapshot_id, query, offline=None, fetch=None, limits=None, format="json"
    ):
        if not isinstance(query, dict) or not query.get("entity_ids"):
            raise ValueError("Licensed collection requires a selected entity universe")
        limits = limits or CaptureLimits()

        def parser(body, *, received_at, max_rows):
            frame = self.parse(body, received_at=received_at, max_rows=max_rows, format=format)
            if not set(frame.entity_id) <= set(query["entity_ids"]):
                raise ValueError("Licensed response includes unrequested entities")
            return frame, ("export_at_row_cap_coverage_unknown",) if len(frame) >= max_rows else ()

        return capture_observations(
            store=store,
            source=self.license.provider,
            dataset=self.license.dataset,
            snapshot_id=snapshot_id,
            query={
                "selection": query,
                "columns": self.column_map,
                "categories": self.value_maps,
                "format": format,
            },
            declaration=self.license.source_record().to_dict(),
            parser=parser,
            fetch=fetch,
            offline=offline,
            limits=limits,
            validate_body=lambda body: self._export_rows(
                body, max_rows=limits.max_rows, format=format
            ),
        )
