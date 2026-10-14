"""Structural and evidence checks; semantic correctness requires separate audit.

Exact citations/numeric presence cannot establish every semantic claim. Accepted
records are explicitly unreviewed until a named human adjudicates them. Graphs
and events default to requiring that audit, including supplier/customer direction.
"""

import json
import math
import re
from datetime import date

from pfe.data.contracts import _instant
from pfe.data.documents import content_hash
from pfe.data.storage import _json_bytes
from pfe.extraction.schemas import ENVELOPES, SCHEMA_VERSION, extraction_schema


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError("Non-finite JSON number")

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite JSON number")
        return result

    return json.loads(
        text, object_pairs_hook=pairs, parse_constant=bad_constant, parse_float=number
    )


def _check(value, schema):
    if "anyOf" in schema:
        for alternative in schema["anyOf"]:
            try:
                _check(value, alternative)
                return
            except ValueError:
                pass
        raise ValueError("Invalid nullable number/range")
    kind = (
        "null"
        if value is None
        else "boolean"
        if isinstance(value, bool)
        else "number"
        if type(value) in (int, float)
        else "string"
        if isinstance(value, str)
        else "array"
        if isinstance(value, list)
        else "object"
        if isinstance(value, dict)
        else "invalid"
    )
    allowed = schema["type"]
    if kind not in (allowed if isinstance(allowed, list) else [allowed]):
        raise ValueError("Field type mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Unknown enum value")
    if kind == "number":
        if (
            not math.isfinite(value)
            or value < schema.get("minimum", -math.inf)
            or value > schema.get("maximum", math.inf)
        ):
            raise ValueError("Number outside finite bounds")
    if kind == "string" and len(value.strip()) < schema.get("minLength", 1):
        raise ValueError("Empty string must use null")
    if kind == "array":
        for item in value:
            _check(item, schema["items"])
    if kind == "object":
        if set(value) != set(schema["required"]):
            raise ValueError("Missing or unknown schema fields")
        for name, item in value.items():
            _check(item, schema["properties"][name])


def _contains(text, name):
    return bool(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, flags=re.I))


def _unit_contains(text, value):
    """Unit markers can touch numbers (20%, USD20, US$20), never other letters."""
    label = str(value)
    prefix = r"(?<![A-Za-z])" if label[0].isalpha() else ""
    suffix = r"(?![A-Za-z])" if label[-1].isalpha() else ""
    return bool(re.search(prefix + re.escape(label) + suffix, text, flags=re.I))


def date_in_evidence(value, quote):
    """ISO or explicit English/French day-month-year, without fiscal inference."""
    point = date.fromisoformat(value)
    if value in quote:
        return True
    months = (
        "January|Jan|janvier",
        "February|Feb|février",
        "March|Mar|mars",
        "April|Apr|avril",
        "May|mai",
        "June|Jun|juin",
        "July|Jul|juillet",
        "August|Aug|août",
        "September|Sept|Sep|septembre",
        "October|Oct|octobre",
        "November|Nov|novembre",
        "December|Dec|décembre",
    )
    month = "(?:" + months[point.month - 1] + r")\.?"
    day = ("0?" if point.day < 10 else "") + rf"{point.day}(?:st|nd|rd|th|er)?"
    patterns = (
        rf"\b{month}\s+{day}\s*,?\s*{point.year}\b",
        rf"\b{day}\s+{month}\s*,?\s*{point.year}\b",
    )
    return any(re.search(pattern, quote, re.I) for pattern in patterns)


def _numbers(value):
    if type(value) in (int, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)


def _percentage_supported(value, quote):
    marker = r"(?:%|percent\b|per\s+cent\b|pour\s+cent\b)"
    number = r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    annotated = [
        float(match.replace(",", ""))
        for match in re.findall(r"(?<![\w.])(" + number + r")\s*" + marker, quote, re.I)
    ]

    if type(value) in (int, float):
        return value in annotated
    if isinstance(value, dict):
        low, high = value["lower_bound"], value["upper_bound"]
        if low in annotated and high in annotated:
            return True
        pattern = (
            r"(?<![\w.])("
            + number
            + r")\s*(?:"
            + marker
            + r")?\s*(?:-|–|to|and)\s*("
            + number
            + r")\s*"
            + marker
        )
        return any(
            float(a.replace(",", "")) == low and float(b.replace(",", "")) == high
            for a, b in re.findall(pattern, quote, re.I)
        )
    return True


def _direction_conflict(record, quote):
    """Reject obvious English inversions; multilingual/indirect language needs audit."""
    a, b = record["source_company"], record["related_entity"]
    role = record["relation_type"]
    if not a or not b or role not in ("supplier", "customer"):
        return False
    supplier, customer = (a, b) if role == "supplier" else (b, a)
    wrong = [
        rf"{re.escape(customer)}\s+(?:supplies|provides\s+\w+\s+to)\s+{re.escape(supplier)}",
        rf"{re.escape(customer)}\s+is\s+(?:a\s+|the\s+)?supplier\s+(?:of|to)\s+{re.escape(supplier)}",
        rf"{re.escape(supplier)}\s+is\s+(?:a\s+|the\s+)?customer\s+of\s+{re.escape(customer)}",
        rf"{re.escape(supplier)}\s+(?:is\s+)?supplied\s+by\s+{re.escape(customer)}",
        rf"{re.escape(supplier)}\s+(?:buys|purchases|sources)\s+[^.;\n]{{0,60}}?from\s+{re.escape(customer)}",
    ]
    return any(re.search(pattern, quote, re.I) for pattern in wrong)


def validate_fact(task, record, document, *, chunk_start=0, chunk_end=None):
    """Check all supplied fields, exact chunk-contained evidence and explicit metadata."""
    schema = extraction_schema(task)["properties"][ENVELOPES[task]]["items"]
    _check(record, schema)
    if record["source_document_id"] != document.source_document_id:
        raise ValueError("Wrong source_document_id")
    quote = record["evidence_quote"]
    end = len(document.text) if chunk_end is None else chunk_end
    offset = document.text.find(quote, chunk_start, end)
    if offset < 0:
        raise ValueError("Citation is not an exact excerpt from the supplied chunk")
    entities = (
        [record.get("source_company"), record.get("related_entity")]
        if task == "A"
        else record.get("entities_involved") or []
        if task == "D"
        else [record.get("company")]
    )
    for entity in entities:
        if entity is not None and not _contains(quote, entity):
            raise ValueError("Entity attribution is absent from evidence_quote")
    numeric_tokens = set()
    number_pattern = r"[+-]?(?:\d[\d,]*\.?\d*|\.\d+)"
    tokens = re.findall(r"(?<!\w)" + number_pattern + r"(?!\w)", quote)
    currency = record.get("currency")
    if currency:
        tokens += re.findall(re.escape(currency) + r"\s*(" + number_pattern + r")(?![\d.])", quote)
    for field in ("unit_scale", "magnitude_unit"):
        unit = record.get(field)
        if isinstance(unit, str):
            tokens += re.findall(r"(?<![\d.])(" + number_pattern + r")\s*" + re.escape(unit), quote)
    for number in tokens:
        try:
            numeric_tokens.add(float(number.replace(",", "")))
        except ValueError:
            pass
    if task == "C":
        for number in re.findall(
            r"\(\s*(?:USD|US\$|\$|€|£)?\s*(" + number_pattern + r")\s*\)", quote
        ):
            numeric_tokens.add(-abs(float(number.replace(",", ""))))
    for key, value in record.items():
        if isinstance(value, dict) and value["lower_bound"] > value["upper_bound"]:
            raise ValueError("Reversed numeric range")
        if key == "unit_scale":
            if value is not None and not _unit_contains(quote, value):
                raise ValueError("Unit scale absent from evidence_quote")
        elif any(number not in numeric_tokens for number in _numbers(value)):
            raise ValueError("Numeric value absent from evidence_quote")
        if key in (
            "revenue_dependency_pct",
            "supply_dependency_pct",
            "exposure_percentage",
        ) and not _percentage_supported(value, quote):
            raise ValueError("Percentage value/range has no explicit percent evidence")
    for field in (
        "currency",
        "magnitude_unit",
        "fiscal_period_label",
        "fiscal_period",
        "effective_period",
        "segment_name",
        "country_or_region",
        "product_or_material",
        "products_or_services",
        "risk_or_opportunity_description",
        "expected_economic_mechanism",
    ):
        value = record.get(field)
        supported = (
            _unit_contains(quote, value)
            if value is not None and field in ("currency", "magnitude_unit")
            else value is None or _contains(quote, value)
        )
        if not supported:
            raise ValueError(f"{field} absent from evidence_quote; preserve original wording")
    for field in (
        "countries_involved",
        "affected_countries",
        "affected_industries",
        "affected_products",
    ):
        if any(not _contains(quote, value) for value in record.get(field) or []):
            raise ValueError(f"{field} absent from evidence_quote")
    if task == "A" and _direction_conflict(record, quote):
        raise ValueError("Supplier/customer direction contradicts evidence_quote")
    if task == "B" and record["exposure_type"] == "revenue":
        if re.search(r"\bheadquarters\b", quote, re.I) and not re.search(
            r"\b(revenue|sales|turnover)\b", quote, re.I
        ):
            raise ValueError("Headquarters alone is not revenue exposure")
    if task == "C":
        speculative = re.search(
            r"\b(guidance|expects?|forecast|estimates?|consensus)\b", quote, re.I
        )
        actual = re.search(r"\b(reported|actual|earned|was|were)\b", quote, re.I)
        analyst = re.search(r"\b(analysts?|consensus)\b", quote, re.I)
        if record["statement_type"] == "reported_actual" and speculative and not actual:
            raise ValueError("Guidance/estimate cannot be labeled reported_actual")
        if (
            record["statement_type"] == "analyst_estimate"
            and not analyst
            and re.search(r"\b(guidance|management|expects?)\b", quote, re.I)
        ):
            raise ValueError("Management guidance cannot be labeled analyst_estimate")
        for field in ("fiscal_period_start", "fiscal_period_end", "announcement_date"):
            value = record[field]
            if value is None:
                continue
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Financial dates require ISO economic dates")
            publication_date = str(document.published_at)[:10] if document.published_at else None
            if not date_in_evidence(value, quote) and not (
                field == "announcement_date" and value == publication_date
            ):
                raise ValueError("Financial date is not explicit in the source")
        start, finish = record["fiscal_period_start"], record["fiscal_period_end"]
        if start and finish and start > finish:
            raise ValueError("Reversed fiscal period")
        low, high = record["lower_bound"], record["upper_bound"]
        if low is not None and high is not None and low > high:
            raise ValueError("Reversed financial bounds")
    if task == "D" and record["event_status"] == "confirmed":
        if re.search(
            r"\b(rumors?|may\s+(?:acquire|launch|merge)|might\s+(?:acquire|launch|merge))\b",
            quote,
            re.I,
        ) and not re.search(r"\bconfirmed\b", quote, re.I):
            raise ValueError("Explicit speculative event cannot be marked confirmed")
    if task == "D" and record["first_known_publication_time"] is not None:
        timestamp = record["first_known_publication_time"]
        instant = _instant(timestamp, "first_known_publication_time")
        metadata_match = document.published_at and instant == _instant(
            document.published_at, "published_at"
        )
        if timestamp not in quote and not metadata_match:
            raise ValueError("Event publication timestamp has no source evidence")
        if instant > _instant(document.received_at, "received_at"):
            raise ValueError("Event publication cannot follow document receipt")
    fact_id = "fact:" + content_hash(_json_bytes({"task": task, "record": record}))
    return {
        "fact_id": fact_id,
        "task": task,
        "record": record,
        "source_document_id": document.source_document_id,
        "text_sha256": document.text_sha256,
        "evidence_start": offset,
        "evidence_end": offset + len(quote),
        "evidence_pages": [
            page["page"]
            for page in document.pages
            if page["start"] < offset + len(quote) and page["end"] > offset
        ],
        "schema_version": SCHEMA_VERSION,
        "review_status": "unreviewed",
        "published_at": document.published_at,
        "document_available_at": document.available_at,
        "source_id": document.source_id,
        "revision_id": document.revision_id,
        "point_in_time": True,
        "availability_basis": "receipt",
    }


def adjudicate_fact(
    fact,
    *,
    approved,
    reviewer,
    reviewed_at,
    notes,
    document,
    effective_from=None,
    effective_to=None,
):
    """Append a named semantic review; never backdate automatic extraction or review.

    Reviewer checks every populated field, attribution, percentages, units,
    actual/guidance/estimate classification, event status and relationship direction.
    Explicit economic dates may be attached only with supporting source evidence.
    """
    if type(approved) is not bool or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Review requires boolean approved and named reviewer")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("Review requires explanatory notes")
    checked = validate_fact(fact["task"], fact["record"], document)
    if checked["fact_id"] != fact["fact_id"]:
        raise ValueError("Review fact identity mismatch")
    if fact["text_sha256"] != document.text_sha256:
        raise ValueError("Review document hash mismatch")
    if (
        document.text[fact["evidence_start"] : fact["evidence_end"]]
        != fact["record"]["evidence_quote"]
    ):
        raise ValueError("Review evidence offsets mismatch")
    for name in ("source_id", "revision_id", "published_at", "source_document_id"):
        if fact.get(name) != getattr(document, name):
            raise ValueError("Review document provenance mismatch")
    at = _instant(reviewed_at, "reviewed_at")
    if at < _instant(fact["available_at"], "available_at"):
        raise ValueError("Review cannot precede extraction availability")
    if _instant(fact["available_at"], "available_at") < _instant(
        document.available_at, "available_at"
    ):
        raise ValueError("Extraction cannot precede document availability")
    for value in (effective_from, effective_to):
        if value is not None and (
            date.fromisoformat(value).isoformat() != value
            or not date_in_evidence(value, fact["record"]["evidence_quote"])
        ):
            raise ValueError("Economic validity date requires exact source evidence")
    if effective_from and effective_to and effective_from >= effective_to:
        raise ValueError("Economic interval must be nonempty [from, to)")
    reviewed = {
        **fact,
        "review_status": "approved" if approved else "rejected",
        "reviewer": reviewer,
        "reviewed_at": at.isoformat(),
        "review_notes": notes,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "available_at": at.isoformat(),
    }
    reviewed.pop("review_id", None)
    reviewed["review_id"] = "review:" + content_hash(_json_bytes(reviewed))
    return reviewed
