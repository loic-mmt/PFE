"""Explicit source schemas for small international captures.

Amounts, economic periods and publication precision are preserved. Missing
values remain null. Date-only updates, filing numbers and GDELT detection times
never become verified publication timestamps. All default availability is receipt.
"""

import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from datetime import UTC, datetime
from html.parser import HTMLParser

import pandas as pd

from pfe.data.contracts import _instant, normalize_observations
from pfe.data.eia import ContextError, ContextLimitError, _observation, _observations, _period


def _json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContextError("Duplicate provider JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            body,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, UnicodeError, RecursionError):
        raise ContextError("Invalid provider JSON") from None


def _number(value, *, thousands=",", decimal=".", percent=False):
    if value is None:
        return None
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ContextError("Nonfinite provider value")
        return float(value)
    if not isinstance(value, str):
        raise ContextError("Provider numeric value must be a string, number or null")
    value = unicodedata.normalize("NFKC", value).strip().replace("−", "-")
    if value in ("", "-", "--", "...", "NA", "N/A", "ND", ":", "x"):
        return None
    if percent and value.endswith("%"):
        value = value[:-1].strip()
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    if thousands and thousands in value:
        signless = value.lstrip("+-")
        integer = signless.split(decimal)[0]
        if not re.fullmatch(r"\d{1,3}(?:" + re.escape(thousands) + r"\d{3})+", integer):
            raise ContextError("Invalid declared thousands grouping")
        value = value.replace(thousands, "")
    if decimal != ".":
        value = value.replace(decimal, ".")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        raise ContextError("Numeric value does not match declared locale")
    result = float(value)
    if not math.isfinite(result):
        raise ContextError("Nonfinite provider value")
    return result


def _finish(records, max_rows):
    if type(max_rows) is not int or max_rows < 1 or len(records) > max_rows:
        raise ContextLimitError("Selected international observations exceed max_rows")
    return normalize_observations(pd.DataFrame(records)) if records else _observations([])


def _point(
    body,
    *,
    received_at,
    source,
    entity_id,
    metric,
    unit,
    period,
    frequency,
    value,
    country,
    currency=None,
    published_at=None,
    evidence=None,
    **metadata,
):
    start, end, _ = _period(period, frequency)
    revision = hashlib.sha256(body).hexdigest()
    record = _observation(
        entity_id=entity_id,
        metric=metric,
        period_start=start,
        period_end=end,
        period_label=period,
        value=value,
        unit=unit,
        source_id=source,
        revision_id=revision,
        received_at=received_at,
        published_at=published_at,
        frequency=frequency,
        calendar=f"{source}_economic_periods",
        currency=currency,
    )
    record.update(
        country=country,
        evidence_quote=evidence,
        source_row_id=metadata.pop("source_row_id", None),
        **metadata,
    )
    return record


class _Tables(HTMLParser):
    """Text cells only. Nested layout tables do not contaminate child rows."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tables, self.stack, self.hidden = [], [], 0
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag == "table":
            table = {"rows": [], "row": None, "cell": None}
            self.tables.append(table["rows"])
            self.stack.append(table)
        if self.stack and tag == "tr":
            self.stack[-1]["row"] = []
        if self.stack and tag in ("th", "td"):
            self.stack[-1]["cell"] = []
        if self.stack and tag == "br" and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"].append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if not self.stack:
            return
        table = self.stack[-1]
        if tag in ("th", "td") and table["cell"] is not None:
            if table["row"] is not None:
                table["row"].append(" ".join("".join(table["cell"]).split()))
            table["cell"] = None
        if tag == "tr" and table["row"] is not None:
            table["rows"].append(table["row"])
            table["row"] = None
        if tag == "table":
            self.stack.pop()

    def handle_data(self, data):
        if not self.hidden and self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"].append(data)


def _text(body, encoding="utf-8-sig"):
    try:
        return body.decode(encoding)
    except UnicodeError:
        raise ContextError("Source text does not match declared encoding") from None


def roc_date(value):
    """Convert explicit ROC YYY/MM/DD dates; never guess Gregorian vs ROC."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{2,3}/\d{2}/\d{2}", value):
        raise ContextError("ROC date requires YYY/MM/DD")
    year, month, day = map(int, value.split("/"))
    try:
        return datetime(year + 1911, month, day).date().isoformat()
    except ValueError:
        raise ContextError("Invalid ROC date") from None


def parse_tsmc_revenue(
    body, *, received_at, year, entity_id, max_rows=5000, publication_times=None
):
    """IR English annual table, millions TWD; total and blank future months excluded."""
    text = _text(body)
    if type(year) is not int or not 1990 <= year <= 2100:
        raise ContextError("TSMC year must be explicit")
    if not re.search(str(year) + r"\s+Monthly Revenue", text, re.I):
        raise ContextError("TSMC page year/title does not match selection")
    if "Millions of New Taiwan Dollars" not in text:
        raise ContextError("TSMC revenue unit not established by source")
    months = {
        m: i + 1
        for i, m in enumerate(
            ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec")
        )
    }
    candidates = [
        rows for rows in _Tables(text).tables if any(r and r[0].rstrip(".") in months for r in rows)
    ]
    if len(candidates) != 1:
        raise ContextError("TSMC monthly table is missing or ambiguous")
    records, seen = [], set()
    for row in candidates[0]:
        if not row or row[0].rstrip(".") not in months:
            continue
        if len(row) != 3:
            raise ContextError("TSMC monthly table must have month/revenue/YoY columns")
        month = f"{year}-{months[row[0].rstrip('.')]:02d}"
        if month in seen:
            raise ContextError("Duplicate TSMC month")
        seen.add(month)
        revenue = _number(row[1])
        publication = (publication_times or {}).get(month)
        if revenue is None:
            period_end = _period(month, "monthly")[1]
            completed_at = pd.Timestamp(period_end, tz="Asia/Taipei") + pd.Timedelta("1D")
            # Blank cells are unreported months, while an explicit missing
            # marker on a completed month is a revision that must mask old data.
            if not row[1].strip() or (
                publication is None and _instant(received_at, "received_at") < completed_at
            ):
                continue
        for metric, unit, value, currency in (
            ("revenue", "million_TWD", revenue, "TWD"),
            ("revenue_yoy", "percentage_points", _number(row[2], percent=True), None),
        ):
            records.append(
                _point(
                    body,
                    received_at=received_at,
                    source="tsmc",
                    entity_id=entity_id,
                    metric=metric,
                    unit=unit,
                    period=month,
                    frequency="monthly",
                    value=value,
                    country="TW",
                    currency=currency,
                    published_at=publication,
                    evidence=" | ".join(row),
                    accounting_basis="consolidated",
                    source_row_id=month,
                )
            )
    return _finish(records, max_rows)


def parse_mops_revenue(
    body, *, received_at, companies, max_rows=5000, publication_times=None, encoding="utf-8-sig"
):
    """Official t187ap05_L snapshot, exact issuer mappings, thousands TWD.

    CSV and JSON supported. 出表日期 is the table date, not issuer publication.
    Only current monthly amount/YoY are selected, never YTD as quarterly revenue.
    """
    text = _text(body, encoding)
    if not isinstance(companies, dict) or not companies:
        raise ContextError("MOPS requires company-code/entity-ID mappings")
    if text.lstrip().startswith("["):
        rows = _json(body)
    else:
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ContextError("MOPS CSV has missing or duplicate headers")
        rows = list(reader)
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ContextLimitError("MOPS source rows exceed max_rows")
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise ContextError("MOPS rows must be objects")
        code = row.get("公司代號")
        if code not in companies:
            continue
        label = row.get("資料年月", "")
        if not isinstance(label, str) or not re.fullmatch(r"\d{5}", label):
            raise ContextError("MOPS period requires ROC YYYMM")
        period = f"{int(label[:3]) + 1911}-{label[3:]}"
        for field, metric, unit, currency in (
            ("營業收入-當月營收", "revenue", "thousand_TWD", "TWD"),
            ("營業收入-去年同月增減(%)", "revenue_yoy", "percentage_points", None),
        ):
            if field not in row:
                raise ContextError("MOPS source misses selected revenue columns")
            records.append(
                _point(
                    body,
                    received_at=received_at,
                    source="mops",
                    entity_id=companies[code],
                    metric=metric,
                    unit=unit,
                    period=period,
                    frequency="monthly",
                    value=_number(row[field], percent=True),
                    country="TW",
                    currency=currency,
                    published_at=(publication_times or {}).get(code + ":" + period),
                    evidence=json.dumps(row, ensure_ascii=False, sort_keys=True),
                    source_row_id=code + ":" + label,
                    table_date=row.get("出表日期"),
                )
            )
    return _finish(records, max_rows)


def parse_twse_daily(
    body, *, received_at, month, stock_no, entity_id, max_rows=5000, close_schedule=None
):
    """Official one-stock/month OHLCV; ROC dates, shares, TWD, adjustment unknown.

    Explicit close_schedule establishes observed_at only; receipt availability
    is retained. Holidays are not synthesized. No US calendar is applied.
    """
    payload = _json(body)
    fields = ("日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價")
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        raise ContextError("TWSE response is not successful")
    header, rows = payload.get("fields"), payload.get("data")
    if (
        not isinstance(header, list)
        or len(set(header)) != len(header)
        or not set(fields) <= set(header)
    ):
        raise ContextError("TWSE response misses exact OHLCV headers")
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ContextLimitError("TWSE source rows exceed max_rows")
    if "date" in payload and str(payload["date"])[:6] != month.replace("-", ""):
        raise ContextError("TWSE returned a different month")
    title = payload.get("title", "")
    if not re.search(r"(?<!\d)" + re.escape(stock_no) + r"(?!\d)", title):
        raise ContextError("TWSE returned stock identity is not established")
    records = []
    for index, values in enumerate(rows):
        if not isinstance(values, list) or len(values) != len(header):
            raise ContextError("Malformed TWSE row")
        row = dict(zip(header, values, strict=True))
        day = roc_date(row["日期"])
        if day[:7] != month:
            raise ContextError("TWSE row outside selected month")
        numbers = {f: _number(row[f]) for f in fields[1:]}
        if numbers["成交股數"] is not None and (numbers["成交股數"] < 0 or numbers["成交股數"] % 1):
            raise ContextError("TWSE volume must be nonnegative shares")
        o, h, lo, c = (numbers[f] for f in fields[3:])
        if all(v is not None for v in (o, h, lo, c)) and not (
            0 < lo <= min(o, c) <= max(o, c) <= h
        ):
            raise ContextError("Inconsistent TWSE OHLC values")
        observed = None
        if close_schedule is not None:
            if day not in close_schedule:
                raise ContextError("TWSE date missing from explicit venue close schedule")
            observed = _instant(close_schedule[day], "venue_close_at")
            if observed.tz_convert("Asia/Taipei").date().isoformat() != day:
                raise ContextError("TWSE close must belong to its local session date")
        for field, metric in zip(
            fields[1:], ("volume", "turnover", "open", "high", "low", "close"), strict=True
        ):
            records.append(
                _point(
                    body,
                    received_at=received_at,
                    source="twse",
                    entity_id=entity_id,
                    metric=metric,
                    unit="shares" if metric == "volume" else "TWD",
                    period=day,
                    frequency="daily",
                    value=numbers[field],
                    country="TW",
                    currency=None if metric == "volume" else "TWD",
                    evidence=json.dumps(row, ensure_ascii=False),
                    source_row_id=str(index),
                    observed_at=observed,
                    exchange="XTAI",
                    price_adjustment="unknown",
                )
            )
    return _finish(records, max_rows)


def parse_opendart_accounts(
    body,
    *,
    received_at,
    corp_code,
    year,
    report_code,
    entity_id,
    accounts,
    max_rows=5000,
    publication_times=None,
):
    """English major accounts; explicit statement/basis/amount/period selection.

    ``accounts`` specifies account_nm, fs_div, sj_div, amount_field, metric,
    period_start/end, period_kind (instant/quarter/YTD/annual) and currency.
    No fiscal boundaries, quarterly subtraction or KRW currency are guessed.
    """
    payload = _json(body)
    if not isinstance(payload, dict) or payload.get("status") not in ("000", "013"):
        raise ContextError("OpenDART provider status is unsuccessful")
    if payload["status"] == "013":
        return _finish([], max_rows)
    rows = payload.get("list")
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ContextLimitError("OpenDART source rows exceed max_rows")
    if not isinstance(accounts, list) or not accounts:
        raise ContextError("OpenDART requires exact selected accounts and fiscal periods")
    records = []
    for selection in accounts:
        required = {
            "account_nm",
            "fs_div",
            "sj_div",
            "amount_field",
            "metric",
            "period_start",
            "period_end",
            "period_kind",
            "currency",
        }
        if not isinstance(selection, dict) or set(selection) != required:
            raise ContextError("OpenDART account selection needs exact fields and fiscal basis")
        if selection["fs_div"] not in ("CFS", "OFS") or selection["amount_field"] not in (
            "thstrm_amount",
            "thstrm_add_amount",
        ):
            raise ContextError("Invalid OpenDART statement/amount basis")
        if selection["period_kind"] not in ("instant", "quarter", "YTD", "annual"):
            raise ContextError("Declare OpenDART instant/quarter/YTD/annual period")
        start, end = selection["period_start"], selection["period_end"]
        _period(start, "daily")
        _period(end, "daily")
        if start > end or selection["period_kind"] == "instant" and start != end:
            raise ContextError("OpenDART fiscal interval is inconsistent")
        selected = [
            r
            for r in rows
            if isinstance(r, dict)
            and all(r.get(k) == selection[k] for k in ("account_nm", "fs_div", "sj_div"))
        ]
        if len(selected) > 1:
            raise ContextError("Ambiguous OpenDART selected account")
        for row in selected:
            if str(row.get("bsns_year")) != str(year) or row.get("reprt_code") != report_code:
                raise ContextError("OpenDART response disagrees with report selection")
            if "corp_code" in row and row["corp_code"] != corp_code:
                raise ContextError("OpenDART company identity disagrees")
            receipt_no = row.get("rcept_no")
            if not isinstance(receipt_no, str) or not re.fullmatch(r"\d{14}", receipt_no):
                raise ContextError("OpenDART filing identifier missing")
            currency = row.get("currency")
            if currency != selection["currency"] or not isinstance(currency, str) or not currency:
                raise ContextError("OpenDART currency disagrees or is absent")
            if selection["amount_field"] not in row:
                raise ContextError("OpenDART selected amount field is absent")
            provider_dates = re.findall(
                r"\b(\d{4})[.\-](\d{2})[.\-](\d{2})\b", row.get("thstrm_dt", "")
            )
            provider_dates = ["-".join(parts) for parts in provider_dates]
            if provider_dates and provider_dates[-1] != end:
                raise ContextError("OpenDART declared period end differs from source term dates")
            if (
                len(provider_dates) == 2
                and selection["amount_field"] == "thstrm_amount"
                and provider_dates[0] != start
            ):
                raise ContextError(
                    "OpenDART declared period start differs from source amount period"
                )
            record = _point(
                body,
                received_at=received_at,
                source="opendart",
                entity_id=entity_id,
                metric=selection["metric"],
                unit=currency,
                period=start,
                frequency="daily",
                value=_number(row[selection["amount_field"]]),
                country="KR",
                currency=currency,
                published_at=(publication_times or {}).get(receipt_no),
                evidence=json.dumps(row, ensure_ascii=False, sort_keys=True),
                source_row_id=receipt_no + ":" + str(row.get("ord", selection["account_nm"])),
                filing_id=receipt_no,
                accounting_basis=selection["fs_div"],
                statement_type="reported_actual",
                period_kind=selection["period_kind"],
                provider_period_label=row.get("thstrm_dt"),
                amount_field=selection["amount_field"],
                **{
                    "quality.period_end_verified": bool(provider_dates),
                    "quality.period_start_declared": selection["amount_field"]
                    == "thstrm_add_amount"
                    or not provider_dates,
                },
            )
            record.update(
                period_end=end,
                observation_period=start + "/" + end,
                frequency=(
                    "annual"
                    if selection["period_kind"] == "annual"
                    and start[:4] == end[:4]
                    and start.endswith("-01-01")
                    and end.endswith("-12-31")
                    else "fiscal"
                ),
            )
            records.append(record)
    return _finish(records, max_rows)


def parse_eurostat(
    body, *, received_at, dataset, dimensions, start, end, entity_id, metric, max_rows=5000
):
    """JSON-stat singleton factor, sparse null/status preservation, latest only."""
    payload = _json(body)
    if not isinstance(payload, dict) or payload.get("class") != "dataset":
        raise ContextError("Eurostat requires a JSON-stat dataset")
    ids, sizes, dims = payload.get("id"), payload.get("size"), payload.get("dimension")
    if (
        not isinstance(ids, list)
        or not isinstance(sizes, list)
        or len(ids) != len(sizes)
        or len(set(ids)) != len(ids)
        or "time" not in ids
        or not isinstance(dims, dict)
    ):
        raise ContextError("Invalid Eurostat cube dimensions")
    if set(dimensions) != set(ids) - {"time"}:
        raise ContextError("Select exactly one code for every non-time Eurostat dimension")
    categories = {}
    for name, size in zip(ids, sizes, strict=True):
        category = dims.get(name, {}).get("category", {}).get("index")
        if isinstance(category, list):
            category = {v: i for i, v in enumerate(category)}
        if (
            not isinstance(category, dict)
            or type(size) is not int
            or size < 1
            or set(category.values()) != set(range(size))
            or len(category) != size
        ):
            raise ContextError("Malformed Eurostat category positions")
        categories[name] = category
        if name != "time" and (size != 1 or set(category) != {dimensions[name]}):
            raise ContextError("Eurostat returned an unselected dimensional combination")
    total = math.prod(sizes)
    if total > max_rows:
        raise ContextLimitError("Eurostat response cells exceed max_rows")
    values, statuses = payload.get("value", {}), payload.get("status", {})
    if not isinstance(values, (dict, list)) or not isinstance(statuses, (dict, list)):
        raise ContextError("Eurostat values/statuses require sparse objects or arrays")
    for cells in (values, statuses):
        if isinstance(cells, list) and len(cells) != total:
            raise ContextError("Eurostat array length differs from cube")
        if isinstance(cells, dict) and any(
            not str(k).isdigit() or not 0 <= int(k) < total for k in cells
        ):
            raise ContextError("Eurostat sparse cell outside cube")
    freq = {"M": "monthly", "Q": "quarterly", "A": "annual"}.get(dimensions.get("freq"))
    if freq is None or "unit" not in dimensions or "geo" not in dimensions:
        raise ContextError("Eurostat requires explicit freq/unit/geo codes")
    _period(start, freq)
    _period(end, freq)
    if start > end:
        raise ContextError("Eurostat period range is reversed")
    records = []
    for label, position in sorted(categories["time"].items(), key=lambda pair: pair[1]):
        if not start <= label <= end:
            raise ContextError("Eurostat period outside requested bounds")
        value = values[position] if isinstance(values, list) else values.get(str(position))
        status = statuses[position] if isinstance(statuses, list) else statuses.get(str(position))
        records.append(
            _point(
                body,
                received_at=received_at,
                source="eurostat",
                entity_id=entity_id,
                metric=metric,
                unit=dimensions["unit"],
                period=label,
                frequency=freq,
                value=_number(value, thousands=None),
                country=dimensions["geo"],
                evidence=json.dumps(
                    {
                        "dataset": dataset,
                        "dimensions": dimensions,
                        "time": label,
                        "value": value,
                        "status": status,
                    }
                ),
                source_row_id=str(position),
                provider_status=status,
                provider_updated_at=payload.get("updated"),
            )
        )
    return _finish(records, max_rows)


def parse_boj(
    body, *, received_at, db, code, start, end, entity_id, metric, unit, frequency, max_rows=5000
):
    """BoJ 2026 English JSON; one series, no DATE/LAST_UPDATE release inference."""
    payload = _json(body)
    if not isinstance(payload, dict) or str(payload.get("STATUS")) != "200":
        raise ContextError("BoJ provider status is unsuccessful")
    series = payload.get("RESULTSET")
    if not isinstance(series, list) or len(series) != 1 or series[0].get("SERIES_CODE") != code:
        raise ContextError("BoJ requires one exact returned series")
    point = series[0]
    expected = {
        "annual": "ANNUAL",
        "quarterly": "QUARTERLY",
        "monthly": "MONTHLY",
        "daily": "DAILY",
    }
    if (
        frequency not in expected
        or point.get("FREQUENCY") != expected[frequency]
        or point.get("UNIT") != unit
    ):
        raise ContextError("BoJ frequency or unit differs from selection")
    vals = point.get("VALUES", {})
    dates, values = vals.get("SURVEY_DATES"), vals.get("VALUES")
    if not isinstance(dates, list) or not isinstance(values, list) or len(dates) != len(values):
        raise ContextError("Malformed BoJ survey dates/values")
    if len(dates) > max_rows:
        raise ContextLimitError("BoJ series exceeds max_rows")
    records = []
    for label, value in zip(dates, values, strict=True):
        label = str(label)
        within = label[:6] if frequency == "daily" else label
        if not start <= within <= end:
            raise ContextError("BoJ survey period outside requested bounds")
        if frequency == "monthly":
            period = label[:4] + "-" + label[4:]
        elif frequency == "quarterly":
            period = label[:4] + "-Q" + str(int(label[4:]))
        elif frequency == "daily":
            period = datetime.strptime(label, "%Y%m%d").date().isoformat()
        else:
            period = label
        records.append(
            _point(
                body,
                received_at=received_at,
                source="boj",
                entity_id=entity_id,
                metric=metric,
                unit=unit,
                period=period,
                frequency=frequency,
                value=_number(value, thousands=None),
                country="JP",
                evidence=json.dumps(
                    {"db": db, "series": code, "survey_date": label, "value": value}
                ),
                source_row_id=label,
                provider_updated_at=str(point.get("LAST_UPDATE")),
            )
        )
    return _finish(records, max_rows)


def parse_estat(
    body,
    *,
    received_at,
    table_id,
    dimensions,
    start_code,
    end_code,
    period_map,
    entity_id,
    metric,
    unit,
    frequency,
    max_rows=5000,
):
    """e-Stat JSON cells with caller-declared time-code to economic-period map."""
    payload = _json(body)
    root = payload.get("GET_STATS_DATA", {}) if isinstance(payload, dict) else {}
    if str(root.get("RESULT", {}).get("STATUS")) != "0":
        raise ContextError("e-Stat provider status is unsuccessful")
    data = root.get("STATISTICAL_DATA", {})
    table = data.get("TABLE_INF", {})
    if str(table.get("@id")) != table_id:
        raise ContextError("e-Stat returned table differs from selection")
    rows = data.get("DATA_INF", {}).get("VALUE", [])
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ContextLimitError("e-Stat source cells exceed max_rows")
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise ContextError("e-Stat cells must be objects")
        returned = {
            k[1:]: v
            for k, v in row.items()
            if k.startswith("@") and k not in ("@time", "@unit", "@annotation")
        }
        if returned != dimensions:
            raise ContextError("e-Stat cell dimensions differ; select every non-time dimension")
        time = row.get("@time")
        if time not in period_map or not start_code <= time <= end_code:
            raise ContextError("e-Stat time code absent from declared period mapping/range")
        if row.get("@unit") != unit:
            raise ContextError("e-Stat source unit differs from selection")
        records.append(
            _point(
                body,
                received_at=received_at,
                source="estat",
                entity_id=entity_id,
                metric=metric,
                unit=unit,
                period=period_map[time],
                frequency=frequency,
                value=_number(row.get("$"), thousands=None),
                country="JP",
                evidence=json.dumps(row, ensure_ascii=False, sort_keys=True),
                source_row_id=time,
                provider_status=row.get("@annotation"),
                provider_updated_at=table.get("UPDATED_DATE"),
            )
        )
    return _finish(records, max_rows)


def parse_nbs_release(
    body,
    *,
    received_at,
    entity_id,
    table_index,
    columns,
    period_map,
    header_evidence,
    max_rows=5000,
    published_at=None,
    encoding="utf-8-sig",
):
    """Selected official HTML table: exact source labels, column indices and units.

    ``columns`` maps metric to {index, unit}; ``period_map`` maps exact row label
    to YYYY-MM. PMI, monthly and cumulative industrial activity remain separate
    metrics. No undocumented National Data API or translation is assumed.
    """
    text = _text(body, encoding)
    tables = _Tables(text).tables
    if type(table_index) is not int or not 0 <= table_index < len(tables):
        raise ContextError("NBS selected table index is absent")
    rows = tables[table_index]
    if (
        not isinstance(header_evidence, str)
        or not header_evidence
        or header_evidence not in " ".join(" ".join(r) for r in rows)
    ):
        raise ContextError("NBS selected table lacks declared header evidence")
    if (
        not isinstance(columns, dict)
        or not columns
        or not isinstance(period_map, dict)
        or not period_map
    ):
        raise ContextError("NBS requires selected columns and exact period labels")
    records, found = [], set()
    for index, row in enumerate(rows):
        if not row or row[0] not in period_map:
            continue
        label = row[0]
        found.add(label)
        for metric, selection in columns.items():
            if (
                set(selection) != {"index", "unit"}
                or type(selection["index"]) is not int
                or not 1 <= selection["index"] < len(row)
            ):
                raise ContextError("NBS metric requires an exact existing value column and unit")
            records.append(
                _point(
                    body,
                    received_at=received_at,
                    source="nbs",
                    entity_id=entity_id,
                    metric=metric,
                    unit=selection["unit"],
                    period=period_map[label],
                    frequency="monthly",
                    value=_number(row[selection["index"]], percent=True),
                    country="CN",
                    published_at=published_at,
                    evidence=" | ".join(row),
                    source_row_id=str(index),
                    provider_period_label=label,
                )
            )
    if found != set(period_map):
        raise ContextError("NBS selected period labels missing from source table")
    return _finish(records, max_rows)


def parse_usgs_csv(
    body,
    *,
    received_at,
    material_id,
    countries,
    country_column,
    period_columns,
    unit,
    max_rows=5000,
    encoding="utf-8-sig",
    header_row=0,
    published_at=None,
    value_suffixes=(),
):
    """One selected MCS world-production CSV, annual amounts, explicit country IDs.

    Period columns map YYYY to exact source column. Units require file metadata.
    Declared estimated/revised suffixes are kept separately; no production share
    or company dependency is inferred from country output.
    """
    rows = list(csv.reader(io.StringIO(_text(body, encoding))))
    if type(header_row) is not int or not 0 <= header_row < len(rows):
        raise ContextError("USGS header row is absent")
    header = rows[header_row]
    if len(set(header)) != len(header) or not {country_column, *period_columns.values()} <= set(
        header
    ):
        raise ContextError("USGS exact selected columns are missing or duplicated")
    if len(rows) > max_rows:
        raise ContextLimitError("USGS CSV rows exceed max_rows")
    if not isinstance(countries, dict) or not countries or not period_columns:
        raise ContextError("USGS requires selected countries and annual columns")
    records = []
    for index, values in enumerate(rows[header_row + 1 :]):
        if not values or all(not v for v in values):
            continue
        if len(values) != len(header):
            raise ContextError("Malformed USGS CSV row")
        row = dict(zip(header, values, strict=True))
        name = row[country_column]
        if name not in countries:
            continue
        for year, column in period_columns.items():
            raw, flags = row[column].strip(), []
            for suffix in value_suffixes:
                if not isinstance(suffix, str) or not suffix:
                    raise ContextError("USGS value suffix must be explicit")
                if raw.endswith(suffix):
                    raw = raw[: -len(suffix)].strip()
                    flags.append(suffix)
            records.append(
                _point(
                    body,
                    received_at=received_at,
                    source="usgs",
                    entity_id=countries[name],
                    metric="production." + material_id,
                    unit=unit,
                    period=year,
                    frequency="annual",
                    value=_number(raw),
                    country=name,
                    published_at=published_at,
                    evidence=json.dumps(row, ensure_ascii=False, sort_keys=True),
                    source_row_id=str(index) + ":" + year,
                    provider_status=" ".join(flags) or None,
                    material_id=material_id,
                )
            )
    return _finish(records, max_rows)


def gdelt_query_params(options, limits):
    query = options.get("query")
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > 512
        or any(ord(c) < 32 for c in query)
    ):
        raise ContextError("GDELT requires a short explicit thematic/entity query")
    start = _instant(options.get("start"), "GDELT start")
    end = _instant(options.get("end"), "GDELT end")
    if not pd.Timedelta("15min") <= end - start <= pd.Timedelta("7D"):
        raise ContextError("GDELT selected interval must be 15 minutes to 7 days")
    count = options.get("max_records", 100)
    if type(count) is not int or not 1 <= count <= min(250, limits.max_rows):
        raise ContextError("GDELT max_records must fit 1..250 and row budget")
    return {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "maxrecords": count,
    }


def parse_gdelt_articles(
    body, *, received_at, query, start, end, entity_id, max_records=100, max_rows=5000
):
    """DOC article discovery, not verified financial events or full article text."""
    payload = _json(body)
    rows = payload.get("articles") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) > min(max_records, max_rows):
        raise ContextLimitError("GDELT articles missing or exceed selected cap")
    first, last, receipt = (
        _instant(start, "start"),
        _instant(end, "end"),
        _instant(received_at, "received_at"),
    )
    records, seen = [], set()
    revision = hashlib.sha256(body).hexdigest()
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("url"), str)
            or not row["url"].startswith(("http://", "https://"))
        ):
            raise ContextError("GDELT article requires an explicit URL")
        try:
            detected = pd.Timestamp(
                datetime.strptime(row["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            )
        except (KeyError, ValueError, TypeError):
            raise ContextError(
                "GDELT seendate must be an explicit UTC detection timestamp"
            ) from None
        if not first <= detected <= last or detected > receipt:
            raise ContextError("GDELT detection outside query/capture interval")
        identity = (row["url"], row["seendate"])
        if identity in seen:
            continue
        seen.add(identity)
        record = _point(
            body,
            received_at=received_at,
            source="gdelt",
            entity_id=entity_id,
            metric="article_detection." + hashlib.sha256(row["url"].encode()).hexdigest()[:24],
            unit="article_metadata",
            period=detected.date().isoformat(),
            frequency="daily",
            value=None,
            country=row.get("sourcecountry"),
            evidence=json.dumps(row, ensure_ascii=False),
            source_row_id=str(index),
            detected_at=detected.isoformat(),
        )
        record.update(
            value={
                "url": row["url"],
                "title": row.get("title"),
                "language": row.get("language"),
                "domain": row.get("domain"),
                "source_country": row.get("sourcecountry"),
                "query": query,
                "detected_at": detected.isoformat(),
                "first_known_publication_time": None,
            },
            revision_id=revision,
            **{"quality.value_known": True},
        )
        records.append(record)
    return _finish(records, max_rows)


def parse_gdelt_events(
    body, *, received_at, entity_id, country_codes, root_codes=(), max_rows=5000
):
    """Selected plain GDELT 2.0 61-column TSV, no compressed archive download.

    Actors/CAMEO codes are provider model outputs, never reviewed business links.
    DATEADDED is detection; SQLDATE is an economic date. Country codes retain
    their original CAMEO/FIPS systems. No automatic ISO/company mapping.
    """
    if not country_codes:
        raise ContextError("GDELT events require selected country codes")
    records = []
    for index, row in enumerate(csv.reader(io.StringIO(_text(body)), delimiter="\t")):
        if index >= max_rows:
            raise ContextLimitError("GDELT TSV rows exceed max_rows")
        if len(row) != 61:
            raise ContextError("GDELT 2.0 Event TSV requires 61 columns")
        if (
            not {row[7], row[17], row[53]} & set(country_codes)
            or root_codes
            and row[29] not in root_codes
        ):
            continue
        try:
            day = datetime.strptime(row[1], "%Y%m%d").date().isoformat()
            detected = datetime.strptime(row[59], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            raise ContextError("Invalid GDELT event economic/detection dates") from None
        if detected > _instant(received_at, "received_at"):
            raise ContextError("GDELT event detection follows receipt")
        record = _point(
            body,
            received_at=received_at,
            source="gdelt_events",
            entity_id=entity_id,
            metric="cameo_event." + row[0],
            unit="provider_event",
            period=day,
            frequency="daily",
            value=None,
            country=row[53] or None,
            evidence="\t".join(row),
            source_row_id=str(index),
            detected_at=detected.isoformat(),
        )
        record.update(
            value={
                "event_id": row[0],
                "actor1_name": row[6] or None,
                "actor1_country_code": row[7] or None,
                "actor2_name": row[16] or None,
                "actor2_country_code": row[17] or None,
                "event_code": row[27],
                "event_root_code": row[29],
                "quad_class": row[30],
                "goldstein_scale": _number(row[31]),
                "num_mentions": _number(row[32]),
                "num_sources": _number(row[33]),
                "num_articles": _number(row[34]),
                "average_tone": _number(row[35]),
                "action_country_code": row[53] or None,
                "source_url": row[60] or None,
                "detected_at": detected.isoformat(),
                "first_known_publication_time": None,
                "is_model_output": True,
            },
            **{"quality.value_known": True},
        )
        records.append(record)
    return _finish(records, max_rows)


PARSERS = {
    "tsmc": parse_tsmc_revenue,
    "mops": parse_mops_revenue,
    "twse": parse_twse_daily,
    "opendart": parse_opendart_accounts,
    "eurostat": parse_eurostat,
    "boj": parse_boj,
    "estat": parse_estat,
    "nbs": parse_nbs_release,
    "usgs": parse_usgs_csv,
    "gdelt": parse_gdelt_articles,
    "gdelt_events": parse_gdelt_events,
}


def response_issues(source, body, config):
    issues = []
    if source == "boj" and _json(body).get("NEXTPOSITION") not in (None, "", 0):
        issues.append("provider_continuation_requires_narrower_query")
    if source == "estat":
        info = _json(body)["GET_STATS_DATA"]["STATISTICAL_DATA"].get("RESULT_INF", {})
        if info.get("NEXT_KEY") not in (None, "", 0):
            issues.append("provider_continuation_requires_narrower_query")
    if source == "gdelt" and len(_json(body)["articles"]) >= config.get("max_records", 100):
        issues.append("article_cap_reached_coverage_unknown")
    if source == "gdelt_events":
        issues.append("selected_event_file_full_feed_coverage_unknown")
    return issues
