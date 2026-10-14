"""Known-at fundamental levels/ratios; no forecast, consensus or silent FX."""

from datetime import date

import numpy as np
import pandas as pd

from pfe.data.contracts import (
    _identifiers,
    _missing,
    _timestamps,
    normalize_observations,
)

_LEVELS = (
    "revenue",
    "eps",
    "net_income",
    "operating_income",
    "gross_profit",
    "assets",
    "liabilities",
    "equity",
    "cash",
    "debt",
    "long_term_debt",
    "operating_cash_flow",
    "capex",
)
_STOCKS = {"assets", "liabilities", "equity", "cash", "debt", "long_term_debt"}
_ADDITIVE = {
    "revenue",
    "net_income",
    "operating_income",
    "gross_profit",
    "operating_cash_flow",
    "capex",
}
_RATIOS = {
    "gross_margin": ("gross_profit", "revenue"),
    "operating_margin": ("operating_income", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "liabilities_to_assets": ("liabilities", "assets"),
    "debt_to_assets": ("debt", "assets"),
    "debt_to_equity": ("debt", "equity"),
    "long_term_debt_to_assets": ("long_term_debt", "assets"),
    "capex_intensity": ("capex", "revenue"),
    "cash_flow_to_revenue": ("operating_cash_flow", "revenue"),
}
_EXTRAS = (
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "free_cash_flow",
    "net_debt",
    "age_seconds",
    "period_age_days",
    "point_in_time",
    "coverage_fraction",
    "derived_quarter",
    "free_cash_flow_derived",
    "quality.ambiguous",
    "quality.incomplete",
)


def _facts(frame):
    required = {
        "financial_metric",
        "concept",
        "taxonomy",
        "period_start",
        "period_end",
        "period_kind",
        "basis",
        "eps_basis",
        "entity_scope",
        "currency",
        "statement_type",
    }
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame):
        raise ValueError(f"Fundamental facts require {sorted(required)}")
    out = normalize_observations(frame)
    _identifiers(out, ("concept", "taxonomy", "basis", "entity_scope"))
    _identifiers(
        out,
        ("financial_metric", "eps_basis", "currency"),
        nullable=("financial_metric", "eps_basis", "currency"),
    )
    if not out.statement_type.isin(
        ("reported_actual", "management_guidance", "analyst_estimate")
    ).all():
        raise ValueError("Invalid fundamental statement_type")
    if not out.period_kind.isin(("annual", "quarter", "ytd", "instant", "unknown")).all():
        raise ValueError("Invalid fundamental period_kind")
    if not out.basis.isin(("us_gaap", "ifrs", "non_gaap", "unknown")).all():
        raise ValueError("Invalid fundamental accounting basis")
    if not out.eps_basis.isin((None, "basic", "diluted")).all():
        raise ValueError("eps_basis must be basic, diluted or null")
    out["period_start"] = pd.Series(
        [None if _missing(value) else value for value in out.period_start], dtype=object
    )
    for column in ("period_start", "period_end"):
        for value in out[column].dropna():
            if not isinstance(value, str):
                raise ValueError("Fundamental periods require ISO economic dates")
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except ValueError as error:
                raise ValueError("Fundamental periods require ISO economic dates") from error
    if out.period_end.isna().any() or out.observation_date.ne(out.period_end).any():
        raise ValueError("Fundamental observation_date must equal period_end")
    if (out.period_start.notna() & (out.period_start > out.period_end)).any():
        raise ValueError("Fundamental period_start cannot follow period_end")
    if out.value.map(
        lambda value: (
            isinstance(value, (dict, list, bool))
            or value is not None
            and not isinstance(value, (int, float))
        )
    ).any():
        raise ValueError("Fundamental values must be numeric or missing")
    return out


def _compatible(*records):
    if any(record is None for record in records):
        return False
    fields = (
        "unit",
        "currency",
        "basis",
        "period_start",
        "period_end",
        "period_kind",
        "entity_scope",
        "source_id",
    )
    return all(len({record[field] for record in records}) == 1 for field in fields)


def _quarterly(known):
    """Difference additive YTD flows only: weighted-average EPS is not additive."""
    additions = []
    keys = [
        "entity_id",
        "financial_metric",
        "concept",
        "taxonomy",
        "unit",
        "currency",
        "basis",
        "entity_scope",
        "source_id",
        "period_start",
    ]
    candidates = known.loc[
        known.financial_metric.isin(_ADDITIVE)
        & known.period_start.notna()
        & known.period_kind.isin(("ytd", "quarter", "annual"))
    ]
    for _, group in candidates.groupby(keys, dropna=False, sort=False):
        group = group.sort_values("period_end", kind="stable")
        for _, current in group.loc[group.period_kind.isin(("ytd", "annual"))].iterrows():
            previous = group.loc[group.period_end < current.period_end]
            if previous.empty:
                continue
            previous = previous.loc[previous.period_end == previous.period_end.max()]
            if len(previous) != 1:
                continue
            prior = previous.iloc[0]
            start = (date.fromisoformat(prior.period_end) + pd.Timedelta("1D")).isoformat()
            duration = (date.fromisoformat(current.period_end) - date.fromisoformat(start)).days + 1
            if not 60 <= duration <= 120 or current.value is None or prior.value is None:
                continue
            derived = current.copy()
            derived["value"] = current.value - prior.value
            derived["period_start"], derived["period_kind"] = start, "quarter"
            derived["available_at"] = max(current.available_at, prior.available_at)
            derived["point_in_time"] = bool(current.point_in_time and prior.point_in_time)
            derived["derived_quarter"] = True
            derived["derivation"] = (
                "annual_minus_previous_ytd"
                if current.period_kind == "annual"
                else "current_ytd_minus_previous_ytd"
            )
            additions.append(derived)
    out = known.copy()
    out["derived_quarter"] = False
    if additions:
        out = pd.concat([out, pd.DataFrame(additions)], ignore_index=True)
    return out


def _choose(known, role, end, kind, concepts):
    candidates = known.loc[known.financial_metric == role]
    if role in concepts:
        candidates = candidates.loc[
            (candidates.taxonomy + ":" + candidates.concept) == concepts[role]
        ]
    if role in _STOCKS:
        candidates = candidates.loc[candidates.period_kind == "instant"]
    else:
        candidates = candidates.loc[candidates.period_kind == kind]
    if end is not None:
        candidates = candidates.loc[candidates.period_end == end]
    if candidates.empty:
        return None, False
    # Prefer direct quarterly facts over a subtraction for the same role/period.
    if not candidates.loc[~candidates.derived_quarter].empty:
        candidates = candidates.loc[~candidates.derived_quarter]
    if "derived_total" in candidates:
        direct = candidates.loc[~candidates.derived_total.eq(True)]
        if not direct.empty:
            candidates = direct
    if len(candidates) != 1:
        return None, True
    return candidates.iloc[0], False


def _growth(known, current, kind):
    if current is None or current.value is None or current.period_start is None:
        return np.nan
    end, start = date.fromisoformat(current.period_end), date.fromisoformat(current.period_start)
    previous = known.loc[
        (known.financial_metric == current.financial_metric)
        & (known.concept == current.concept)
        & (known.taxonomy == current.taxonomy)
        & (known.unit == current.unit)
        & (known.currency == current.currency)
        & (known.basis == current.basis)
        & (known.source_id == current.source_id)
        & known.eps_basis.fillna("").eq(current.eps_basis or "")
        & (known.period_kind == kind)
        & (known.period_end < current.period_end)
        & known.period_start.notna()
    ]
    if previous.empty:
        return np.nan
    distances = previous.period_end.map(
        lambda value: abs((end - date.fromisoformat(value)).days - 365)
    )
    duration = (end - start).days
    same_span = previous.apply(
        lambda row: (
            abs(
                (date.fromisoformat(row.period_end) - date.fromisoformat(row.period_start)).days
                - duration
            )
            <= 15
        ),
        axis=1,
    )
    previous = previous.loc[distances.le(15) & same_span]
    if previous.empty:
        return np.nan
    previous = previous.loc[previous.period_end == previous.period_end.max()]
    direct = previous.loc[~previous.derived_quarter]
    if not direct.empty:
        previous = direct
    if len(previous) != 1 or previous.value.iloc[0] in (None, 0):
        return np.nan
    return (current.value - previous.value.iloc[0]) / abs(previous.value.iloc[0])


def compute_fundamental_features(
    decisions,
    *,
    facts,
    entity_id=None,
    period_kind="quarter",
    basis="us_gaap",
    eps_basis="diluted",
    currency=None,
    source_id=None,
    entity_scope="entity",
    context=None,
    concepts=None,
    derive_quarters=True,
    names=None,
    max_age=None,
):
    """Align actual fundamentals at aware decisions, with sparse vintage replay.

    decisions is an aware DatetimeIndex (one entity) or DataFrame with decision_at
    and entity_id. Every metric uses only available_at <= decision. Periods,
    GAAP/IFRS/non-GAAP and basic/diluted EPS remain separate. Currency/source and
    concept ambiguity require an explicit selection or yield an ambiguity mask;
    no conversion or automatic revenue-tag substitution. Stock/flow ratios match
    economic end dates; flow ratios additionally match start/unit/basis/source.

    Quarter derivation subtracts compatible known additive YTD flows sharing
    fiscal start/concept/unit/basis/source; never subtracts weighted-average EPS.
    Q4 may use annual minus nine-month YTD with the same 60–120 day span check.
    FCF = operating cash flow - capex (positive cash-outflow convention), explicitly
    labelled derived. Long-term debt is not silently relabelled total debt.
    YoY growth matches exact concept/base and approximately year-separated equal
    spans (15-day tolerance accommodates 52/53-week fiscal periods); denominator
    is abs(previous), with zero/missing producing NaN. Duration classifications
    from Company Facts remain a disclosed heuristic, not verified fiscal labels.
    max_age masks the selected current-period inputs, without falling back to an
    older period. Historical comparatives used for YoY stay usable when known.
    Outputs include UTC available_at=decision_at and mask.<feature> columns.
    """
    data = _facts(facts)
    if entity_scope not in ("entity", "segment", "unknown"):
        raise ValueError("Select an explicit financial entity scope")
    if entity_scope == "segment" and not context:
        raise ValueError("Segment features require an exact context")
    if context is not None:
        import json

        if not isinstance(context, dict) or "context_json" not in data:
            raise ValueError("Financial context requires a mapping and context_json facts")
        data = data.loc[data.context_json.map(lambda value: json.loads(value) == context)]
    if (
        period_kind not in ("annual", "quarter", "ytd")
        or basis not in ("us_gaap", "ifrs", "non_gaap")
        or eps_basis not in ("basic", "diluted")
    ):
        raise ValueError("Choose explicit duration kind, accounting basis and EPS basis")
    if not isinstance(derive_quarters, bool):
        raise ValueError("derive_quarters requires a boolean")
    concepts = {} if concepts is None else concepts
    if not isinstance(concepts, dict) or any(
        role not in _LEVELS or not isinstance(tag, str) or ":" not in tag
        for role, tag in concepts.items()
    ):
        raise ValueError("concepts maps financial roles to explicit taxonomy:concept names")
    age = None if max_age is None else pd.Timedelta(max_age)
    if age is not None and (pd.isna(age) or age < pd.Timedelta(0)):
        raise ValueError("max_age requires a non-negative fixed duration")
    if isinstance(decisions, pd.DatetimeIndex):
        if (
            decisions.tz is None
            or decisions.hasnans
            or not decisions.is_unique
            or not decisions.is_monotonic_increasing
        ):
            raise ValueError("decisions require a sorted unique aware DatetimeIndex")
        entities = data.entity_id.unique()
        entity = entity_id if entity_id is not None else entities[0] if len(entities) == 1 else None
        base = pd.DataFrame({"decision_at": decisions, "entity_id": entity})
        result_index = decisions
    elif isinstance(decisions, pd.DataFrame) and {"decision_at", "entity_id"}.issubset(decisions):
        base = decisions.copy().reset_index(drop=True)
        if entity_id is not None and not base.entity_id.eq(entity_id).all():
            raise ValueError("entity_id argument differs from decision entities")
        result_index = decisions.index
    else:
        raise ValueError("decisions require an aware index or decision_at/entity_id columns")
    base["decision_at"] = _timestamps(base.decision_at, "decision_at")
    _identifiers(base, ("entity_id",))
    if base.duplicated(["entity_id", "decision_at"]).any():
        raise ValueError("Duplicate entity/decision")
    catalog = (*_LEVELS, *_RATIOS, *_EXTRAS, *(f"{role}_age_seconds" for role in _LEVELS))
    catalog = tuple(f"fundamentals.{name}" for name in catalog)
    selected = catalog if names is None else (names,) if isinstance(names, str) else tuple(names)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(catalog):
        raise ValueError("Unknown or duplicate fundamental feature names")
    output = []
    for decision in base.itertuples(index=False):
        now = decision.decision_at
        # Input was validated once above. Resolve vintages on that canonical
        # frame without re-cloning every fact for every daily decision.
        keys = ["entity_id", "metric", "observation_date", "source_id", "instrument_id"]
        known = data.loc[
            (data.entity_id == decision.entity_id) & (data.available_at <= now)
        ].drop_duplicates(keys, keep="last")
        known = known.loc[
            (known.basis == basis)
            & (known.entity_scope == entity_scope)
            & (known.statement_type == "reported_actual")
            & (known.period_end <= now.date().isoformat())
            & ((known.financial_metric != "eps") | (known.eps_basis == eps_basis))
        ]
        if currency is not None:
            known = known.loc[known.currency.eq(currency)]
        if source_id is not None:
            known = known.loc[known.source_id.eq(source_id)]
        known = (
            _quarterly(known)
            if derive_quarters and period_kind == "quarter"
            else known.assign(derived_quarter=False)
        )
        duration = known.loc[known.period_kind == period_kind]
        target = (
            duration.period_end.max()
            if not duration.empty
            else known.period_end.max()
            if not known.empty
            else None
        )
        levels, ambiguous = {}, False
        for role in _LEVELS:
            levels[role], conflict = _choose(known, role, target, period_kind, concepts)
            if (
                levels[role] is not None
                and age is not None
                and now - levels[role].available_at > age
            ):
                levels[role] = None
            ambiguous |= conflict
        row = {name: np.nan for name in catalog}
        for role, record in levels.items():
            if record is not None:
                row[f"fundamentals.{role}"] = np.nan if record.value is None else record.value
                row[f"fundamentals.{role}_age_seconds"] = (
                    now - record.available_at
                ).total_seconds()
        for metric, (left, right) in _RATIOS.items():
            a, b = levels[left], levels[right]
            if _compatible(a, b) and a.value is not None and b.value not in (None, 0):
                row[f"fundamentals.{metric}"] = a.value / b.value
        for role in ("revenue", "eps"):
            row[f"fundamentals.{role}_growth_yoy"] = _growth(known, levels[role], period_kind)
        cash, capex = levels["operating_cash_flow"], levels["capex"]
        if _compatible(cash, capex) and cash.value is not None and capex.value is not None:
            row["fundamentals.free_cash_flow"] = cash.value - capex.value
            row["fundamentals.free_cash_flow_derived"] = 1.0
        debt, cash = levels["debt"], levels["cash"]
        if _compatible(debt, cash) and debt.value is not None and cash.value is not None:
            row["fundamentals.net_debt"] = debt.value - cash.value
        used = [record for record in levels.values() if record is not None]
        row["fundamentals.quality.ambiguous"] = float(ambiguous)
        row["fundamentals.quality.incomplete"] = 1.0
        row["fundamentals.coverage_fraction"] = sum(
            record is not None and record.value is not None for record in levels.values()
        ) / len(_LEVELS)
        if used:
            row["fundamentals.age_seconds"] = (
                now - max(record.available_at for record in used)
            ).total_seconds()
            row["fundamentals.period_age_days"] = (now.date() - date.fromisoformat(target)).days
            row["fundamentals.point_in_time"] = float(all(record.point_in_time for record in used))
            row["fundamentals.derived_quarter"] = float(
                any(record.derived_quarter for record in used)
            )
        output.append({name: row[name] for name in selected})
    result = pd.concat([base, pd.DataFrame(output, columns=selected)], axis=1)
    result.loc[:, selected] = result.loc[:, selected].replace([np.inf, -np.inf], np.nan)
    result["available_at"] = result.decision_at
    masks = result.loc[:, selected].notna().rename(columns=lambda name: f"mask.{name}")
    result = pd.concat([result, masks], axis=1)
    result.index = result_index
    result.attrs.update(
        timestamp_semantics="explicit_decision",
        availability="explicit decision_at",
        ratio_units="decimal",
        coverage="fraction of defined level inputs, not full company coverage",
        fcf_definition="derived operating_cash_flow minus positive capex outflow",
    )
    return result


def compute_fundamental_feature(decisions, name, **options):
    """Compute one selected output using the same temporal/compatibility rules."""
    return compute_fundamental_features(decisions, names=(name,), **options)[name]
