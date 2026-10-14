"""Explicit accounting mappings, fiscal calendars and non-overlapping debt totals."""

import hashlib
import json
from dataclasses import asdict, dataclass

import pandas as pd

from pfe.data.contracts import _instant, observations_asof


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class FinancialMapping:
    taxonomy: str
    concept: str
    financial_metric: str
    basis: str
    reference: str
    available_at: str
    eps_basis: str | None = None
    entity_scope: str | None = None
    unit: str | None = None
    context: dict | None = None

    def __post_init__(self):
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.taxonomy, self.concept, self.financial_metric, self.reference)
        ):
            raise ValueError(
                "Financial mappings require exact concepts, roles and proof references"
            )
        if self.basis not in ("us_gaap", "ifrs", "non_gaap") or self.eps_basis not in (
            None,
            "basic",
            "diluted",
        ):
            raise ValueError("Choose explicit accounting/EPS bases")
        if self.financial_metric == "eps" and self.eps_basis is None:
            raise ValueError("EPS mapping requires basic/diluted")
        _instant(self.available_at, "mapping_available_at")
        if self.context is not None and not isinstance(self.context, dict):
            raise ValueError("Financial context must be an explicit object")


def apply_financial_mappings(facts, mappings, *, periods=()):
    """Keep unmapped facts; explicitly map IFRS/custom/segment concepts.

    Period definitions require entity_id/start/end/kind/fiscal_year/label,
    reference and available_at; no fiscal date is inferred from an accession's
    filing year. Mapping knowledge gates the result. Source values are unchanged.
    """
    from pfe.features.fundamentals import _facts

    frame = _facts(facts)
    rules = [FinancialMapping(**rule) if isinstance(rule, dict) else rule for rule in mappings]
    if any(not isinstance(rule, FinancialMapping) for rule in rules):
        raise ValueError("Invalid financial mapping")
    calendar = []
    for definition in periods:
        required = {
            "entity_id",
            "period_start",
            "period_end",
            "period_kind",
            "fiscal_year",
            "label",
            "reference",
            "available_at",
        }
        if not isinstance(definition, dict) or not required <= definition.keys():
            raise ValueError("Fiscal definition misses explicit period/proof fields")
        from datetime import date

        if definition["period_kind"] not in ("annual", "quarter", "ytd", "instant"):
            raise ValueError("Invalid declared fiscal kind")
        start, end = definition["period_start"], definition["period_end"]
        if start is not None and date.fromisoformat(start) > date.fromisoformat(end):
            raise ValueError("Invalid fiscal bounds")
        date.fromisoformat(end)
        _instant(definition["available_at"], "fiscal_available_at")
        calendar.append(definition)
    records = []
    for original in frame.to_dict("records"):
        row = original.copy()
        matches = [
            rule
            for rule in rules
            if rule.taxonomy == row["taxonomy"]
            and rule.concept == row["concept"]
            and (rule.unit is None or rule.unit == row["unit"])
            and (rule.context is None or json.loads(row.get("context_json", "{}")) == rule.context)
        ]
        if len(matches) > 1:
            raise ValueError("Overlapping financial mapping rules")
        definitions = [
            d
            for d in calendar
            if d["entity_id"] == row["entity_id"]
            and d["period_start"] == row["period_start"]
            and d["period_end"] == row["period_end"]
        ]
        if len(definitions) > 1:
            raise ValueError("Ambiguous fiscal period definition")
        references = []
        if matches:
            rule = matches[0]
            row.update(
                financial_metric=rule.financial_metric,
                basis=rule.basis,
                eps_basis=rule.eps_basis,
                mapping_reference=rule.reference,
            )
            if rule.entity_scope is not None:
                row["entity_scope"] = rule.entity_scope
            row["available_at"] = max(
                row["available_at"], _instant(rule.available_at, "mapping_available_at")
            )
            references.append(asdict(rule))
        if definitions:
            definition = definitions[0]
            row.update(
                period_kind=definition["period_kind"],
                declared_fiscal_year=definition["fiscal_year"],
                declared_fiscal_period=definition["label"],
                fiscal_reference=definition["reference"],
            )
            row["quality.period_inferred"] = False
            row["available_at"] = max(
                row["available_at"], _instant(definition["available_at"], "fiscal_available_at")
            )
            references.append(definition)
        if references:
            row["source_revision_id"] = row["revision_id"]
            row["revision_id"] = _digest([row["revision_id"], references])
            row["availability_basis"], row["point_in_time"] = "unknown", False
        records.append(row)
    return _facts(pd.DataFrame(records)) if records else frame


@dataclass(frozen=True)
class DebtRecipe:
    financial_metric: str
    components: tuple[str, ...]
    reference: str
    available_at: str
    components_disjoint: bool

    def __post_init__(self):
        components = tuple(self.components)
        if (
            self.financial_metric not in ("debt", "long_term_debt")
            or len(components) < 2
            or len(set(components)) != len(components)
        ):
            raise ValueError("Debt recipe needs a total and distinct component roles")
        if (
            self.components_disjoint is not True
            or not isinstance(self.reference, str)
            or not self.reference.strip()
        ):
            raise ValueError("Debt components require an explicit disjointness proof reference")
        if self.financial_metric in components or (
            "long_term_debt" in components
            and set(components) & {"long_term_debt_current", "long_term_debt_noncurrent"}
        ):
            raise ValueError("Debt recipe contains overlapping totals/components")
        object.__setattr__(self, "components", components)
        _instant(self.available_at, "recipe_available_at")


def derive_debt_totals(facts, recipe):
    """Replay known components; missing/incompatible/ambiguous legs produce null.

    Direct totals take precedence even when explicitly withdrawn. No silent sum
    across source, filing, currency, entity/segment context or economic end date.
    A disjointness declaration remains a caller assertion, not a data certification.
    """
    from pfe.features.fundamentals import _facts

    recipe = DebtRecipe(**recipe) if isinstance(recipe, dict) else recipe
    if not isinstance(recipe, DebtRecipe):
        raise ValueError("Choose an explicit DebtRecipe")
    frame = _facts(facts)
    relevant = frame.loc[
        frame.period_kind.eq("instant")
        & frame.statement_type.eq("reported_actual")
        & frame.financial_metric.isin([recipe.financial_metric, *recipe.components])
    ].copy()
    if relevant.empty:
        return frame
    if "context_json" not in relevant:
        relevant["context_json"] = "{}"
    groups = [
        "entity_id",
        "source_id",
        "basis",
        "unit",
        "currency",
        "period_end",
        "entity_scope",
        "context_json",
    ]
    outputs = []
    for _, history in relevant.groupby(groups, dropna=False, sort=False):
        prior = None
        for epoch in sorted(
            set(history.available_at) | {_instant(recipe.available_at, "recipe_available_at")}
        ):
            if epoch < _instant(recipe.available_at, "recipe_available_at"):
                continue
            known = observations_asof(history, epoch)
            if known.empty or known.financial_metric.eq(recipe.financial_metric).any():
                continue
            legs = [known.loc[known.financial_metric.eq(name)] for name in recipe.components]
            compatible = all(len(leg) == 1 and leg.value.iloc[0] is not None for leg in legs)
            if compatible:
                compatible = pd.concat(legs).source_document_id.nunique(dropna=False) == 1
            value = sum(float(leg.value.iloc[0]) for leg in legs) if compatible else None
            lineage = [sorted(leg.revision_id.dropna().tolist()) for leg in legs]
            identity = _digest([asdict(recipe), lineage, value])
            if identity == prior:
                continue
            prior = identity
            row = known.iloc[-1].to_dict()
            row.update(
                value=value,
                financial_metric=recipe.financial_metric,
                metric="derived.debt:" + _digest([asdict(recipe), [row[k] for k in groups]])[:24],
                taxonomy="derived",
                concept="DisjointDebtTotal",
                revision_id=identity,
                available_at=epoch,
                availability_basis="unknown",
                point_in_time=False,
                derived_total=True,
                recipe_reference=recipe.reference,
                component_revision_ids=json.dumps(lineage),
                coverage_complete=False,
            )
            outputs.append(row)
    return (
        _facts(pd.concat([frame, pd.DataFrame(outputs)], ignore_index=True)) if outputs else frame
    )
