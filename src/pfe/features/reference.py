"""Sparse causal reference prices, matched-horizon benchmarks and declared exposures."""

from bisect import insort
from collections.abc import Mapping

import numpy as np
import pandas as pd

from pfe.data.contracts import _timestamps
from pfe.data.reference_assets import normalize_reference_assets
from pfe.features._utils import validate_windows

REFERENCE_FEATURE_GROUPS = ("levels", "returns", "relative", "risk", "exposure")


def _decisions(value):
    if isinstance(value, pd.DatetimeIndex):
        return _timestamps(value, "decision_at"), value.copy()
    if isinstance(value, pd.DataFrame) and "decision_at" in value:
        return _timestamps(value.decision_at, "decision_at"), value.index.copy()
    raise ValueError("Decisions require an aware DatetimeIndex or a decision_at DataFrame")


def _ratio(new, old):
    return new / old - 1 if np.isfinite(new) and np.isfinite(old) and old != 0 else np.nan


class _State:
    def __init__(self):
        self.rows, self.periods = {}, []

    def update(self, row):
        period = row["observation_date"]
        if period not in self.rows:
            insort(self.periods, period)
        self.rows[period] = row

    def ordered(self):
        return [self.rows[period] for period in self.periods]


def _number(row):
    if row is None or row["value"] is None or not row["quality.feature_eligible"]:
        return np.nan
    return float(row["value"])


def _metrics(state, windows):
    rows = state.ordered()
    current = rows[-1] if rows else None
    values = np.asarray([_number(row) for row in rows])
    latest = values[-1] if len(values) else np.nan
    prior = values[-2] if len(values) > 1 else np.nan
    metrics = {
        "level": latest,
        "return": _ratio(latest, prior),
        "log_return": np.log(latest / prior) if latest > 0 and prior > 0 else np.nan,
        "point_in_time": float(current["point_in_time"]) if current is not None else np.nan,
        "price_basis_known": float(current["price_adjustment"] != "unknown") if current else np.nan,
    }
    for window in windows:
        sample = values[-window:]
        ready = len(sample) == window and np.isfinite(sample).all()
        metrics[f"mean_w{window}"] = sample.mean() if ready else np.nan
        std = sample.std(ddof=0) if ready else np.nan
        metrics[f"std_w{window}"] = std
        metrics[f"zscore_w{window}"] = (
            (latest - sample.mean()) / std if ready and std > 0 else np.nan
        )
        metrics[f"momentum_w{window}"] = (
            _ratio(latest, values[-window - 1]) if len(values) > window else np.nan
        )
    stamp = current["available_at"].value if current is not None else np.nan
    observed = (
        current["observed_at"].value
        if current is not None and pd.notna(current["observed_at"])
        else np.nan
    )
    return metrics, stamp, observed


def _joint_metrics(target, reference, windows):
    a, b = target.ordered(), reference.ordered()
    result = {
        f"{metric}_w{window}": np.nan
        for window in windows
        for metric in ("relative_strength", "beta", "corr", "sample_count")
    }
    if not a or not b:
        return result, np.nan
    # Returns are dimensionless, but monetary numeraire/calendar/frequency must
    # agree. Exact market closes establish a common horizon, not civil dates.
    compatible = (
        a[-1]["currency"] is not None
        and a[-1]["currency"] == b[-1]["currency"]
        and a[-1]["calendar"] != "unknown"
        and a[-1]["calendar"] == b[-1]["calendar"]
        and a[-1]["frequency"] == b[-1]["frequency"]
    )
    if not compatible:
        return result, np.nan
    lookup = {row["observed_at"]: row for row in b if pd.notna(row["observed_at"])}
    matched_latest = lookup.get(a[-1]["observed_at"])
    matched_stamp = matched_latest["available_at"].value if matched_latest is not None else np.nan
    for window in windows:
        pairs = []
        tail = a[-window - 1 :]
        for old, new in zip(tail, tail[1:], strict=False):
            old_b, new_b = lookup.get(old["observed_at"]), lookup.get(new["observed_at"])
            pairs.append(
                (_ratio(_number(new), _number(old)), _ratio(_number(new_b), _number(old_b)))
            )
        valid = [(x, y) for x, y in pairs if np.isfinite(x) and np.isfinite(y)]
        result[f"sample_count_w{window}"] = float(len(valid))
        if len(pairs) != window or len(valid) != window:
            continue
        x, y = np.asarray(valid).T
        variance = np.var(y)
        covariance = np.mean((x - x.mean()) * (y - y.mean()))
        result[f"beta_w{window}"] = covariance / variance if variance > 0 else np.nan
        scale = np.std(x) * np.std(y)
        result[f"corr_w{window}"] = covariance / scale if scale > 0 else np.nan
        old_a, new_a = a[-window - 1], a[-1]
        old_b, new_b = lookup.get(old_a["observed_at"]), lookup.get(new_a["observed_at"])
        ra = _ratio(_number(new_a), _number(old_a))
        rb = _ratio(_number(new_b), _number(old_b))
        result[f"relative_strength_w{window}"] = (1 + ra) / (1 + rb) - 1 if rb != -1 else np.nan
    return result, matched_stamp


def _align(times, values, decisions):
    if not len(times):
        return np.full(len(decisions), np.nan)
    positions = np.searchsorted(times, decisions, side="right") - 1
    out = np.full(len(decisions), np.nan)
    valid = positions >= 0
    out[valid] = np.asarray(values, dtype=float)[positions[valid]]
    return out


def _weights(value, label):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} requires explicit reference-ID to coefficient mappings")
    for name, weight in value.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not np.isfinite(weight)
        ):
            raise ValueError(f"{label} requires explicit finite coefficients and reference IDs")
    return dict(value)


def compute_reference_features(
    decisions,
    *,
    references,
    target=None,
    target_id=None,
    reference_ids=None,
    windows=(5, 20),
    groups=None,
    names=None,
    max_age=None,
    fx_exposures=None,
    energy_exposures=None,
    reporting_currency=None,
):
    """Compute sparse reference transforms then align backward by availability.

    Levels/moments use distinct economic periods, never repeated forward-filled
    rows. Simultaneous updates are applied as a batch; an old-period revision
    cannot replace the newer period. Windows count periods (returns need w+1).
    Missing/provisional values remain NaN with mask.<feature> outputs.

    Benchmark beta/corr and relative strength require explicit target observations
    and exactly matched observed_at horizons, calendar/frequency and numeraire.
    No same-date join, asynchronous-return pairing or hidden FX conversion.
    Constant/insufficient samples are NaN. max_age expires source levels using
    their actual availability; every operand of joint metrics is age checked.

    Exposures are caller-declared signed coefficients, not inferred company
    relationships. FX effects express the other currency in reporting_currency:
    invert a USD/JPY quote for JPY exposure in USD, keep EUR/USD direct in USD.
    Energy effect multiplies the reference return by the supplied coefficient.
    Missing components stay missing; never become zero. No earnings consensus,
    guidance, causality or stock-return prediction is manufactured.
    """
    decision_times, index = _decisions(decisions)
    now = pd.DatetimeIndex(decision_times).as_unit("ns").asi8
    data = normalize_reference_assets(references)
    windows = validate_windows(windows)
    selected_groups = set(REFERENCE_FEATURE_GROUPS if groups is None else groups)
    if isinstance(groups, str) or selected_groups - set(REFERENCE_FEATURE_GROUPS):
        raise ValueError("Unknown reference feature groups")
    if names is not None and groups is not None:
        raise ValueError("Choose exact names or groups")
    names = (names,) if isinstance(names, str) else tuple(names) if names is not None else None
    identifiers = (
        tuple(sorted(data.instrument_id.unique()))
        if reference_ids is None
        else tuple(reference_ids)
    )
    if (
        isinstance(reference_ids, str)
        or len(set(identifiers)) != len(identifiers)
        or not set(identifiers) <= set(data.instrument_id)
    ):
        raise ValueError("reference_ids must be unique known instrument IDs")
    target_data = None
    if target is not None:
        target_data = normalize_reference_assets(target)
        available_ids = set(target_data.instrument_id)
        if target_id is None and len(available_ids) == 1:
            target_id = next(iter(available_ids))
        if target_id not in available_ids:
            raise ValueError("Select an explicit target_id")
        target_data = target_data.loc[target_data.instrument_id.eq(target_id)]
    elif target_id is not None:
        raise ValueError("target_id requires target observations")
    fx, energy = (
        _weights(fx_exposures, "FX exposures"),
        _weights(energy_exposures, "Energy exposures"),
    )
    if not (set(fx) | set(energy)) <= set(identifiers):
        raise ValueError("Exposure references must be included in reference_ids")
    age_limit = None if max_age is None else pd.Timedelta(max_age)
    if age_limit is not None and (pd.isna(age_limit) or age_limit < pd.Timedelta(0)):
        raise ValueError("max_age must be nonnegative")
    out = {"available_at": pd.DatetimeIndex(decision_times)}
    for instrument in identifiers:
        frame = data.loc[data.instrument_id.eq(instrument)]
        metadata = frame.iloc[0]
        prefix = f"reference[{instrument}]."
        events = frame.to_dict("records")
        if target_data is not None:
            events += [dict(row, _target=True) for row in target_data.to_dict("records")]
        events.sort(key=lambda row: row["available_at"])
        state, target_state = _State(), _State()
        histories, stamps, observed, target_stamps, matched_stamps, times = {}, [], [], [], [], []
        cursor = 0
        while cursor < len(events):
            instant = events[cursor]["available_at"]
            while cursor < len(events) and events[cursor]["available_at"] == instant:
                row = events[cursor]
                (target_state if row.get("_target") else state).update(row)
                cursor += 1
            metrics, stamp, observation = _metrics(state, windows)
            if target_data is not None:
                joint_metrics, matched_stamp = _joint_metrics(target_state, state, windows)
                metrics.update(joint_metrics)
            else:
                matched_stamp = np.nan
                metrics.update(
                    {
                        f"{metric}_w{window}": np.nan
                        for window in windows
                        for metric in ("relative_strength", "beta", "corr", "sample_count")
                    }
                )
            for name, value in metrics.items():
                histories.setdefault(name, []).append(value)
            stamps.append(stamp)
            observed.append(observation)
            matched_stamps.append(matched_stamp)
            target_rows = target_state.ordered()
            target_stamps.append(target_rows[-1]["available_at"].value if target_rows else np.nan)
            times.append(instant.value)
        times = np.asarray(times, dtype=np.int64)
        reference_stamp = _align(times, stamps, now)
        reference_age = (now - reference_stamp) / 1e9
        target_age = (now - _align(times, target_stamps, now)) / 1e9
        matched_age = (now - _align(times, matched_stamps, now)) / 1e9
        expired = (
            np.zeros(len(now), dtype=bool)
            if age_limit is None
            else reference_age > age_limit.total_seconds()
        )
        target_expired = (
            np.zeros(len(now), dtype=bool)
            if age_limit is None
            else target_age > age_limit.total_seconds()
        )
        matched_expired = (
            np.zeros(len(now), dtype=bool)
            if age_limit is None
            else matched_age > age_limit.total_seconds()
        )
        arrays = {name: _align(times, values, now) for name, values in histories.items()}
        arrays["age_seconds"] = reference_age
        arrays["observation_age_seconds"] = (now - _align(times, observed, now)) / 1e9
        arrays["feature_age_seconds"] = (now - _align(times, times, now)) / 1e9
        for name, values in arrays.items():
            family = (
                "relative"
                if name.startswith("relative_strength")
                else "risk"
                if name.startswith(("beta", "corr", "sample_count"))
                else "returns"
                if name in ("return", "log_return") or name.startswith("momentum")
                else "levels"
            )
            if names is not None or family in selected_groups:
                values = values.copy()
                if not name.endswith("age_seconds"):
                    stale = (
                        matched_expired | target_expired
                        if family in ("relative", "risk")
                        else expired
                    )
                    values[stale] = np.nan
                out[prefix + name] = values
        if instrument in fx:
            if metadata.asset_class != "fx" or reporting_currency not in (
                metadata.base_currency,
                metadata.quote_currency,
            ):
                raise ValueError(
                    "FX exposure requires explicit reporting currency within the qualified pair"
                )
            effect = arrays["return"].copy()
            if reporting_currency == metadata.base_currency:
                effect = np.where(effect != -1, 1 / (1 + effect) - 1, np.nan)
            effect *= fx[instrument]
            effect[expired] = np.nan
            if names is not None or "exposure" in selected_groups:
                out[prefix + "fx_effect"] = effect
        if instrument in energy:
            if metadata.asset_class not in ("commodity", "energy"):
                raise ValueError("Energy exposure requires a declared commodity/energy reference")
            if reporting_currency is not None and metadata.currency != reporting_currency:
                raise ValueError(
                    "Energy reference and reporting currency differ; convert explicitly"
                )
            effect = arrays["return"] * energy[instrument]
            effect[expired] = np.nan
            if names is not None or "exposure" in selected_groups:
                out[prefix + "energy_effect"] = effect
    if names is not None:
        if (
            not names
            or len(set(names)) != len(names)
            or not set(names) <= set(out) - {"available_at"}
        ):
            raise ValueError("Unknown, empty or duplicate reference feature names")
        out = {"available_at": out["available_at"], **{name: out[name] for name in names}}
    result = pd.DataFrame(out, index=index).replace([np.inf, -np.inf], np.nan)
    numeric = result.drop(columns="available_at")
    result = pd.concat(
        [result, numeric.notna().rename(columns=lambda name: f"mask.{name}")], axis=1
    )
    result.attrs.update(
        timestamp_semantics="explicit_decision",
        availability="backward lookup on explicit available_at",
        windows=windows,
        window_units="distinct economic periods",
        reference_ids=identifiers,
        matched_horizons="exact observed_at, compatible calendar/frequency/currency",
    )
    return result


def compute_reference_feature(decisions, name, **options):
    """Return one selected output under the same availability/convention rules."""
    return compute_reference_features(decisions, names=(name,), **options)[name]
