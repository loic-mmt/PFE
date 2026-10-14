"""Selected option-chain summaries and rolling observed prints, aligned causally."""

import numpy as np
import pandas as pd

from pfe.data.options import (
    GREEKS,
    normalize_option_candles,
    normalize_option_chain,
    normalize_option_flow,
)
from pfe.features.aggregation import fixed_duration
from pfe.features.macro import _align, _ns, _selection

OPTION_FEATURE_GROUPS = (
    "activity",
    "volatility",
    "greeks",
    "positioning",
    "flow",
    "contract",
    "quality",
)
_ACTIVITY = (
    "call_volume",
    "put_volume",
    "total_volume",
    "put_call_volume_ratio",
    "volume_imbalance",
    "call_premium",
    "put_premium",
    "total_premium",
    "put_call_premium_ratio",
    "premium_imbalance",
)
_VOLATILITY = (
    "iv_mean",
    "iv_std",
    "iv_volume_weighted",
    "atm_call_iv",
    "atm_put_iv",
    "atm_iv",
    "put_call_atm_iv_spread",
    "risk_reversal",
    "butterfly",
    "term_spread",
    "term_slope",
    "expected_move_relative",
)
_POSITIONING = (
    "call_open_interest",
    "put_open_interest",
    "put_call_open_interest_ratio",
    "oi_concentration",
    "call_oi_peak_strike",
    "put_oi_peak_strike",
    "oi_weighted_strike",
    "unsigned_gamma_oi",
)
_QUALITY = (
    "contract_count",
    "active_contract_count",
    "expired_fraction",
    "iv_coverage",
    "open_interest_coverage",
    "quote_coverage",
    "point_in_time",
    "limit_reached",
    "snapshot_age_seconds",
    "oldest_trade_age_seconds",
)
_FLOW = (
    *_ACTIVITY,
    "print_count",
    "average_premium",
    "iv_volume_weighted",
    "delta_contracts",
    "point_in_time_fraction",
)
_CONTRACT = (
    "log_return",
    "intrabar_log_return",
    "range_relative",
    "iv",
    *GREEKS,
    "volume",
    "premium",
    "print_count",
    "age_seconds",
)
_CHUNK = 262144


def _ratio(left, right):
    return left / right if np.isfinite(left) and np.isfinite(right) and right != 0 else np.nan


def _sum(values):
    values = np.asarray(values, dtype=float)
    return values.sum() if np.isfinite(values).all() else np.nan


def _weighted(frame, field, weight):
    values, weights = frame[field].to_numpy(float), frame[weight].to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(weights).all():
        return np.nan
    return _ratio(np.dot(values, weights), weights.sum())


def _activity(frame, metric):
    field = "volume" if "volume" in metric else "premium"
    if metric.startswith("call_"):
        return _sum(frame.loc[frame.contract_type == "call", field])
    if metric.startswith("put_") and not metric.startswith("put_call_"):
        return _sum(frame.loc[frame.contract_type == "put", field])
    if metric.startswith("total_"):
        return _sum(frame[field])
    call = _sum(frame.loc[frame.contract_type == "call", field])
    put = _sum(frame.loc[frame.contract_type == "put", field])
    return _ratio(put, call) if metric.startswith("put_call_") else _ratio(call - put, call + put)


class _Surface:
    """Lazy surface queries on one observed snapshot; no pricing-model reconstruction."""

    def __init__(self, frame, *, atm_tolerance, delta_target, delta_tolerance):
        self.frame = frame
        self.atm_tolerance = atm_tolerance
        self.delta_target = delta_target
        self.delta_tolerance = delta_tolerance
        self.cache = {}

    def atm(self, expiry, side):
        key = (expiry, side)
        if key not in self.cache:
            frame = self.frame.loc[
                (self.frame.expiry == expiry)
                & (self.frame.contract_type == side)
                & self.frame.iv.notna()
                & self.frame.underlying_price.notna()
            ]
            if frame.empty:
                self.cache[key] = np.nan
            else:
                distance = (frame.strike / frame.underlying_price - 1).abs()
                i = distance.idxmin()
                self.cache[key] = (
                    float(frame.at[i, "iv"]) if distance.loc[i] <= self.atm_tolerance else np.nan
                )
        return self.cache[key]

    def atm_mean(self, expiry):
        values = np.array([self.atm(expiry, side) for side in ("call", "put")])
        return values[np.isfinite(values)].mean() if np.isfinite(values).any() else np.nan

    def wing(self, expiry, side):
        target = self.delta_target if side == "call" else -self.delta_target
        frame = self.frame.loc[
            (self.frame.expiry == expiry)
            & (self.frame.contract_type == side)
            & self.frame.iv.notna()
            & self.frame.delta.notna()
        ]
        if frame.empty:
            return np.nan
        distance = (frame.delta - target).abs()
        i = distance.idxmin()
        return float(frame.at[i, "iv"]) if distance.loc[i] <= self.delta_tolerance else np.nan


def _volatility(frame, metric, surface, snapshot):
    if metric in ("iv_mean", "iv_std"):
        values = frame.iv.dropna().to_numpy(float)
        if not len(values):
            return np.nan
        return values.mean() if metric == "iv_mean" else values.std(ddof=0)
    if metric == "iv_volume_weighted":
        return _weighted(frame, "iv", "volume")
    expiries = sorted(frame.expiry.unique())
    if not expiries:
        return np.nan
    front = expiries[0]
    if metric in ("atm_call_iv", "atm_put_iv"):
        return surface.atm(front, "call" if metric == "atm_call_iv" else "put")
    if metric == "put_call_atm_iv_spread":
        return surface.atm(front, "put") - surface.atm(front, "call")
    if metric == "atm_iv":
        return surface.atm_mean(front)
    if metric in ("risk_reversal", "butterfly"):
        call, put = surface.wing(front, "call"), surface.wing(front, "put")
        return (
            call - put if metric == "risk_reversal" else (call + put) / 2 - surface.atm_mean(front)
        )
    if metric == "expected_move_relative":
        days = (front - snapshot.normalize()).days
        return surface.atm_mean(front) * np.sqrt(days / 365) if days > 0 else np.nan
    if len(expiries) < 2:
        return np.nan
    spread = surface.atm_mean(expiries[-1]) - surface.atm_mean(front)
    return spread if metric == "term_spread" else spread / (expiries[-1] - front).days


def _positioning(frame, metric):
    if metric.startswith("call_") or metric.startswith("put_"):
        if metric == "put_call_open_interest_ratio":
            call = _sum(frame.loc[frame.contract_type == "call", "open_interest"])
            put = _sum(frame.loc[frame.contract_type == "put", "open_interest"])
            return _ratio(put, call)
        side = "call" if metric.startswith("call_") else "put"
        subset = frame.loc[frame.contract_type == side]
        if metric.endswith("peak_strike"):
            if subset.empty or subset.open_interest.isna().any() or subset.open_interest.sum() == 0:
                return np.nan
            grouped = subset.groupby("strike").open_interest.sum()
            return float(grouped.idxmax())
        return _sum(subset.open_interest)
    if frame.empty or frame.open_interest.isna().any():
        return np.nan
    if metric == "oi_weighted_strike":
        return _weighted(frame, "strike", "open_interest")
    if metric == "unsigned_gamma_oi":
        return _sum(frame.gamma.to_numpy(float) * frame.open_interest.to_numpy(float))
    totals = frame.groupby("strike").open_interest.sum().to_numpy(float)
    return np.sum((totals / totals.sum()) ** 2) if totals.sum() > 0 else np.nan


def _snapshot_metric(raw, active, snapshot, group, metric, surface):
    if group == "activity":
        return _activity(active, metric)
    if group == "volatility":
        return _volatility(active, metric, surface, snapshot)
    if group == "greeks":
        field = metric.removesuffix("_mean").removesuffix("_volume_weighted")
        if metric.endswith("_volume_weighted"):
            return _weighted(active, field, "volume")
        values = active[field].dropna().to_numpy(float)
        return values.mean() if len(values) else np.nan
    if group == "positioning":
        return _positioning(active, metric)
    if metric == "contract_count":
        return len(raw)
    if metric == "active_contract_count":
        return len(active)
    if metric == "expired_fraction":
        return (raw.expiry < snapshot.normalize()).mean()
    if metric == "point_in_time":
        return float(raw.point_in_time.all())
    if metric == "limit_reached":
        return float(raw.attrs.get("limit_reached", False))
    if metric == "snapshot_age_seconds":
        return snapshot.value
    if metric == "oldest_trade_age_seconds":
        time = raw.quote_at.min()
        return time.value if pd.notna(time) else np.nan
    if not len(active):
        return np.nan
    if metric == "quote_coverage":
        return (active.bid.notna() & active.ask.notna()).mean()
    return (
        active[
            {
                "iv_coverage": "iv",
                "open_interest_coverage": "open_interest",
            }[metric]
        ]
        .notna()
        .mean()
    )


def _alive(times, deadlines, decisions):
    out = np.zeros(len(decisions), dtype=bool)
    for left in range(0, len(out), _CHUNK):
        right = min(left + _CHUNK, len(out))
        positions = np.searchsorted(times, decisions[left:right], side="right") - 1
        valid = positions >= 0
        out[left:right][valid] = decisions[left:right][valid] < deadlines[positions[valid]]
    return out


def _interval_sum(starts, ends, weights, decisions):
    """Known contributions exist on [available_at, event_at + window), never backfilled."""
    valid = starts < ends
    starts, ends, weights = starts[valid], ends[valid], weights[valid]
    first, last = np.argsort(starts, kind="stable"), np.argsort(ends, kind="stable")
    cumulative_first = np.r_[0.0, np.cumsum(weights[first])]
    cumulative_last = np.r_[0.0, np.cumsum(weights[last])]
    starts, ends = starts[first], ends[last]
    out = np.empty(len(decisions), dtype=float)
    for left in range(0, len(out), _CHUNK):
        right = min(left + _CHUNK, len(out))
        now = decisions[left:right]
        out[left:right] = (
            cumulative_first[np.searchsorted(starts, now, side="right")]
            - cumulative_last[np.searchsorted(ends, now, side="right")]
        )
    return out


def _flow_values(frame, decisions, window, metrics, source):
    observed = (
        _ns(frame.trade_at)
        if source == "prints"
        else _ns(frame.bar_at + pd.Timedelta(frame.attrs["source_frequency"]))
    )
    start, end = _ns(frame.available_at), observed + window
    cache = {}

    def aggregate(field, side=None, weighted_field=None):
        key = (field, side, weighted_field)
        if key not in cache:
            mask = np.ones(len(frame), dtype=bool)
            if side:
                mask &= frame.contract_type.eq(side).to_numpy()
            values = np.ones(len(frame)) if field == "count" else frame[field].to_numpy(float)
            if weighted_field:
                values = values * frame[weighted_field].to_numpy(float)
            invalid = ~np.isfinite(values)
            result = _interval_sum(
                start[mask], end[mask], np.where(invalid, 0.0, values)[mask], decisions
            )
            if invalid[mask].any():
                missing = _interval_sum(
                    start[mask], end[mask], invalid[mask].astype(float), decisions
                )
                result[missing > 0.5] = np.nan
            cache[key] = result
        return cache[key]

    def divide(left, right):
        result = np.full(len(decisions), np.nan)
        np.divide(left, right, out=result, where=right != 0)
        return result

    output = {}
    for metric in metrics:
        if metric in _ACTIVITY:
            field = "volume" if "volume" in metric else "premium"
            if metric.startswith("call_"):
                result = aggregate(field, "call")
            elif metric.startswith("put_") and not metric.startswith("put_call_"):
                result = aggregate(field, "put")
            elif metric.startswith("total_"):
                result = aggregate(field)
            else:
                call, put = aggregate(field, "call"), aggregate(field, "put")
                result = (
                    divide(put, call)
                    if metric.startswith("put_call_")
                    else divide(call - put, call + put)
                )
        elif metric == "print_count":
            result = aggregate("count" if source == "prints" else "print_count")
        elif metric == "average_premium":
            result = divide(
                aggregate("premium"), aggregate("count" if source == "prints" else "print_count")
            )
        elif metric == "iv_volume_weighted":
            result = divide(aggregate("iv", weighted_field="volume"), aggregate("volume"))
        elif metric == "delta_contracts":
            result = aggregate("delta", weighted_field="volume")
        else:
            result = divide(aggregate("point_in_time"), aggregate("count"))
        output[metric] = result
    return output


def compute_option_features(
    bars,
    *,
    chain=None,
    flow=None,
    candles=None,
    underlying=None,
    names=None,
    groups=None,
    source_frequency="1min",
    flow_window="30min",
    flow_source=None,
    min_dte=0,
    max_dte=90,
    atm_tolerance=0.03,
    delta_target=0.25,
    delta_tolerance=0.10,
    max_age=None,
):
    """Compute selected observed option features at market bar close.

    Never combine print and candle flow: choose flow_source='prints'/'candles'.
    Chain metrics expire at the snapshot's next UTC midnight, since DTE/expiry
    selection changes daily. Quality ages remain visible. Ratios and aggregates
    describe the supplied subset, not an inferred complete chain or dealer book.
    """
    index = bars.index if isinstance(bars, pd.DataFrame) else bars
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("bars require a sorted unique timezone-aware DatetimeIndex")
    step = fixed_duration(source_frequency, "source_frequency")
    window = fixed_duration(flow_window, "flow_window").value
    age_limit = fixed_duration(max_age, "max_age").value if max_age is not None else None
    if (
        any(isinstance(x, bool) or not isinstance(x, int) for x in (min_dte, max_dte))
        or not 0 <= min_dte <= max_dte
    ):
        raise ValueError("Require integer 0 <= min_dte <= max_dte")
    for value, label in ((atm_tolerance, "atm_tolerance"), (delta_tolerance, "delta_tolerance")):
        if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{label} must lie in [0, 1]")
    if isinstance(delta_target, bool) or not np.isfinite(delta_target) or not 0 < delta_target < 1:
        raise ValueError("delta_target must lie in (0, 1)")
    if names is not None and groups is not None:
        raise ValueError("Choose names or groups, not both")
    if not any(frame is not None for frame in (chain, flow, candles)):
        raise ValueError("Provide chain, flow and/or candles")
    if names is not None:
        requested = _selection(names, "names")
        needed_groups = {
            "contract" if name.startswith("options.contract[") else name.split(".")[1]
            for name in requested
            if isinstance(name, str) and name.startswith("options.") and len(name.split(".")) > 1
        }
    else:
        chosen_groups = OPTION_FEATURE_GROUPS if groups is None else _selection(groups, "groups")
        needed_groups = set(chosen_groups)
    source = flow_source or ("prints" if flow is not None else "candles")
    if source not in ("prints", "candles"):
        raise ValueError("flow_source must be prints or candles")
    need_chain = bool(
        needed_groups & {"activity", "volatility", "greeks", "positioning", "quality"}
    )
    frames = {
        "chain": normalize_option_chain(chain) if chain is not None and need_chain else None,
        "prints": normalize_option_flow(flow)
        if flow is not None and "flow" in needed_groups and source == "prints"
        else None,
        "candles": normalize_option_candles(
            candles,
            source_frequency=(
                candles.attrs.get("source_frequency", "1min") if candles is not None else "1min"
            ),
        )
        if candles is not None
        and ("contract" in needed_groups or ("flow" in needed_groups and source == "candles"))
        else None,
    }
    symbols = set().union(
        *(set(frame.underlying) for frame in frames.values() if frame is not None)
    )
    if underlying is None:
        if len(symbols) > 1:
            raise ValueError("Specify underlying for multi-instrument option data")
        underlying = next(iter(symbols), None)
    elif not isinstance(underlying, str) or not underlying.strip():
        raise ValueError("underlying must be a ticker")
    else:
        underlying = underlying.strip().upper()
        if symbols and underlying not in symbols:
            raise ValueError("underlying is absent from option inputs")
    for key, frame in frames.items():
        if frame is not None and underlying is not None:
            frames[key] = frame.loc[frame.underlying == underlying]
    catalog = {}
    if frames["chain"] is not None:
        for group, metrics in (
            ("activity", _ACTIVITY),
            ("volatility", _VOLATILITY),
            ("positioning", _POSITIONING),
            ("quality", _QUALITY),
            ("greeks", tuple(f"{g}_{m}" for g in GREEKS for m in ("mean", "volume_weighted"))),
        ):
            for metric in metrics:
                catalog[f"options.{group}.{metric}"] = (group, None, metric)
    if frames[source] is not None:
        for metric in _FLOW:
            catalog[f"options.flow.{metric}"] = ("flow", None, metric)
    if frames["candles"] is not None:
        for ticker in frames["candles"].ticker.unique():
            for metric in _CONTRACT:
                catalog[f"options.contract[{ticker}].{metric}"] = ("contract", ticker, metric)
    if names is not None:
        selected = requested
        if set(selected) - catalog.keys():
            raise ValueError(f"Unknown option features: {sorted(set(selected) - catalog.keys())}")
    else:
        chosen = chosen_groups
        if set(chosen) - set(OPTION_FEATURE_GROUPS):
            raise ValueError("Unknown option feature group")
        selected = tuple(name for name, (group, _, _) in catalog.items() if group in chosen)
    decisions = _ns(index + step)
    output = {}
    chain_names = [name for name in selected if catalog[name][0] not in ("flow", "contract")]
    if chain_names:
        history, deadlines = [], []
        values = {name: [] for name in chain_names}
        latest = None
        for (available, snapshot), raw in frames["chain"].groupby(
            ["available_at", "snapshot_at"], sort=True
        ):
            if latest is not None and snapshot < latest:
                continue
            latest = snapshot
            days = (raw.expiry - snapshot.normalize()).dt.days
            active = raw.loc[(days >= min_dte) & (days <= max_dte)]
            surface = _Surface(
                active,
                atm_tolerance=atm_tolerance,
                delta_target=delta_target,
                delta_tolerance=delta_tolerance,
            )
            history.append(available.value)
            deadlines.append((snapshot.normalize() + pd.Timedelta("1D")).value)
            for name in chain_names:
                group, _, metric = catalog[name]
                values[name].append(_snapshot_metric(raw, active, snapshot, group, metric, surface))
        times = np.asarray(history, dtype=np.int64)
        live = _alive(times, np.asarray(deadlines, dtype=np.int64), decisions)
        for name, samples in values.items():
            group, _, metric = catalog[name]
            age = metric.endswith("age_seconds")
            dtype = np.int64 if metric == "snapshot_age_seconds" else float
            out = _align(
                times,
                np.asarray(samples, dtype=dtype),
                decisions,
                reference=times,
                max_age=None if group == "quality" else age_limit,
                age=age,
            )
            if group != "quality":
                out[~live] = np.nan
            output[name] = out
    flow_names = [name for name in selected if catalog[name][0] == "flow"]
    if flow_names:
        metrics = {catalog[name][2] for name in flow_names}
        data = frames[source]
        event_day = (data.trade_at if source == "prints" else data.bar_at).dt.normalize()
        days = (data.expiry - event_day).dt.days
        data = data.loc[(days >= min_dte) & (days <= max_dte)]
        values = _flow_values(data, decisions, window, metrics, source)
        output.update({name: values[catalog[name][2]] for name in flow_names})
    contract_names = [name for name in selected if catalog[name][0] == "contract"]
    for ticker in dict.fromkeys(catalog[name][1] for name in contract_names):
        frame = frames["candles"].loc[frames["candles"].ticker == ticker].sort_values("bar_at")
        times = _ns(frame.available_at)
        if np.any(np.diff(times) < 0):
            raise ValueError("Contract candle availability must follow bar order")
        close_times = _ns(frame.bar_at + pd.Timedelta(frame.attrs["source_frequency"]))
        for name in contract_names:
            if catalog[name][1] != ticker:
                continue
            metric = catalog[name][2]
            if metric == "age_seconds":
                samples = close_times
            elif metric == "log_return":
                close = frame.close.to_numpy(float)
                samples = np.full(len(frame), np.nan)
                valid = (
                    (close[1:] > 0)
                    & (close[:-1] > 0)
                    & (np.diff(close_times) == pd.Timedelta(frame.attrs["source_frequency"]).value)
                )
                samples[1:][valid] = np.log(close[1:][valid]) - np.log(close[:-1][valid])
            elif metric == "intrabar_log_return":
                opening, close = frame.open.to_numpy(float), frame.close.to_numpy(float)
                samples = np.full(len(frame), np.nan)
                valid = (opening > 0) & (close > 0)
                samples[valid] = np.log(close[valid]) - np.log(opening[valid])
            elif metric == "range_relative":
                samples = np.full(len(frame), np.nan)
                np.divide(
                    (frame.high - frame.low).to_numpy(float),
                    frame.close.to_numpy(float),
                    out=samples,
                    where=frame.close.to_numpy(float) > 0,
                )
            else:
                samples = frame[metric].to_numpy(float)
            output[name] = _align(
                times,
                samples,
                decisions,
                reference=close_times,
                max_age=None if metric == "age_seconds" else age_limit,
                age=metric == "age_seconds",
            )
    result = pd.DataFrame({name: output[name] for name in selected}, index=index, copy=False)
    result.attrs.update(
        source_frequency=str(step),
        timestamp_semantics="bar_open",
        available_at="index + source_frequency",
        underlying=underlying,
        flow_source=source,
    )
    return result


def compute_option_feature(bars, name, **options):
    """Calculate one requested option output, without evaluating its whole group."""
    frame = compute_option_features(bars, names=(name,), **options)
    result = frame[name]
    result.attrs.update(frame.attrs)
    return result
