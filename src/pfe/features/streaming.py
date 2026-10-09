"""Bounded-memory feature batches with exact state for RSI, EMA and ATR."""

import numpy as np
import pandas as pd
from numba import njit

from pfe.features.aggregation import fixed_duration
from pfe.features.selected import (
    _names,
    _native_catalog,
    _parameters,
    _plan,
    _validate,
    _windows,
    compute_selected_features,
)
from pfe.features.targeted import aggregate_selected_features

_FINITE_NATIVE = {
    "SMA",
    "WMA",
    "TRIMA",
    "BBANDS",
    "STDDEV",
    "VAR",
    "LINEARREG",
    "LINEARREG_SLOPE",
    "LINEARREG_INTERCEPT",
    "LINEARREG_ANGLE",
    "TSF",
    "BOP",
    "AVGPRICE",
    "MEDPRICE",
    "TYPPRICE",
    "WCLPRICE",
    "MOM",
    "ROC",
    "ROCP",
    "ROCR",
    "ROCR100",
    "WILLR",
    "AROON",
    "AROONOSC",
    "TRANGE",
    "CCI",
    "STOCH",
    "STOCHF",
    "MIDPOINT",
    "MIDPRICE",
    "BETA",
    "CORREL",
}


@njit(cache=True, nogil=True, fastmath=False)
def _recurrence(close, high, low, timestamps, step, window, kind, reset, state):
    # state: previous close, previous time, seed count, sum/average A, sum/average B.
    previous, last_time, count, a, b = state
    result = np.full(len(close), np.nan)
    for i in range(len(close)):
        if last_time != 0 and timestamps[i] <= last_time:
            raise ValueError("Feature batches must be globally sorted and non-overlapping")
        if reset and last_time != 0 and timestamps[i] - last_time != step:
            previous, count, a, b = np.nan, 0.0, 0.0, 0.0
        price = close[i]
        if kind == 1:  # EMA seeds with the first window's mean, as TA-Lib does.
            if count < window:
                a += price
                count += 1
                if count == window:
                    a /= window
                    result[i] = a
            else:
                a += (2.0 / (window + 1)) * (price - a)
                result[i] = a
        elif np.isfinite(previous):
            if kind == 0:
                change = price - previous
                up, down = max(change, 0.0), max(-change, 0.0)
            else:
                up = max(high[i] - low[i], abs(high[i] - previous), abs(low[i] - previous))
                down = 0.0
            if count < window:
                a += up
                b += down
                count += 1
                if count == window:
                    a /= window
                    b /= window
            else:
                a = (a * (window - 1) + up) / window
                b = (b * (window - 1) + down) / window
            if count >= window:
                result[i] = a if kind == 2 else (100.0 * a / (a + b) if a + b > 1e-8 else 0.0)
        previous, last_time = price, timestamps[i]
    return result, (previous, last_time, count, a, b)


class _Stream:
    def __init__(self, names, step, gap_policy, validate):
        from talib import abstract

        self.names, self.step, self.gap_policy, self.validate = names, step, gap_policy, validate
        self.tail, self.previous_time = None, None
        self.states, self.recurrent, self.finite = {}, {}, []
        self.history = 0
        catalog = _native_catalog(_windows(names))
        for name in names:
            if name in catalog:
                native, label, window, transform = catalog[name]
                if native in {"RSI", "EMA", "ATR"} and transform is None:
                    self.recurrent[name] = (window, {"RSI": 0, "EMA": 1, "ATR": 2}[native])
                    self.states[name] = (np.nan, 0, 0, 0.0, 0.0)
                    continue
                if (
                    native not in _FINITE_NATIVE
                    and not native.startswith("CDL")
                    or transform == "age"
                ):
                    raise ValueError(
                        f"Streaming state is not implemented for {name}; use compute_feature"
                    )
                function = abstract.Function(native)
                function.set_parameters(_parameters(function, window))
                history = function.lookback + (
                    int(transform) - 1 if isinstance(transform, int) else 0
                )
            else:
                family = name.split(".", 1)[0]
                if family in {"calendar", "session", "complexity"}:
                    raise ValueError(
                        f"Streaming state is not implemented for {name}; use compute_feature"
                    )
                plan = _plan(family, _windows(names))
                if name not in plan.output:
                    raise ValueError(f"Unknown feature: {name}")
                history = plan.output[name].history
                if history is None:
                    raise ValueError(
                        f"Streaming state is not implemented for {name}; use compute_feature"
                    )
            self.history = max(self.history, history)
            self.finite.append(name)

    def process(self, batch):
        data, _ = _validate(batch, self.step, self.validate)
        if self.previous_time is not None:
            delta = data.index[0].value - self.previous_time
            if delta <= 0 or delta % self.step.value:
                raise ValueError("Batches must be globally sorted, unique and on the source grid")
        out = {}
        if self.finite:
            combined = data if self.tail is None else pd.concat([self.tail, data])
            result = compute_selected_features(
                combined,
                self.finite,
                source_frequency=self.step,
                gap_policy=self.gap_policy,
                validate=False,
            )
            out.update({name: result[name].iloc[-len(data) :].to_numpy() for name in self.finite})
            self.tail = combined.iloc[-self.history :].copy() if self.history else None
        close = data.close.to_numpy(dtype=float, copy=False)
        high = data.high.to_numpy(dtype=float, copy=False)
        low = data.low.to_numpy(dtype=float, copy=False)
        for name, (window, kind) in self.recurrent.items():
            values, state = _recurrence(
                close,
                high,
                low,
                data.index.asi8,
                self.step.value,
                window,
                kind,
                self.gap_policy == "reset",
                self.states[name],
            )
            out[name], self.states[name] = values, state
        self.previous_time = data.index[-1].value
        result = pd.DataFrame(
            {name: out[name] for name in self.names}, index=data.index, copy=False
        )
        result.attrs.update(
            source_frequency=str(self.step),
            available_at="index + source_frequency",
            timestamp_semantics="bar_open",
        )
        return result


def iter_features(
    batches,
    names,
    *,
    source_frequency="1min",
    gap_policy="reset",
    validate=True,
    frequency=None,
    **aggregation_options,
):
    """Yield selected feature frames without retaining input/output history.

    Source mode supports finite dependency plans plus exact RSI/EMA/ATR state.
    Unsupported infinite-history descriptors are rejected, never approximated.
    Target mode supports every path/profile feature, and the same streaming target
    OHLCV indicators. Input batches must be sorted/non-overlapping OHLCV frames.
    """
    names = _names(names)
    source = fixed_duration(source_frequency, "source_frequency")
    if gap_policy not in {"reset", "allow"}:
        raise ValueError("gap_policy must be 'reset' or 'allow'")
    if frequency is None:
        if aggregation_options:
            raise TypeError("Aggregation options require frequency")
        stream = _Stream(names, source, gap_policy, validate)
        for batch in batches:
            if not batch.empty:
                yield stream.process(batch)
        return
    duration = fixed_duration(frequency, "frequency")
    if duration < source or duration.value % source.value:
        raise ValueError("frequency must be an integer multiple of source_frequency")
    if aggregation_options.get("partial", "drop") != "drop" or "as_of" in aggregation_options:
        raise ValueError("Streaming target mode requires partial='drop' and no as_of")
    options = {**aggregation_options, "partial": "keep"}
    path = [name for name in names if name.split(".", 1)[0] in {"path", "profile", "quality"}]
    standard = [
        name
        for name in names
        if name not in path and name not in {"open", "high", "low", "close", "volume"}
    ]
    fields = list(
        dict.fromkeys(
            [
                *path,
                *[name for name in names if name in {"open", "high", "low", "close", "volume"}],
                *(["open", "high", "low", "close", "volume"] if standard else []),
            ]
        )
    )
    stream = _Stream(tuple(standard), duration, gap_policy, False) if standard else None
    carry, first_observation, previous = None, None, None
    for batch in batches:
        if batch.empty:
            continue
        data, _ = _validate(batch, source, validate)
        if previous is not None and (
            data.index[0].value <= previous or (data.index[0].value - previous) % source.value
        ):
            raise ValueError("Batches must be globally sorted, unique and on the source grid")
        previous = data.index[-1].value
        if first_observation is None:
            first_observation = data.index[0]
            origin = options.get("origin", "start_day")
            if origin in {"start_day", "start"}:
                options["origin"] = (
                    first_observation.normalize() if origin == "start_day" else first_observation
                )
        combined = data if carry is None else pd.concat([carry, data])
        target = aggregate_selected_features(
            combined, fields, duration, source_frequency=source, validate=False, **options
        )
        ready = (target.available_at <= data.index[-1] + source) & (
            target.index >= first_observation
        )
        completed = target.loc[ready]
        provisional = target.index[target.available_at > data.index[-1] + source]
        carry = combined.loc[combined.index >= provisional[-1]].copy() if len(provisional) else None
        if completed.empty:
            continue
        if stream is not None:
            computed = stream.process(completed)
            completed = completed.join(computed)
        yield completed[[*names, "available_at"]]
