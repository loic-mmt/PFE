"""Confirmed swing/shape descriptors. Never backdate signals to pivot bars."""

import numpy as np
import pandas as pd

from pfe.features._utils import event_age
from pfe.features.rolling import rolling_kernel, slope_convergence


def structure_features(bars: pd.DataFrame, windows: tuple[int, ...], threshold: float) -> dict:
    h, low, c = bars.high, bars.low, bars.close
    out = {
        "structure.higher_high": (h > h.shift()).astype(float).where(h.shift().notna()),
        "structure.higher_low": (low > low.shift()).astype(float).where(low.shift().notna()),
        "structure.lower_high": (h < h.shift()).astype(float).where(h.shift().notna()),
        "structure.lower_low": (low < low.shift()).astype(float).where(low.shift().notna()),
    }
    for w in windows:
        side = max(1, w // 2)
        lookback = 2 * side + 1
        confirmed_high = h.shift(side).where(h.shift(side) == h.rolling(lookback).max())
        confirmed_low = low.shift(side).where(low.shift(side) == low.rolling(lookback).min())
        for kind, pivots in (("high", confirmed_high), ("low", confirmed_low)):
            valid = h.rolling(lookback).count() == lookback
            event = pivots.notna()
            latest = pivots.ffill()
            previous = pivots[event].shift().reindex(bars.index).ffill()
            base = f"structure.confirmed_{kind}_w{w}"
            out[base] = latest
            out[f"{base}_event"] = event.astype(float).where(valid)
            out[f"{base}_age"] = event_age(event).where(valid)
            out[f"{base}_distance"] = c / latest - 1
            out[f"{base}_previous_distance"] = latest / previous - 1
            # Explicit tolerance-based repeated confirmed pivots, not a claim
            # that a discretionary chart pattern has a unique definition.
            double = (latest / previous - 1).abs() <= threshold
            out[f"structure.double_{'top' if kind == 'high' else 'bottom'}_w{w}"] = (
                (double & event).astype(float).where(previous.notna())
            )
        # Slope convergence of rolling extrema is a causal triangle descriptor.
        x = np.arange(w, dtype=float)
        weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
        high_slope = rolling_kernel(h, w, "dot", weights)
        low_slope = rolling_kernel(low, w, "dot", weights)
        out[f"structure.triangle_convergence_w{w}"] = slope_convergence(
            h, low, w, weights, high_slope, low_slope
        )
        out[f"structure.consolidation_width_w{w}"] = (h.rolling(w).max() - low.rolling(w).min()) / c
    # ZigZag state emits a pivot only once the threshold reversal is observed.
    close = c.to_numpy()
    last_pivot = np.full(len(c), np.nan)
    direction = np.zeros(len(c))
    event = np.zeros(len(c))
    if len(c):
        extreme, confirmed, state = close[0], np.nan, 0
        initial_high = initial_low = close[0]
        for i, price in enumerate(close):
            if state == 0:
                initial_high, initial_low = max(initial_high, price), min(initial_low, price)
                if price / initial_low - 1 >= threshold:
                    confirmed, state, extreme, event[i] = initial_low, 1, price, 1
                elif price / initial_high - 1 <= -threshold:
                    confirmed, state, extreme, event[i] = initial_high, -1, price, -1
            elif state > 0:
                extreme = max(extreme, price)
                if price / extreme - 1 <= -threshold:
                    confirmed, state, extreme, event[i] = extreme, -1, price, -1
            else:
                extreme = min(extreme, price)
                if price / extreme - 1 >= threshold:
                    confirmed, state, extreme, event[i] = extreme, 1, price, 1
            last_pivot[i], direction[i] = confirmed, state
    out.update(
        {
            "structure.zigzag_confirmed_pivot": last_pivot,
            "structure.zigzag_direction": direction,
            "structure.zigzag_event": event,
        }
    )
    return out
