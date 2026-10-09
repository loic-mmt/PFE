"""Clock/session descriptors and levels from previous observed sessions."""

from datetime import time

import numpy as np
import pandas as pd

from pfe.features._utils import divide


def _session_gap_inputs(bars, source_duration, *, timezone, session_start, session_end):
    """Vectorized opening gaps from exact observed regular-session endpoints.

    Work is linear in source rows; only first/last positions are summarized.
    Local calendar days, rather than elapsed UTC hours, classify weekend closures.
    """
    opening, closing = time.fromisoformat(session_start), time.fromisoformat(session_end)
    open_seconds = opening.hour * 3600 + opening.minute * 60 + opening.second
    close_seconds = closing.hour * 3600 + closing.minute * 60 + closing.second
    if close_seconds <= open_seconds:
        raise ValueError("session_end must follow session_start on the same local date")
    local = bars.index.tz_convert(timezone)
    seconds = np.asarray(
        local.hour * 3600 + local.minute * 60 + local.second + local.microsecond / 1e6
    )
    positions = np.flatnonzero(
        (seconds >= open_seconds)
        & (seconds + source_duration.total_seconds() <= close_seconds)
        & (local.dayofweek < 5)
    )
    gap = np.full(len(bars), np.nan)
    weekend = np.full(len(bars), np.nan)
    if len(positions):
        days = local[positions].tz_localize(None).normalize().as_unit("ns").asi8
        starts = np.r_[0, np.flatnonzero(np.diff(days)) + 1]
        counts = np.diff(np.r_[starts, len(positions)])
        first = positions[starts]
        last = positions[starts + counts - 1]
        day_numbers = days[starts] // pd.Timedelta("1D").value
        values = np.full(len(starts), np.nan)
        crosses_weekend = np.zeros(len(starts))
        valid = (seconds[first[1:]] == open_seconds) & (
            seconds[last[:-1]] + source_duration.total_seconds() == close_seconds
        )
        opening_prices = bars.open.to_numpy(dtype=float, copy=False)[first[1:]]
        closing_prices = bars.close.to_numpy(dtype=float, copy=False)[last[:-1]]
        values[1:] = np.where(valid, opening_prices / closing_prices - 1, np.nan)
        # 1970-01-01 was Thursday (weekday 3); Saturday starts the weekend.
        previous_weekday = (day_numbers[:-1] + 3) % 7
        crosses_weekend[1:] = np.diff(day_numbers) >= 5 - previous_weekday
        gap[positions] = np.repeat(values, counts)
        weekend[positions] = np.repeat(crosses_weekend, counts)
    return pd.DataFrame({"gap": gap, "crosses_weekend": weekend}, index=bars.index)


def session_features(
    bars: pd.DataFrame,
    source_duration: pd.Timedelta,
    *,
    timezone: str,
    session_start: str,
    session_end: str,
    opening_range: str,
) -> dict:
    opening, closing = time.fromisoformat(session_start), time.fromisoformat(session_end)
    open_seconds = opening.hour * 3600 + opening.minute * 60 + opening.second
    close_seconds = closing.hour * 3600 + closing.minute * 60 + closing.second
    if close_seconds <= open_seconds:
        raise ValueError("session_end must follow session_start on the same local date")
    opening_duration = pd.Timedelta(opening_range)
    if (
        opening_duration <= pd.Timedelta(0)
        or opening_duration.total_seconds() > close_seconds - open_seconds
    ):
        raise ValueError("opening_range must fit inside the session")
    local = bars.index.tz_convert(timezone)
    day = local.normalize()
    seconds = local.hour * 3600 + local.minute * 60 + local.second + local.microsecond / 1e6
    elapsed = pd.Series(seconds - open_seconds, index=bars.index)
    within = (
        (elapsed >= 0)
        & (elapsed + source_duration.total_seconds() <= close_seconds - open_seconds)
        & (local.dayofweek < 5)
    )
    day_keys = pd.Series(day, index=bars.index)
    data = bars.loc[within]
    keys = day_keys.loc[within]
    out = {
        "calendar.hour_sin": np.sin(2 * np.pi * seconds / 86400),
        "calendar.hour_cos": np.cos(2 * np.pi * seconds / 86400),
        "calendar.weekday_sin": np.sin(2 * np.pi * local.dayofweek / 7),
        "calendar.weekday_cos": np.cos(2 * np.pi * local.dayofweek / 7),
        "calendar.month_sin": np.sin(2 * np.pi * (local.month - 1) / 12),
        "calendar.month_cos": np.cos(2 * np.pi * (local.month - 1) / 12),
        "calendar.weekday": local.dayofweek.astype(float),
        "session.clock_regular_hours": within.astype(float),
        "session.elapsed_minutes": (elapsed / 60).where(within),
        "session.remaining_minutes": (
            (close_seconds - open_seconds - elapsed - source_duration.total_seconds()) / 60
        ).where(within),
    }
    names = (
        "return_from_open",
        "high_so_far",
        "low_so_far",
        "range_so_far",
        "volume_so_far",
        "vwap_proxy",
        "vwap_distance",
        "volume_same_slot_relative",
        "volatility_same_slot_relative",
        "opening_range_high",
        "opening_range_low",
        "opening_range_position",
        "opening_range_coverage",
    )
    for name in names:
        out[f"session.{name}"] = pd.Series(np.nan, index=bars.index)
    pivot_names = ["previous_high", "previous_low", "previous_close", "classic_p", "woodie_p"]
    for family, count in (("classic", 3), ("woodie", 2), ("fibonacci", 3), ("camarilla", 4)):
        pivot_names.extend(
            f"{family}_{side}{i}" for side in ("r", "s") for i in range(1, count + 1)
        )
    for name in pivot_names:
        for suffix in ("", "_distance", "_cross"):
            out[f"session.{name}{suffix}"] = pd.Series(np.nan, index=bars.index)
    out["session.previous_coverage"] = pd.Series(np.nan, index=bars.index)
    out["session.overnight_gap"] = pd.Series(np.nan, index=bars.index)
    gaps = _session_gap_inputs(
        bars,
        source_duration,
        timezone=timezone,
        session_start=session_start,
        session_end=session_end,
    )
    out["session.night_gap"] = gaps["gap"].where(gaps["crosses_weekend"] == 0)
    out["session.weekend_gap"] = gaps["gap"].where(gaps["crosses_weekend"] == 1)
    if data.empty:
        return out
    grouped = data.groupby(keys, sort=False)
    high = grouped.high.cummax()
    low = grouped.low.cummin()
    volume = grouped.volume.cumsum()
    first_open = grouped.open.transform("first")
    first_elapsed = elapsed.loc[within].groupby(keys).transform("first")
    first_open = first_open.where(first_elapsed == 0)
    typical = (data.high + data.low + data.close) / 3
    vwap = divide((typical * data.volume).groupby(keys).cumsum(), volume)
    out.update(
        {
            "session.return_from_open": (data.close / first_open - 1).reindex(bars.index),
            "session.high_so_far": high.reindex(bars.index),
            "session.low_so_far": low.reindex(bars.index),
            "session.range_so_far": (high - low).reindex(bars.index),
            "session.volume_so_far": volume.reindex(bars.index),
            "session.vwap_proxy": vwap.reindex(bars.index),
            "session.vwap_distance": (data.close / vwap - 1).reindex(bars.index),
        }
    )
    slot = pd.Series(np.asarray(seconds)[within.to_numpy()], index=data.index)
    # Shift within the same clock slot: only preceding observed sessions count.
    past_volume = data.volume.groupby(slot).transform(lambda s: s.shift().expanding().mean())
    movement = np.log(data.close / data.open).abs()
    past_movement = movement.groupby(slot).transform(lambda s: s.shift().expanding().mean())
    out["session.volume_same_slot_relative"] = divide(data.volume, past_volume).reindex(bars.index)
    out["session.volatility_same_slot_relative"] = divide(movement, past_movement).reindex(
        bars.index
    )
    or_mask = (
        elapsed.loc[within] + source_duration.total_seconds() <= opening_duration.total_seconds()
    )
    initial = (
        data.loc[or_mask]
        .groupby(keys.loc[or_mask])
        .agg(high=("high", "max"), low=("low", "min"), count=("close", "size"))
    )
    ready = elapsed + source_duration.total_seconds() >= opening_duration.total_seconds()
    for name in ("high", "low"):
        out[f"session.opening_range_{name}"] = day_keys.map(initial[name]).where(within & ready)
    out["session.opening_range_coverage"] = (
        day_keys.map(initial["count"]) * source_duration / opening_duration
    ).where(within & ready)
    out["session.opening_range_position"] = divide(
        bars.close - out["session.opening_range_low"],
        out["session.opening_range_high"] - out["session.opening_range_low"],
    )
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        count=("close", "size"),
    )
    previous = daily.shift()
    ph, pl, pc = (day_keys.map(previous[name]) for name in ("high", "low", "close"))
    p = (ph + pl + pc) / 3
    width = ph - pl
    levels = {
        "previous_high": ph,
        "previous_low": pl,
        "previous_close": pc,
        "classic_p": p,
        "classic_r1": 2 * p - pl,
        "classic_s1": 2 * p - ph,
        "classic_r2": p + width,
        "classic_s2": p - width,
        "classic_r3": ph + 2 * (p - pl),
        "classic_s3": pl - 2 * (ph - p),
        "woodie_p": (ph + pl + 2 * pc) / 4,
    }
    wp = levels["woodie_p"]
    levels.update(
        woodie_r1=2 * wp - pl, woodie_s1=2 * wp - ph, woodie_r2=wp + width, woodie_s2=wp - width
    )
    for i, factor in enumerate((0.382, 0.618, 1.0), 1):
        levels[f"fibonacci_r{i}"] = p + factor * width
        levels[f"fibonacci_s{i}"] = p - factor * width
    for i, divisor in enumerate((12, 6, 4, 2), 1):
        levels[f"camarilla_r{i}"] = pc + 1.1 * width / divisor
        levels[f"camarilla_s{i}"] = pc - 1.1 * width / divisor
    for name, value in levels.items():
        out[f"session.{name}"] = value
        out[f"session.{name}_distance"] = bars.close / value - 1
        out[f"session.{name}_cross"] = (
            (np.sign(bars.close - value).diff().abs() > 0).astype(float).where(value.notna())
        )
    out["session.previous_coverage"] = (
        day_keys.map(previous["count"])
        * source_duration.total_seconds()
        / (close_seconds - open_seconds)
    )
    out["session.overnight_gap"] = (first_open / keys.map(previous["close"]) - 1).reindex(
        bars.index
    )
    return out
