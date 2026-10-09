"""Finite price/time profiles reconstructed from OHLCV sub-bars."""

import numpy as np
import pandas as pd

from pfe.features._utils import entropy, ratio, runs


def price_time_profile(
    bars: pd.DataFrame,
    *,
    start: pd.Timestamp,
    duration: pd.Timedelta,
    bins: int = 10,
    segments: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return presence[bins,segments], close zones, price centers, time segments.

    Price rows divide the observed high-low range equally. A source bar marks
    every row intersecting its [low, high]; this is a range-based TPO proxy,
    not a claim that each tick traded. Equal-time segments never use row count.
    A single letter denotes presence, not exact dwell time or trade count.
    """
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (bins, segments)):
        raise ValueError("bins and segments must be positive integers")
    start, duration = pd.Timestamp(start), pd.Timedelta(duration)
    if start.tzinfo is None or pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("start must be timezone-aware and duration positive")
    if bars.empty or bars.index.min() < start or bars.index.max() >= start + duration:
        raise ValueError("Profile bars must lie in the non-empty requested time window")
    lo, hi = float(bars.low.min()), float(bars.high.max())
    close = bars.close.to_numpy()
    fraction = (bars.index - start) / duration
    time_ids = np.clip(np.floor(fraction * segments).astype(int), 0, segments - 1)
    presence = np.zeros((bins, segments), dtype=bool)
    if hi == lo:
        centers = np.full(bins, lo)
        zones = np.full(len(bars), bins // 2, dtype=int)
        presence[bins // 2, np.unique(time_ids)] = True
    else:
        edges = np.linspace(lo, hi, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        zones = np.clip(np.floor((close - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
        for low, high, segment in zip(bars.low, bars.high, time_ids, strict=True):
            first = np.clip(int(np.floor((low - lo) / (hi - lo) * bins)), 0, bins - 1)
            last = np.clip(int(np.floor((high - lo) / (hi - lo) * bins)), 0, bins - 1)
            presence[first : last + 1, segment] = True
    return presence, zones, centers, time_ids


def profile_features(bars, *, start, duration, bins, segments) -> dict[str, float]:
    presence, zones, centers, time_ids = price_time_profile(
        bars, start=start, duration=duration, bins=bins, segments=segments
    )
    counts = np.bincount(zones, minlength=bins)
    tpo = presence.sum(axis=1)
    out = {
        "profile.close_entropy": entropy(counts),
        "profile.close_concentration": float(np.sum((counts / len(bars)) ** 2)),
        "profile.occupied_zones": float(np.count_nonzero(counts)),
        "profile.zone_transitions": float(np.count_nonzero(np.diff(zones))),
        "profile.zone_crossings": float(np.abs(np.diff(zones)).sum()),
        "profile.zone_revisits": float(
            sum(max(0, np.count_nonzero(runs(zones)[0] == z) - 1) for z in range(bins))
        ),
        "profile.longest_zone_stay": float(max(runs(zones)[1], default=0)),
        "profile.mean_zone_stay": float(np.mean(runs(zones)[1])),
        "profile.first_zone": float(zones[0]),
        "profile.last_zone": float(zones[-1]),
        "profile.top_to_bottom": float(
            np.count_nonzero((zones[:-1] >= bins * 2 / 3) & (zones[1:] < bins / 3))
        ),
        "profile.bottom_to_top": float(
            np.count_nonzero((zones[:-1] < bins / 3) & (zones[1:] >= bins * 2 / 3))
        ),
        "profile.tpo_entropy": entropy(tpo),
        "profile.tpo_single_block_rows": float(np.count_nonzero(tpo == 1)),
    }
    # Ties: choose the central tied POC row (lower central row for an even tie).
    tied = np.flatnonzero(tpo == tpo.max())
    poc = int(tied[(len(tied) - 1) // 2])
    out["profile.tpo_poc"] = float(centers[poc])
    out["profile.tpo_poc_close_distance"] = float(bars.close.iloc[-1] / centers[poc] - 1)
    # Contiguous 70% value area expanded from POC; equal neighbors add both.
    left = right = poc
    covered = float(tpo[poc])
    while covered < 0.7 * tpo.sum() and (left > 0 or right < bins - 1):
        lcount = tpo[left - 1] if left > 0 else -1
        rcount = tpo[right + 1] if right < bins - 1 else -1
        if lcount >= rcount and left > 0:
            left -= 1
            covered += tpo[left]
        if rcount >= lcount and right < bins - 1:
            right += 1
            covered += tpo[right]
    out["profile.tpo_value_area_low"] = float(centers[left])
    out["profile.tpo_value_area_high"] = float(centers[right])
    out["profile.tpo_value_area_width"] = float(centers[right] - centers[left])
    for zone in range(bins):
        out[f"profile.close_share_bin{zone}"] = float(counts[zone] / len(bars))
        out[f"profile.tpo_blocks_bin{zone}"] = float(tpo[zone])
        out[f"profile.tpo_share_bin{zone}"] = ratio(float(tpo[zone]), float(segments))
        for segment in range(segments):
            out[f"profile.tpo_bin{zone}_segment{segment}"] = float(presence[zone, segment])
    previous = None
    for segment in range(segments):
        mask = time_ids == segment
        active = np.flatnonzero(presence[:, segment])
        out[f"profile.segment{segment}_mean_close_zone"] = (
            float(zones[mask].mean()) if mask.any() else np.nan
        )
        out[f"profile.segment{segment}_center"] = (
            float(centers[active].mean()) if len(active) else np.nan
        )
        if previous is not None:
            intersection = np.count_nonzero(presence[:, previous] & presence[:, segment])
            union = np.count_nonzero(presence[:, previous] | presence[:, segment])
            out[f"profile.segment{segment}_overlap_previous"] = ratio(intersection, union)
            out[f"profile.segment{segment}_center_change"] = (
                out[f"profile.segment{segment}_center"] - out[f"profile.segment{previous}_center"]
            )
        previous = segment
    return out


_price_time_profile_python = price_time_profile


def price_time_profile(bars, *, start, duration, bins=10, segments=5):
    """Reconstruct TPO in C++; same arrays and conventions as the reference."""
    from pfe.features.backend import native, using_cpp

    if not using_cpp():
        return _price_time_profile_python(
            bars, start=start, duration=duration, bins=bins, segments=segments
        )
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (bins, segments)):
        raise ValueError("bins and segments must be positive integers")
    start, duration = pd.Timestamp(start), pd.Timedelta(duration)
    if start.tzinfo is None or pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("start must be timezone-aware and duration positive")
    if bars.empty or bars.index.min() < start or bars.index.max() >= start + duration:
        raise ValueError("Profile bars must lie in the non-empty requested time window")
    index = bars.index if bars.index.unit == "ns" else bars.index.as_unit("ns")
    return native.profile_arrays(
        bars.low.to_numpy(dtype=float, copy=False),
        bars.high.to_numpy(dtype=float, copy=False),
        bars.close.to_numpy(dtype=float, copy=False),
        index.asi8,
        start.value,
        duration.value,
        bins,
        segments,
    )
