"""Descriptors of the observed fine-grained path inside one closed target bar."""

import numpy as np
import pandas as pd

from pfe.features._utils import correlation, entropy, ratio, regression, runs
from pfe.features.profiles import profile_features


def excursion_features(prices: np.ndarray, seconds: np.ndarray, threshold: float) -> dict:
    out = {}
    for invert, prefix in ((False, "drawdown"), (True, "drawup")):
        path = 1 / prices if invert else prices
        peaks = np.maximum.accumulate(path)
        depths = 1 - path / peaks
        trough = int(np.argmax(depths))
        peak = int(np.flatnonzero(path[: trough + 1] == peaks[trough])[-1])
        recovered = np.flatnonzero(path[trough + 1 :] >= peaks[trough])
        recovery = int(trough + 1 + recovered[0]) if len(recovered) else None
        under = depths > 0
        states, lengths = runs(under)
        episode_count = np.count_nonzero(
            (depths >= threshold) & np.r_[True, depths[:-1] < threshold]
        )
        underwater_seconds = np.diff(seconds) * under[1:]
        longest_duration = 0.0
        active_duration = 0.0
        for is_under, dt in zip(under[1:], np.diff(seconds), strict=True):
            active_duration = active_duration + dt if is_under else 0.0
            longest_duration = max(longest_duration, active_duration)
        out.update(
            {
                f"path.max_{prefix}": float(depths[trough]),
                f"path.{prefix}_peak_to_trough_seconds": float(seconds[trough] - seconds[peak]),
                f"path.{prefix}_recovery_seconds": float(seconds[recovery] - seconds[trough])
                if recovery is not None
                else np.nan,
                f"path.{prefix}_recovered": float(recovery is not None or depths[trough] == 0),
                f"path.{prefix}_recovery_fraction": ratio(
                    float(path[-1] - path[trough]), float(peaks[trough] - path[trough])
                ),
                f"path.{prefix}_threshold_episodes": float(episode_count),
                f"path.{prefix}_underwater_observations": float(np.count_nonzero(under)),
                f"path.{prefix}_underwater_seconds": float(underwater_seconds.sum()),
                f"path.{prefix}_longest_underwater_seconds": float(longest_duration),
                f"path.{prefix}_longest_underwater_run": float(
                    max((n for s, n in zip(states, lengths, strict=True) if s), default=0)
                ),
            }
        )
    return out


def intrabar_features(
    bars: pd.DataFrame,
    *,
    start: pd.Timestamp,
    duration: pd.Timedelta,
    source_duration: pd.Timedelta,
    segments: int,
    bins: int,
    threshold: float,
    internal_scales: tuple[int, ...],
) -> dict[str, float]:
    o, h, low, c, v = (bars[x].to_numpy() for x in ("open", "high", "low", "close", "volume"))
    n = len(bars)
    elapsed = np.asarray((bars.index - start) / duration, dtype=float)
    gaps = np.asarray(bars.index[1:] - bars.index[:-1]) != source_duration.to_timedelta64()
    r = np.r_[np.log(c[0] / o[0]), np.diff(np.log(c))]
    r[1:][gaps] = np.nan
    valid = r[np.isfinite(r)]
    signs = np.sign(valid)
    nonzero = signs[signs != 0]
    sign_states, sign_lengths = runs(signs)
    expected = int(duration / source_duration)
    complete = n == expected and not gaps.any() and bars.index[0] == start
    energy = valid**2
    rv_observed = float(energy.sum())
    length = float(np.abs(valid).sum())
    out = {
        "quality.observed_bars": float(n),
        "quality.expected_bars": float(expected),
        "quality.missing_bars": float(max(0, expected - n)),
        "quality.coverage": n / expected,
        "quality.complete": float(complete),
        "quality.return_coverage": float(np.isfinite(r).mean()),
        "quality.largest_gap_seconds": float(
            max(
                np.asarray((bars.index[1:] - bars.index[:-1]).total_seconds()),
                default=source_duration.total_seconds(),
            )
        ),
        "quality.observed_span_seconds": float(
            (bars.index[-1] - bars.index[0] + source_duration).total_seconds()
        ),
        "quality.unchanged_close_fraction": float(np.mean(np.diff(c) == 0)) if n > 1 else np.nan,
        "quality.crosses_local_date": float(bars.index[0].date() != bars.index[-1].date()),
        "path.realized_variance": rv_observed if complete else np.nan,
        "path.realized_volatility": float(np.sqrt(rv_observed)) if complete else np.nan,
        "path.realized_variance_observed": rv_observed,
        "path.absolute_return_sum": length,
        "path.price_quadratic_variation": float(np.sum(np.diff(np.r_[o[0], c]) ** 2))
        if complete
        else np.nan,
        "path.up_semivariance": float(np.sum(valid[valid > 0] ** 2)),
        "path.down_semivariance": float(np.sum(valid[valid < 0] ** 2)),
        "path.directional_efficiency": ratio(abs(float(valid.sum())), length),
        "path.up_fraction": float(np.mean(valid > 0)),
        "path.down_fraction": float(np.mean(valid < 0)),
        "path.flat_fraction": float(np.mean(valid == 0)),
        "path.up_return_sum": float(valid[valid > 0].sum()),
        "path.down_return_sum": float(valid[valid < 0].sum()),
        "path.directional_imbalance": ratio(float(np.sign(valid).sum()), float(len(valid))),
        "path.turn_count": float(np.count_nonzero(np.diff(nonzero))),
        "path.mean_run_length": float(sign_lengths.mean()),
        "path.longest_up_run": float(
            max(
                (length for s, length in zip(sign_states, sign_lengths, strict=True) if s == 1),
                default=0,
            )
        ),
        "path.longest_down_run": float(
            max(
                (length for s, length in zip(sign_states, sign_lengths, strict=True) if s == -1),
                default=0,
            )
        ),
        "path.max_positive_return": float(max(valid.max(), 0)),
        "path.max_negative_return": float(min(valid.min(), 0)),
        "path.max_absolute_return": float(np.abs(valid).max()),
        "path.large_move_count": float(np.count_nonzero(np.abs(valid) >= threshold)),
        "path.return_energy_concentration": ratio(float(np.sum(energy**2)), rv_observed**2),
        "path.range_mean": float(np.mean(h - low)),
        "path.range_std": float(np.std(h - low)),
        "path.range_expansion_slope": regression(h - low)["slope"],
    }
    out["path.semivariance_ratio"] = ratio(
        out["path.up_semivariance"], out["path.down_semivariance"]
    )
    for k in (1, 3, 5):
        out[f"path.top{k}_energy_share"] = ratio(float(np.sort(energy)[-k:].sum()), rv_observed)
    hi_ids, lo_ids = np.flatnonzero(h == h.max()), np.flatnonzero(low == low.min())
    for prefix, ids in (("high", hi_ids), ("low", lo_ids)):
        out[f"path.{prefix}_first_time_fraction"] = float(elapsed[ids[0]])
        out[f"path.{prefix}_last_time_fraction"] = float(elapsed[ids[-1]])
        out[f"path.{prefix}_retests"] = float(len(ids) - 1)
        out[f"path.{prefix}_to_close_seconds"] = float(
            (bars.index[-1] - bars.index[ids[-1]]).total_seconds()
        )
    out["path.high_before_low"] = float(hi_ids[0] < lo_ids[0]) if hi_ids[0] != lo_ids[0] else np.nan
    out["path.extrema_delay_fraction"] = float(abs(elapsed[hi_ids[0]] - elapsed[lo_ids[0]]))
    seconds = np.r_[
        0.0, np.asarray((bars.index - start + source_duration).total_seconds(), dtype=float)
    ]
    out.update(excursion_features(np.r_[o[0], c], seconds, threshold))
    logpath = np.log(c / o[0])
    for name, value in regression(logpath).items():
        out[f"path.regression_{name}"] = value
    out["path.curvature"] = float(np.mean(np.diff(logpath, n=2))) if n > 2 else np.nan
    out["path.signed_area"] = float(np.mean(c / o[0] - 1))
    out["path.absolute_area"] = float(np.mean(np.abs(c / o[0] - 1)))
    straight = np.log(c[-1] / o[0]) * np.clip(elapsed + source_duration / duration, 0, 1)
    out["path.straight_line_error"] = float(np.sqrt(np.mean((logpath - straight) ** 2)))
    out["path.return_autocorr"] = correlation(r[:-1], r[1:])
    out["path.absolute_return_autocorr"] = correlation(np.abs(r[:-1]), np.abs(r[1:]))
    out["path.absolute_return_slope"] = regression(np.abs(valid))["slope"]
    absolute = np.where(np.isfinite(r), np.abs(r), -np.inf)
    out["path.largest_move_time_fraction"] = float(elapsed[np.argmax(absolute)])
    first = elapsed < 0.5
    out["path.first_half_energy_share"] = ratio(float(np.nansum(r[first] ** 2)), rv_observed)
    out["path.last_half_energy_share"] = ratio(float(np.nansum(r[~first] ** 2)), rv_observed)
    out["path.half_direction_agreement"] = float(
        np.sign(np.nansum(r[first])) == np.sign(np.nansum(r[~first]))
    )
    above = c > o[0]
    out["path.open_crossings"] = float(np.count_nonzero(np.diff(above)))
    vtotal = float(v.sum())
    out.update(
        {
            "path.volume_concentration": ratio(float(np.sum(v**2)), vtotal**2),
            "path.volume_entropy": entropy(v),
            "path.volume_peak_time_fraction": float(elapsed[np.argmax(v)]),
            "path.volume_time_center": ratio(float(np.dot(v, elapsed)), vtotal),
            "path.volume_slope": regression(v)["slope"],
            "path.zero_volume_count": float(np.count_nonzero(v == 0)),
            "path.return_volume_corr": correlation(r, v),
            "path.absolute_return_volume_corr": correlation(np.abs(r), v),
            "path.range_volume_corr": correlation(h - low, v),
            "path.volume_up_share": ratio(float(v[r > 0].sum()), vtotal),
            "path.volume_down_share": ratio(float(v[r < 0].sum()), vtotal),
            "path.volume_weighted_return": ratio(
                float(np.nansum(v * r)), float(v[np.isfinite(r)].sum())
            ),
            "path.price_volume_peak_delay_fraction": float(
                elapsed[np.argmax(v)] - elapsed[np.argmax(absolute)]
            ),
        }
    )
    valid_positions = np.flatnonzero(np.isfinite(r))
    top_positions = valid_positions[
        np.argsort(np.abs(r[valid_positions]))[-min(3, len(valid_positions)) :]
    ]
    out["path.top3_moves_volume_share"] = ratio(float(v[top_positions].sum()), vtotal)
    high_volume = v > 2 * v.mean()
    states, lengths = runs(high_volume)
    out["path.high_volume_episode_count"] = float(np.count_nonzero(states))
    out["path.high_volume_longest_run"] = float(
        max((length for state, length in zip(states, lengths, strict=True) if state), default=0)
    )
    out["path.high_volume_observations"] = float(np.count_nonzero(high_volume))
    out["path.large_move_low_volume_count"] = float(
        np.count_nonzero((np.abs(r) >= threshold) & (v < np.median(v)))
    )
    for lag in (1, 2):
        out[f"path.return_volume_corr_lag{lag}"] = correlation(r[lag:], v[:-lag])
    segment_ids = np.clip(np.floor(elapsed * segments).astype(int), 0, segments - 1)
    last_return, last_volatility, last_volume = np.nan, np.nan, np.nan
    for segment in range(segments):
        mask = segment_ids == segment
        key = f"path.segment{segment}"
        for name in (
            "return",
            "range",
            "realized_variance",
            "volatility",
            "volume",
            "volume_share",
            "close_position",
            "return_change",
            "volatility_change",
            "volume_change",
        ):
            out[f"{key}_{name}"] = np.nan
        if not mask.any():
            continue
        rv = float(np.nansum(r[mask] ** 2))
        returns = float(np.log(c[mask][-1] / o[mask][0]))
        volatility = float(np.sqrt(rv))
        volume = float(v[mask].sum())
        hi, lo = float(h[mask].max()), float(low[mask].min())
        out.update(
            {
                f"{key}_return": returns,
                f"{key}_range": hi - lo,
                f"{key}_realized_variance": rv,
                f"{key}_volatility": volatility,
                f"{key}_volume": volume,
                f"{key}_volume_share": ratio(volume, vtotal),
                f"{key}_close_position": ratio(float(c[mask][-1] - lo), hi - lo),
                f"{key}_return_change": returns - last_return,
                f"{key}_volatility_change": volatility - last_volatility,
                f"{key}_volume_change": volume - last_volume,
            }
        )
        last_return, last_volatility, last_volume = returns, volatility, volume
    out["path.end_start_volatility_ratio"] = ratio(
        out[f"path.segment{segments - 1}_volatility"], out["path.segment0_volatility"]
    )
    out.update(profile_features(bars, start=start, duration=duration, bins=bins, segments=segments))
    # Internal resolutions are integer multiples of source duration. Compare
    # completed, contiguous sub-bars; do not bridge holes with synthetic returns.
    for scale in internal_scales:
        key = f"path.scale{scale}"
        out[f"{key}_realized_variance"] = np.nan
        out[f"{key}_rv_ratio"] = np.nan
        out[f"{key}_turn_count"] = np.nan
        out[f"{key}_direction_agreement"] = np.nan
        if not complete or expected % scale or scale > expected:
            continue
        coarse_c = c.reshape(-1, scale)[:, -1]
        coarse_r = np.diff(np.log(np.r_[o[0], coarse_c]))
        value = float(np.sum(coarse_r**2))
        out[f"{key}_realized_variance"] = value
        out[f"{key}_rv_ratio"] = ratio(value, rv_observed)
        coarse_signs = np.sign(coarse_r[coarse_r != 0])
        out[f"{key}_turn_count"] = float(np.count_nonzero(np.diff(coarse_signs)))
        out[f"{key}_direction_agreement"] = float(
            np.mean(np.sign(coarse_r) == np.sign(np.log(c[-1] / o[0])))
        )
    return out
