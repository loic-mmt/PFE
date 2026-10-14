"""Calendar-history adapter for the existing sparse macro feature engine."""

import numpy as np
import pandas as pd


def calendar_history_features(history, decisions, specifications, *, window, max_age):
    """Resolve at knowledge epochs, not once per dense market row.

    Surprise baselines are frozen strictly before a verified release; subsequent
    actual revisions remain visible. Withdrawals/cancellations are full-state
    changes and never fall back to an earlier version of the same event.
    """
    from pfe.features.macro import _align, _event_history, _schedule

    output = {name: np.full(len(decisions), np.nan) for name in specifications}
    epochs = history.available_at.array.as_unit("ns").asi8
    positions = np.searchsorted(epochs, decisions, side="right")
    for end in np.unique(positions):
        locations = np.flatnonzero(positions == end)
        now = decisions[locations]
        visible = history.iloc[:end]
        snapshot = visible.drop_duplicates("event_id", keep="last").copy()
        snapshot = snapshot.loc[snapshot.status.ne("cancelled")]
        first = visible.groupby("event_id").available_at.min()
        snapshot["known_at"] = snapshot.event_id.map(first)
        # The forecast field can be shown as currently known, but the surprise
        # benchmark must be the last FULL version strictly before announcement.
        baseline = snapshot.copy()
        for i, row in baseline.iterrows():
            before = (
                visible.loc[
                    visible.event_id.eq(row.event_id) & (visible.available_at < row.release_at)
                ]
                if pd.notna(row.release_at)
                else visible.iloc[:0]
            )
            if before.empty:
                baseline.loc[i, "forecast"] = np.nan
                continue
            prior = before.iloc[-1]
            if (
                prior.status == "cancelled"
                or prior.unit != row.unit
                or pd.isna(prior.forecast_available_at)
                or prior.forecast_available_at >= row.release_at
            ):
                baseline.loc[i, "forecast"] = np.nan
                continue
            baseline.loc[i, "forecast"] = prior.forecast
            baseline.loc[i, "forecast_available_at"] = prior.forecast_available_at
            baseline.loc[i, "point_in_time"] = bool(row.point_in_time and prior.point_in_time)
            baseline.loc[i, "scheduled_at"] = row.release_at
        for name, (group, key, metric) in specifications.items():
            if group == "calendar":
                value = _schedule(snapshot, now, metric, window)
            else:
                source = (
                    snapshot
                    if metric in ("actual", "forecast", "previous", "actual_change", "age_seconds")
                    else baseline
                )
                times, values = _event_history(source.loc[source.event.eq(key)], metric)
                is_age = metric == "age_seconds"
                value = _align(
                    times,
                    values,
                    now,
                    reference=times,
                    max_age=None if is_age else max_age,
                    age=is_age,
                )
            output[name][locations] = value
    return output
