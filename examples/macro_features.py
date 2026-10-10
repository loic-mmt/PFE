"""Synthetic macro observations and announcement surprises; no LSE/network needed."""

import pandas as pd

from pfe import compute_macro_feature, compute_macro_features

bars = pd.date_range("2025-04-01 12:00", periods=8, freq="min", tz="UTC")
observations = pd.DataFrame(
    {
        "series": ["policy_rate"] * 3,
        "observation_at": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"], utc=True),
        "available_at": pd.to_datetime(["2025-01-10", "2025-02-10", "2025-03-10"], utc=True),
        "value": [4.5, 4.25, 4.0],
    }
)
events = pd.DataFrame(
    {
        "event_id": [1],
        "event": ["CPI"],
        "scheduled_at": [pd.Timestamp("2025-04-01T12:04:00Z")],
        "known_at": [pd.Timestamp("2025-03-25T00:00:00Z")],
        "actual": [3.1],
        "forecast": [3.0],
        "actual_available_at": [pd.Timestamp("2025-04-01T12:04:05Z")],
        "forecast_available_at": [pd.Timestamp("2025-03-31T00:00:00Z")],
    }
)

rate = compute_macro_feature(bars, "macro.policy_rate.level", observations=observations)
features = compute_macro_features(
    bars,
    observations=observations,
    events=events,
    windows=(3,),
    names=(
        "macro.policy_rate.change",
        "macro.policy_rate.mean_w3",
        "macro.calendar.seconds_to_next_event",
        "macro.event[CPI].surprise",
    ),
)
assert rate.eq(4.0).all()
assert features["macro.event[CPI].surprise"].iloc[:4].isna().all()
assert features["macro.event[CPI].surprise"].iloc[4:].notna().all()
print(features.round(3))
