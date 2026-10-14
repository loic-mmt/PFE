"""Offline synthetic option snapshots and observed prints; no LSE credentials."""

import pandas as pd

from pfe import compute_option_feature, compute_option_features

bars = pd.date_range("2025-01-14 12:00", periods=8, freq="min", tz="UTC")
chain = pd.DataFrame(
    [
        dict(
            underlying="QQQ",
            ticker=f"EXAMPLE_{side}",
            contract_type=side,
            strike=100,
            expiry="2025-01-31",
            snapshot_at=pd.Timestamp("2025-01-14T12:00:00Z"),
            available_at=pd.Timestamp("2025-01-14T12:02:00Z"),
            underlying_price=100,
            iv=iv,
            delta=delta,
            volume=volume,
            premium=volume * 100,
        )
        for side, iv, delta, volume in [("call", 0.20, 0.5, 10), ("put", 0.24, -0.5, 20)]
    ]
)
flow = (
    chain.iloc[:2]
    .drop(columns=["snapshot_at", "available_at"])
    .assign(
        trade_id=[1, 2],
        trade_at=pd.to_datetime(["2025-01-14T12:01:00Z", "2025-01-14T12:02:00Z"]),
        available_at=pd.to_datetime(["2025-01-14T12:02:00Z", "2025-01-14T12:03:00Z"]),
    )
)

iv = compute_option_feature(bars, "options.volatility.atm_iv", chain=chain)
features = compute_option_features(
    bars,
    chain=chain,
    flow=flow,
    flow_window="3min",
    names=(
        "options.volatility.atm_iv",
        "options.activity.put_call_volume_ratio",
        "options.flow.total_premium",
        "options.flow.put_call_volume_ratio",
    ),
)
assert pd.isna(iv.iloc[0])
assert abs(iv.iloc[1] - 0.22) < 1e-12
assert features["options.flow.total_premium"].iloc[0] == 0
print(features.round(4))
