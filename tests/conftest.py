"""Synthetic bars only; no downloaded market data in the test suite."""

import pandas as pd
import pytest


@pytest.fixture
def raw_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-05T14:30:00Z", "2026-01-05T14:31:00Z"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 0.0],
        }
    )
