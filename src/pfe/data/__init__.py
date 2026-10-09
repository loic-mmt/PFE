"""Local market data loading and normalization."""

from pfe.data.bars import load_bars, normalize_bars
from pfe.data.streaming import iter_bars

__all__ = ["iter_bars", "load_bars", "normalize_bars"]
