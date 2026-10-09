"""Context-local backend selection for reproducible comparisons and debugging."""

from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module

try:
    native = import_module("pfe._native")
except ImportError:
    native = None

_backend = ContextVar("pfe_feature_backend", default="cpp" if native is not None else "python")


def using_cpp():
    return _backend.get() == "cpp"


@contextmanager
def feature_backend(name: str):
    """Temporarily select cpp or python; other threads/tasks retain their choice.

    Default is C++ in a compiled installation. Python remains an exact reference
    implementation. Selecting unavailable C++ raises instead of silently falling
    back. TA-Lib/NumPy/Numba kernels outside this port are unchanged.
    """
    if name not in {"cpp", "python"}:
        raise ValueError("backend must be 'cpp' or 'python'")
    if name == "cpp" and native is None:
        raise RuntimeError("C++ extension is not installed; run uv sync --reinstall-package pfe")
    token = _backend.set(name)
    try:
        yield
    finally:
        _backend.reset(token)
