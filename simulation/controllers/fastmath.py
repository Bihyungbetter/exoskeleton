"""Cheaper drop-ins for small NumPy calls in the 1 kHz controller stack.

Each helper returns exactly the same floats as the NumPy call it replaces, so
trajectories stay byte-identical. Don't add anything that isn't bit-identical
(e.g. hypot in place of norm).

norm: same ravel, dot, sqrt sequence as np.linalg.norm for a real 1-D array.
clip: the clip ufunc np.clip calls, without the wrappers.
clip_scalar: min(max(x, lo), hi); matches np.clip incl. signed zero and NaN in
x. Finite bounds only.
"""
from __future__ import annotations

import math

import numpy as np

try:                                   # NumPy 2.x
    from numpy._core.umath import clip as _clip_ufunc
except ImportError:                    # NumPy 1.x
    from numpy.core.umath import clip as _clip_ufunc


def norm(v: np.ndarray) -> float:
    """`float(np.linalg.norm(v))` for a real 1-D array, without the wrapper."""
    x = v.ravel(order="K")
    return math.sqrt(float(x.dot(x)))


def clip(a: np.ndarray, lo, hi) -> np.ndarray:
    """`np.clip(a, lo, hi)` with both bounds given, via the ufunc directly."""
    return _clip_ufunc(a, lo, hi)


def clip_scalar(x: float, lo: float, hi: float) -> float:
    """`float(np.clip(x, lo, hi))` for a scalar `x` and finite bounds."""
    x = float(x)
    return min(max(x, lo), hi)
