"""Probability mass functions (PMFs) on the score grid.

Every predicate returns a score in [0, 1] with one decimal, so a predicate
PMF lives on the grid {0, 0.1, ..., 1} (m = 11 points).  A candidate set's
score is a sum of predicate scores, hence its PMF lives on the grid
{0, 0.1, ..., L/10}.  All PMFs are stored as dense numpy vectors indexed by
tenths; sums of independent predicate PMFs are sum-convolutions.
"""
from __future__ import annotations

import numpy as np

GRID_M = 11           # number of grid points of a predicate PMF
STEP = 0.1


def pmf_from_samples(samples, m: int = GRID_M) -> np.ndarray:
    """Turn x repeated responses (floats in [0,1], one decimal) into a PMF."""
    pmf = np.zeros(m)
    for s in samples:
        pmf[int(round(s * (m - 1)))] += 1.0
    return pmf / pmf.sum()


def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sum-convolution of two PMFs on the same grid step."""
    return np.convolve(a, b)


def convolve_all(pmfs) -> np.ndarray:
    """Sum-convolution of a list of PMFs (identity = point mass at 0)."""
    out = np.ones(1)
    for p in pmfs:
        out = np.convolve(out, p)
    return out


def bounds(pmf: np.ndarray):
    """(min, max) of the values with non-zero mass, in score units."""
    nz = np.flatnonzero(pmf > 0)
    return nz[0] * STEP, nz[-1] * STEP


def mean(pmf: np.ndarray) -> float:
    return float(np.dot(pmf, np.arange(len(pmf)) * STEP))
