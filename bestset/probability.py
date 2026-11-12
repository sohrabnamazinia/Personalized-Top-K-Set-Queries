"""Bounds on the winning probabilities of the candidate sets.

The winning probability of c is the probability that its score is strictly
larger than the score of every other candidate set.  A call that has not been
issued has no PMF, so the probability is bounded:

    LB(c) = P(F-(c) > max_{c' != c} F+(c'))
    UB(c) = P(F+(c) > max_{c' != c} F-(c'))

where F- and F+ are the lower- and upper-bound scores (unknown calls at MIN
and at MAX).  Once every call of the candidate sets is issued, LB = UB.

ProbInd (independent candidate sets)
    LB(c) = sum_x f-_c(x) * prod_{c' != c} P(F+(c') < x), and symmetrically
    for UB, computed for all sets at once from the prefix-sum CDFs (the
    product over the other sets is obtained by dividing the product over
    all sets).

ProbDep (dependent candidate sets, i.e. sets that share predicate calls)
    The comparison events of a set against its competitors are dependent.
    ProbDep evaluates their intersection under the joint distribution of the
    issued calls, in which a shared call is a single random variable, with a
    sample-based evaluation (common random numbers across iterations).  An
    unknown call shared by two sets cancels in their comparison, so it is
    not set to MIN in one set and to MAX in the other.
"""
from __future__ import annotations

import numpy as np

from . import pmf as P

MAX_SCORE = (P.GRID_M - 1) * P.STEP      # MAX of a predicate call (MIN is 0)
EPS = 1e-9


def _log_cdf_products(cdf: np.ndarray):
    """Per-column product over all rows of cdf, with zero bookkeeping."""
    with np.errstate(divide="ignore"):
        logc = np.where(cdf > 0, np.log(np.maximum(cdf, 1e-300)), 0.0)
    zeros = (cdf <= 0).sum(axis=0)
    return logc, logc.sum(axis=0), zeros


def _prod_others(logc, logsum, zeros, cdf_row, logc_row):
    """prod_{rows != r} cdf[row, x] for one row r (prefix-sum trick)."""
    out = np.exp(logsum - logc_row)
    out[(zeros - (cdf_row <= 0)) > 0] = 0.0
    return out


def _beats_all(own: np.ndarray, others: np.ndarray) -> np.ndarray:
    """sum_x own[r, x] * prod_{r' != r} P(others[r'] < x), for every row r."""
    cdf = np.cumsum(others, axis=1) - others            # P(c' < x)
    logc, logsum, zeros = _log_cdf_products(cdf)
    out = np.empty(len(own))
    for r in range(len(own)):
        out[r] = np.dot(own[r], _prod_others(logc, logsum, zeros, cdf[r], logc[r]))
    return out


def prob_ind(space, idx=None):
    """(LB, UB) of the winning probabilities of the alive candidate sets (ProbInd)."""
    idx = space.alive_idx() if idx is None else idx
    lo, hi = space.lo[idx], space.hi[idx]
    return _beats_all(lo, hi), _beats_all(hi, lo)


def _max_of_others(v: np.ndarray) -> np.ndarray:
    """Per row, the maximum of the other columns."""
    order = np.argsort(v, axis=1)
    top, second = order[:, -1], order[:, -2]
    rows = np.arange(len(v))
    out = np.repeat(v[rows, top][:, None], v.shape[1], axis=1)
    out[rows, top] = v[rows, second]
    return out


class ProbDep:
    """(LB, UB) of the winning probabilities of the alive candidate sets (ProbDep).

    The issued calls are drawn jointly from their PMFs (a shared call is drawn
    once, so the dependence between candidate sets is kept) and a bound is the
    fraction of draws in which the comparison events of the set hold against
    every competitor.  The draws are fixed for the whole run (common random
    numbers) and the per-draw scores are updated incrementally: when a call
    is issued only the sets containing it change.

    The cancellation of shared unknown calls needs the number of unknown
    calls shared by every pair of sets; above ``max_pairwise`` alive sets it
    is skipped, which keeps the bounds valid and only makes them looser.
    """

    def __init__(self, space, n_samples: int = 2000, seed: int = 0, max_pairwise: int = 300):
        self.space, self.n, self.max_pairwise = space, n_samples, max_pairwise
        n_calls = int(space.pred_ids.max()) + 1
        self.u = np.random.default_rng(seed).random((n_samples, n_calls))
        self.vals = np.zeros((n_samples, n_calls))
        self.scores = np.zeros((n_samples, space.M))      # sum of the issued calls of every set
        for q in space.known:
            self.update(int(q))

    def update(self, q: int):
        """Call q has been issued: draw it and add it to the sets that contain it."""
        grid = np.arange(P.GRID_M) * P.STEP
        cdf = np.cumsum(self.space.known[q])
        new = grid[np.minimum(np.searchsorted(cdf, self.u[:, q], side="right"), P.GRID_M - 1)]
        cols = self.space.cands_of.get(q)
        if cols is not None:
            self.scores[:, cols] += (new - self.vals[:, q])[:, None]
        self.vals[:, q] = new

    def __call__(self, idx=None):
        space = self.space
        idx = space.alive_idx() if idx is None else idx
        if len(idx) == 1:
            return np.ones(1), np.ones(1)
        low = np.round(self.scores[:, idx], 6)                       # unknown calls at MIN
        slack = space.n_unknown[idx] * MAX_SCORE
        high = low + slack                                           # unknown calls at MAX
        if len(idx) > self.max_pairwise:
            lb = (low > _max_of_others(high) + EPS).mean(axis=0)
            ub = (high > _max_of_others(low) + EPS).mean(axis=0)
            return lb, ub
        mask = space.unknown_mask(idx).astype(np.float32)
        shared = (mask @ mask.T) * MAX_SCORE                         # unknown calls shared by two sets
        lb, ub = np.empty(len(idx)), np.empty(len(idx))
        for r in range(len(idx)):
            rival_high = high - shared[r]
            rival_high[:, r] = -np.inf
            lb[r] = (low[:, r] > rival_high.max(axis=1) + EPS).mean()
            rival_low = low + shared[r]
            rival_low[:, r] = -np.inf
            ub[r] = (high[:, r] > rival_low.max(axis=1) + EPS).mean()
        return lb, ub


def prob_dep(space, n_samples: int = 2000, seed: int = 0, idx=None):
    """One-off ProbDep evaluation (see class ProbDep)."""
    return ProbDep(space, n_samples, seed)(idx)
