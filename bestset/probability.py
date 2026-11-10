"""Winning probabilities of the candidate sets.

The winning probability of c is the probability that no other candidate
set scores higher; an exact tie between two sets counts one half for each
(so P(c beats c') = P(F(c) > F(c')) + P(F(c) = F(c')) / 2).  Sets whose
scores are fully known and tied are resolved by the pruning rule.

ProbInd (independent candidate sets)
    P(c = c*) = sum_x P(F(c) = x) * prod_{c' != c} P(F(c') <= x)
    computed for all M sets at once from the prefix-sum CDFs (the product
    over the other sets is obtained by dividing the product over all sets).

ProbDep (dependent candidate sets, i.e. sets that share items)
    Candidate sets that share items share predicate calls, so their scores
    are dependent and the product form of ProbInd is only an approximation.
    ProbDep evaluates the joint event "c scores at least as high as every
    other set" under the joint distribution of all predicate values, in
    which every shared call is a single random variable; the computation is
    a sample-based evaluation of that joint distribution (common random
    numbers across iterations).  When no items are shared, the two agree.
"""
from __future__ import annotations

import numpy as np

from . import pmf as P


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


def prob_ind(space, idx=None) -> np.ndarray:
    """Winning probabilities of the alive candidate sets (ProbInd)."""
    idx = space.alive_idx() if idx is None else idx
    pm = space.pmfs[idx]
    cdf = np.cumsum(pm, axis=1) - 0.5 * pm          # P(c' < x) + P(c' = x) / 2
    logc, logsum, zeros = _log_cdf_products(cdf)
    out = np.empty(len(idx))
    for r in range(len(idx)):
        others = _prod_others(logc, logsum, zeros, cdf[r], logc[r])
        out[r] = np.dot(pm[r], others)
    return out


class ProbDep:
    """Winning probabilities of the alive candidate sets (ProbDep).

    The predicate values are drawn jointly from their PMFs (every shared
    call is drawn once, so the dependence between candidate sets that share
    items is kept exactly) and the winning probability of a set is the
    fraction of draws in which it holds the maximum score (ties split
    equally).  The draws are fixed for the whole run (common random numbers)
    and the per-draw scores are updated incrementally: when a call becomes
    known only the sets containing it change.
    """

    def __init__(self, space, n_samples: int = 2000, seed: int = 0):
        self.space, self.n = space, n_samples
        n_pred = int(space.pred_ids.max()) + 1
        self.u = np.random.default_rng(seed).random((n_samples, n_pred))
        self.vals = np.empty((n_samples, n_pred))
        self.version = {}
        for q in np.unique(space.pred_ids):
            self._draw(int(q))
        self.scores = np.zeros((n_samples, space.M))
        for j in range(space.pred_ids.shape[1]):
            self.scores += self.vals[:, space.pred_ids[:, j]]

    def _draw(self, q):
        grid = np.arange(P.GRID_M) * P.STEP
        cdf = np.cumsum(self.space.pred_pmf(q))
        self.vals[:, q] = grid[np.minimum(np.searchsorted(cdf, self.u[:, q], side="right"), P.GRID_M - 1)]
        self.version[q] = q in self.space.known

    def update(self, q: int):
        """Call q became known: redraw it and patch the sets that contain it."""
        old = self.vals[:, q].copy()
        self._draw(q)
        delta = self.vals[:, q] - old
        cols = self.space.cands_of.get(q)
        if cols is not None:
            self.scores[:, cols] += delta[:, None]

    def __call__(self, idx=None) -> np.ndarray:
        space = self.space
        idx = space.alive_idx() if idx is None else idx
        if len(idx) == 1:
            return np.ones(1)
        sc = np.round(self.scores[:, idx], 6)
        mx = sc.max(axis=1, keepdims=True)
        win = sc >= mx - 1e-9
        return (win / win.sum(axis=1, keepdims=True)).mean(axis=0)


def prob_dep(space, n_samples: int = 2000, seed: int = 0, idx=None) -> np.ndarray:
    """One-off ProbDep evaluation (see class ProbDep)."""
    return ProbDep(space, n_samples, seed)(idx)
