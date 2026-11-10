"""Candidate sets, score bounds and pruning.

After the unary predicates have been evaluated on the pool, the candidate
sets are the k-subsets of the n' items with the largest upper-bound score.
A candidate set keeps the sum-convolution of the PMFs of its predicate calls
(a call that has not been issued has a uniform PMF); its lower and upper
bound are the smallest and the largest score with non-zero mass.

Bound-based pruning removes a set whose upper bound is below the largest
lower bound.  Threshold pruning (``tau``) also removes the sets whose upper
bound is below tau times the largest upper bound.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import pmf as P


def n_candidate_items(k: int, n: int, top_items: int = 16) -> int:
    """Number of items the candidate sets are built from (at least k, at most n)."""
    return max(k, min(n, top_items))


class CandidateSpace:
    """Candidate sets over ``items``, with incrementally maintained PMFs and bounds."""

    def __init__(self, F, items: np.ndarray, k: int, known: dict, key_id: dict, sets=None):
        self.F, self.k = F, k
        self.items = np.asarray(items)
        if sets is None:
            sets = list(combinations(sorted(self.items.tolist()), k))
        self.sets = np.array(sets, dtype=int)
        self.M = len(self.sets)
        # ids of the predicate calls of every candidate set
        self.key_id = key_id
        ids = []
        for s in self.sets:
            row = []
            for pi, p in enumerate(F.predicates):
                if p.arity == 1:
                    row += [key_id[(pi, i)] for i in s]
                else:
                    row += [key_id[(pi, i, j)] for i, j in combinations(s, 2)]
            ids.append(row)
        self.pred_ids = np.array(ids, dtype=int)
        self.n_pred = self.pred_ids.shape[1]
        # call id -> candidate sets that contain it
        self.cands_of = {}
        for ci, row in enumerate(self.pred_ids):
            for q in row:
                self.cands_of.setdefault(int(q), []).append(ci)
        self.cands_of = {q: np.array(v) for q, v in self.cands_of.items()}
        self.L = (P.GRID_M - 1) * self.n_pred + 1
        self.pmfs = np.zeros((self.M, self.L))
        self.alive = np.ones(self.M, dtype=bool)
        self.known = known                       # call id -> PMF of the calls issued so far
        self._unif = P.uniform_pmf()
        for ci in range(self.M):
            self.pmfs[ci] = self._score_pmf(ci)
        self._refresh_bounds()

    def pred_pmf(self, q: int):
        return self.known.get(q, self._unif)

    def _score_pmf(self, ci):
        return P.convolve_all([self.pred_pmf(int(q)) for q in self.pred_ids[ci]])

    def _refresh_bounds(self):
        nz = self.pmfs > 0
        self.lb = nz.argmax(axis=1) * P.STEP
        self.ub = (self.L - 1 - nz[:, ::-1].argmax(axis=1)) * P.STEP

    def update(self, q: int):
        """Call q has been issued: refresh the sets that contain it."""
        for ci in self.cands_of.get(q, []):
            if self.alive[ci]:
                self.pmfs[ci] = self._score_pmf(ci)
        self._refresh_bounds()

    def prune(self, tau: float | None = None) -> int:
        """Bound-based pruning, and threshold pruning if tau is given; returns the number of sets removed."""
        before = self.alive.sum()
        idx = self.alive_idx()
        leader = idx[np.argmax(self.lb[idx])]
        max_lb = self.lb[leader]
        # ties are resolved in favour of the set with the largest lower bound
        self.alive &= self.ub > max_lb
        self.alive[leader] = True
        if tau is not None:
            self.alive &= self.ub >= tau * self.ub[self.alive].max()
        return int(before - self.alive.sum())

    def unknown_preds(self, ci) -> list:
        return [int(q) for q in self.pred_ids[ci] if int(q) not in self.known]

    def alive_idx(self):
        return np.flatnonzero(self.alive)

    def means(self):
        return self.pmfs @ (np.arange(self.L) * P.STEP)
