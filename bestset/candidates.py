"""Candidate sets, score bounds and pruning.

After the unary predicates have been evaluated on the pool, the candidate
space is built from the bounds of the unary calls: an item is dropped when
every set that contains it has an upper bound below the lower bound of
another set.  ``top_items`` bounds the number of items that are kept (those
with the largest upper bound); the candidate sets are their k-subsets.
A call that has not been issued has no PMF, only its range [MIN, MAX].  A
candidate set therefore keeps two score PMFs: the sum-convolution of the PMFs
of its issued calls with every other call at MIN (lower-bound score) and at
MAX (upper-bound score).  Its lower and upper bound are the smallest score of
the first and the largest score of the second.

Exact pruning removes a set whose upper bound is below the largest
lower bound.  Heuristic pruning (``tau``) also removes the sets whose upper
bound is below tau times the largest upper bound.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import pmf as P


def n_candidate_items(k: int, n: int, top_items: int = 100) -> int:
    """Number of items the candidate sets are built from (at least k, at most n)."""
    return max(k, min(n, top_items))


def viable_items(lb: np.ndarray, ub: np.ndarray, k: int, n_binary: int) -> np.ndarray:
    """Items that belong to a set whose upper bound is not below the largest lower bound.

    ``lb`` and ``ub`` are the bounds of the items after the unary calls; every
    binary call is unknown and contributes between 0 and 1 per pair.
    """
    best_lb = np.sort(lb)[-k:].sum()
    slack = n_binary * k * (k - 1) / 2
    order = np.argsort(-ub, kind="stable")
    top = ub[order[:k]].sum()
    in_top = np.zeros(len(ub), dtype=bool)
    in_top[order[:k]] = True
    # largest upper bound of a set that contains item i
    best_with = np.where(in_top, top, top - ub[order[k - 1]] + ub)
    return np.flatnonzero(best_with + slack >= best_lb - 1e-9)


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
        self.lo = np.zeros((self.M, self.L))     # PMF of the lower-bound score
        self.hi = np.zeros((self.M, self.L))     # PMF of the upper-bound score
        self.n_unknown = np.zeros(self.M, dtype=int)
        self.alive = np.ones(self.M, dtype=bool)
        self.known = known                       # call id -> PMF of the calls issued so far
        for ci in range(self.M):
            self._set_score_pmfs(ci)
        self._refresh_bounds()

    def _set_score_pmfs(self, ci):
        issued = [self.known[int(q)] for q in self.pred_ids[ci] if int(q) in self.known]
        pmf = P.convolve_all(issued)
        self.n_unknown[ci] = self.n_pred - len(issued)
        shift = (P.GRID_M - 1) * self.n_unknown[ci]          # every unknown call at MAX
        self.lo[ci] = 0.0
        self.hi[ci] = 0.0
        self.lo[ci, :len(pmf)] = pmf
        self.hi[ci, shift:shift + len(pmf)] = pmf

    def _refresh_bounds(self):
        self.lb = (self.lo > 0).argmax(axis=1) * P.STEP
        self.ub = (self.L - 1 - (self.hi > 0)[:, ::-1].argmax(axis=1)) * P.STEP

    def update(self, q: int):
        """Call q has been issued: refresh the sets that contain it."""
        for ci in self.cands_of.get(q, []):
            if self.alive[ci]:
                self._set_score_pmfs(ci)
        self._refresh_bounds()

    def prune(self, tau: float | None = None) -> int:
        """Exact pruning, and heuristic pruning if tau is given; returns the number of sets removed."""
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

    def unknown_mask(self, idx) -> np.ndarray:
        """Boolean matrix (sets in idx) x (calls): the unknown calls of each set."""
        n_calls = int(self.pred_ids.max()) + 1
        issued = np.zeros(n_calls, dtype=bool)
        issued[list(self.known)] = True
        mask = np.zeros((len(idx), n_calls), dtype=bool)
        rows = np.repeat(np.arange(len(idx)), self.n_pred)
        cols = self.pred_ids[idx].ravel()
        mask[rows, cols] = ~issued[cols]
        return mask
