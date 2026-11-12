"""Baselines.

Exhaustive  issues every unary and every binary call on the pool and returns
            the candidate set with the largest score.
LOTUS       semantic-operator pipeline: sem_map evaluates the unary
            predicates on every item, a comparison-based sem_topk keeps the
            n' most relevant items, sem_join evaluates the binary predicates
            on every pair of them; the best set is computed from the results.
TourRank    tournament ranking: groups of g items, the top half of each
            group advances until n' items remain; binary predicates on every
            pair of the remaining items.

Every baseline returns (best set, number of calls, items its candidate sets
are built from).  ``Reference`` holds the ground truth of a query under full
information and the winning probability of any set against it.
"""
from __future__ import annotations

from itertools import combinations
from math import ceil, log2

import numpy as np

from . import pmf as P
from .candidates import n_candidate_items


def _score(responder, key) -> float:
    return P.mean(P.pmf_from_samples(responder.call(key)))


def _unary_scores(F, responder, items, n):
    rel = np.zeros(n)
    for pi, p in enumerate(F.predicates):
        if p.arity == 1:
            for i in items:
                rel[i] += _score(responder, (pi, int(i)))
    return rel


def _binary_scores(F, responder, pairs, n):
    pairs = list(pairs)
    div = np.zeros((n, n))
    for pi, p in enumerate(F.predicates):
        if p.arity == 2:
            for i, j in pairs:
                i, j = sorted((int(i), int(j)))
                div[i, j] = div[j, i] = div[i, j] + _score(responder, (pi, i, j))
    return div


def set_score(rel, div, s) -> float:
    return float(rel[list(s)].sum() + sum(div[i, j] for i, j in combinations(sorted(s), 2)))


def _best_subset(rel, div, items, k):
    best = max(combinations(sorted(int(i) for i in items), k), key=lambda s: set_score(rel, div, s))
    return tuple(best)


class Reference:
    """Ground truth of a query on the recorded responses, with every call of the
    candidate sets issued (the k-subsets of ``items``, or ``sets`` when given).

    ``best_set`` is the set Exhaustive returns.  The winning probability of a
    set is the probability that its score is strictly larger than the score of
    every other candidate set, under the joint distribution of the response
    PMFs (a call shared by several sets is one random variable); it is
    evaluated on ``n_samples`` joint draws.
    """

    def __init__(self, F, pool, responder, k, items, sets=None, n_samples=2000, seed=0):
        n = pool.n
        self.F, self.responder, self.n_samples = F, responder, n_samples
        self.rel = _unary_scores(F, responder, range(n), n)
        if sets is None:
            self.sets = list(combinations(sorted(int(i) for i in items), k))
        else:
            self.sets = [tuple(sorted(int(i) for i in s)) for s in sets]
        pairs = sorted({pair for s in self.sets for pair in combinations(s, 2)})
        self.div = _binary_scores(F, responder, pairs, n)
        self.best_set = max(self.sets, key=lambda s: set_score(self.rel, self.div, s))
        self.best_score = set_score(self.rel, self.div, self.best_set)
        self._rng = np.random.default_rng(seed)
        self._draws, self._scores = {}, None

    def _call_draws(self, key) -> np.ndarray:
        """Joint draws of one call (drawn once, reused by every set that contains it)."""
        if key not in self._draws:
            cdf = np.cumsum(P.pmf_from_samples(self.responder.call(key)))
            u = self._rng.random(self.n_samples)
            self._draws[key] = np.minimum(np.searchsorted(cdf, u, side="right"), P.GRID_M - 1) * P.STEP
        return self._draws[key]

    def _set_draws(self, s) -> np.ndarray:
        s = sorted(int(i) for i in s)
        out = np.zeros(self.n_samples)
        for pi, p in enumerate(self.F.predicates):
            keys = [(pi, i) for i in s] if p.arity == 1 else [(pi, i, j) for i, j in combinations(s, 2)]
            for key in keys:
                out = out + self._call_draws(key)
        return np.round(out, 6)

    def _all_draws(self) -> np.ndarray:
        if self._scores is None:
            self._scores = np.stack([self._set_draws(s) for s in self.sets], axis=1)
        return self._scores

    def winning_probability(self, s) -> float:
        s = tuple(sorted(int(i) for i in s))
        rivals = self._all_draws()
        if s in self.sets:
            rivals = np.delete(rivals, self.sets.index(s), axis=1)
        if rivals.shape[1] == 0:
            return 1.0
        return float((self._set_draws(s) > rivals.max(axis=1) + 1e-9).mean())

    def qualifying_set(self, theta) -> tuple:
        """The candidate set whose winning probability is at least theta, or () if there is none."""
        p = self.winning_probability(self.best_set)
        return self.best_set if p >= theta else ()

    def agrees(self, s, theta) -> bool:
        """True if s is a correct answer at theta: its winning probability is at
        least theta, or no set is returned and no candidate set reaches theta."""
        if len(s) == 0:
            return self.qualifying_set(theta) == ()
        return self.winning_probability(s) >= theta


def exhaustive(F, pool, responder, k, top_items=100, **_):
    n = pool.n
    if hasattr(responder, "prefetch"):
        responder.prefetch([(pi, i, j) for pi, p in enumerate(F.predicates) if p.arity == 2
                            for i, j in combinations(range(n), 2)])
    rel = _unary_scores(F, responder, range(n), n)
    div = _binary_scores(F, responder, combinations(range(n), 2), n)
    items = np.argsort(-rel, kind="stable")[:n_candidate_items(k, n, top_items)]
    return _best_subset(rel, div, items, k), responder.calls, items


def lotus(F, pool, responder, k, top_items=100, **_):
    n = pool.n
    n_top = n_candidate_items(k, n, top_items)
    rel = _unary_scores(F, responder, range(n), n)                 # sem_map
    # sem_topk compares items with model calls, about n * log2(n') comparisons;
    # they are charged here and the ranking is read from the sem_map scores
    if n_top < n:
        responder.calls += n * ceil(log2(max(n_top, 2))) * responder.x
    items = np.argsort(-rel, kind="stable")[:n_top]
    div = _binary_scores(F, responder, combinations(sorted(items.tolist()), 2), n)   # sem_join
    return _best_subset(rel, div, items, k), responder.calls, items


def tourrank(F, pool, responder, k, top_items=100, g=10, seed=0, **_):
    n = pool.n
    n_top = n_candidate_items(k, n, top_items)
    rng = np.random.default_rng(seed)
    alive = list(range(n))
    rel = np.zeros(n)
    while True:
        rng.shuffle(alive)
        winners = []
        for start in range(0, len(alive), g):
            group = alive[start:start + g]
            rel[group] = _unary_scores(F, responder, group, n)[group]
            winners += sorted(group, key=lambda i: -rel[i])[:max(ceil(len(group) / 2), 1)]
        if len(winners) < n_top:               # last round: the n' best items of this round
            alive = sorted(alive, key=lambda i: -rel[i])[:n_top]
            break
        alive = winners
        if len(alive) == n_top:
            break
    items = np.array(alive)
    div = _binary_scores(F, responder, combinations(sorted(items.tolist()), 2), n)
    return _best_subset(rel, div, items, k), responder.calls, items


BASELINES = {"exhaustive": exhaustive, "lotus": lotus, "tourrank": tourrank}
