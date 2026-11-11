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
are built from).
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
    """The set Exhaustive returns on the recorded responses, restricted to the
    candidate sets of the run under evaluation (the k-subsets of ``items``,
    or ``sets`` when given).  Only the calls these sets depend on are read."""

    def __init__(self, F, pool, responder, k, items, sets=None):
        n = pool.n
        self.rel = _unary_scores(F, responder, range(n), n)
        if sets is None:
            pairs = list(combinations(sorted(int(i) for i in items), 2))
        else:
            pairs = sorted({pair for s in sets for pair in combinations(sorted(int(i) for i in s), 2)})
        self.div = _binary_scores(F, responder, pairs, n)
        if sets is None:
            self.best_set = _best_subset(self.rel, self.div, items, k)
        else:
            self.best_set = max((tuple(int(i) for i in s) for s in sets),
                                key=lambda s: set_score(self.rel, self.div, s))
        self.best_score = set_score(self.rel, self.div, self.best_set)

    def agrees(self, s) -> bool:
        """True if s has the score of the best set (sets with equal scores are the same answer)."""
        return set_score(self.rel, self.div, s) >= self.best_score - 1e-9


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
