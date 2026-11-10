"""Best set retrieval.

1. Issue every unary call on the n items of the pool.
2. Build the candidate sets over the n' items with the largest upper-bound score.
3. Repeat: compute the winning probabilities (ProbInd or ProbDep); stop when
   a candidate set reaches theta; otherwise select the next call (EntrRed,
   Greedy or Random), issue it, update the PMFs and bounds, prune.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import pmf as P
from .candidates import CandidateSpace, n_candidate_items
from .nextcall import SELECTORS
from .probability import ProbDep, prob_ind


@dataclass
class RunResult:
    best_set: tuple
    calls: int
    iterations: int
    M: int
    n_items: int
    time_s: float
    final_prob: float
    timings: dict = field(default_factory=dict)
    pruned_final: int = 0


class BestSetRetrieval:
    def __init__(self, F, pool, responder, k, theta=0.999, prob="dep", selector="entrred",
                 top_items=16, tau=None, seed=0, dep_samples=2000,
                 max_iterations=100000, layout="dependent"):
        self.F, self.pool, self.responder, self.k = F, pool, responder, k
        self.theta, self.prob_name, self.selector_name = theta, prob, selector
        self.top_items, self.tau = top_items, tau
        self.rng = np.random.default_rng(seed)
        self.dep_samples = dep_samples
        self.max_iterations = max_iterations
        self.layout = layout          # "dependent": all k-subsets of the items (sets share items)
                                      # "independent": the items partitioned into disjoint sets
        self.key_id, self.id_key = {}, []
        self.known = {}
        self.timings = {"unary": 0.0, "bounds": 0.0, "prob": 0.0, "next": 0.0}

    def _id(self, key):
        if key not in self.key_id:
            self.key_id[key] = len(self.id_key)
            self.id_key.append(key)
        return self.key_id[key]

    def _ask(self, q: int):
        self.known[q] = P.pmf_from_samples(self.responder.call(self.id_key[q]))

    def _evaluate_unary(self):
        t0 = time.perf_counter()
        n = self.pool.n
        ub = np.zeros(n)
        if hasattr(self.responder, "prefetch"):
            self.responder.prefetch([(pi, i) for pi, p in enumerate(self.F.predicates)
                                     if p.arity == 1 for i in range(n)])
        for pi, p in enumerate(self.F.predicates):
            if p.arity == 1:
                for i in range(n):
                    q = self._id((pi, i))
                    self._ask(q)
                    ub[i] += P.bounds(self.known[q])[1]
        self.timings["unary"] += time.perf_counter() - t0
        return ub

    def _build_space(self, rel_ub):
        order = np.lexsort((self.rng.random(self.pool.n), -rel_ub))   # ties broken at random
        if self.layout == "independent":
            n_top = (self.pool.n // self.k) * self.k
            items = order[:n_top]
            sets = [sorted(items[i:i + self.k].tolist()) for i in range(0, n_top, self.k)]
            items = np.sort(items)
        else:
            n_top = n_candidate_items(self.k, self.pool.n, self.top_items)
            items = np.sort(order[:n_top])
            sets = None
        for pi, p in enumerate(self.F.predicates):
            if p.arity == 2:
                for i, j in combinations(items.tolist(), 2):
                    self._id((pi, i, j))
        return CandidateSpace(self.F, items, self.k, self.known, self.key_id, sets)

    def _probs(self, space):
        if self.prob_name == "dep":
            return self._dep()
        return prob_ind(space)

    def run(self) -> RunResult:
        t_start = time.perf_counter()
        rel_ub = self._evaluate_unary()
        space = self._build_space(rel_ub)
        self.space = space
        if self.prob_name == "dep":
            self._dep = ProbDep(space, self.dep_samples, seed=int(self.rng.integers(1 << 31)))
        space.prune(self.tau)
        select = SELECTORS[self.selector_name]
        it, q = 0, 0
        while True:
            t0 = time.perf_counter()
            probs = self._probs(space)
            self.timings["prob"] += time.perf_counter() - t0
            idx = space.alive_idx()
            best = int(np.argmax(probs))
            if probs[best] >= self.theta or len(idx) == 1 or it >= self.max_iterations:
                break
            t0 = time.perf_counter()
            q = select(space, probs, self.rng)
            self.timings["next"] += time.perf_counter() - t0
            if q < 0:                    # no unknown call is left
                break
            self._ask(q)
            t0 = time.perf_counter()
            space.update(q)
            if self.prob_name == "dep":
                self._dep.update(q)
            space.prune(self.tau)
            self.timings["bounds"] += time.perf_counter() - t0
            it += 1
        if q < 0:
            # every call of the remaining sets has been issued: return the set
            # with the largest score (ties broken by the winning probability)
            means = space.means()[idx]
            top = np.flatnonzero(means >= means.max() - 1e-9)
            best = int(top[np.argmax(probs[top])])
        best_set = tuple(int(i) for i in space.sets[idx[best]])
        return RunResult(best_set, self.responder.calls, it, space.M, len(space.items),
                         time.perf_counter() - t_start, float(probs[best]), dict(self.timings),
                         int((~space.alive).sum()))
