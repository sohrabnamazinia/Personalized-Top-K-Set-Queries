"""Best set retrieval.

1. Issue every unary call on the n items of the pool.
2. Build the candidate space: drop the items of sets whose upper bound is below
   the lower bound of another set, and take the k-subsets of the kept items.
3. Repeat: compute the bounds LB(c), UB(c) on the winning probabilities
   (ProbInd or ProbDep); return a candidate set as soon as LB(c) >= theta;
   stop without an answer when UB(c) < theta for every set; otherwise select
   the next call (EntrRed, Greedy or Random), issue it, update the score
   PMFs and bounds, prune.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import pmf as P
from .candidates import CandidateSpace, n_candidate_items, viable_items
from .nextcall import SELECTORS
from .probability import ProbDep, prob_ind


@dataclass
class RunResult:
    best_set: tuple            # empty when no candidate set reaches theta
    calls: int
    iterations: int
    M: int
    n_items: int
    time_s: float
    final_prob: float          # lower bound of the winning probability of the returned set
    timings: dict = field(default_factory=dict)
    pruned_final: int = 0


class BestSetRetrieval:
    def __init__(self, F, pool, responder, k, theta=0.999, prob="dep", selector="entrred",
                 top_items=100, tau=None, seed=0, dep_samples=2000,
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
        lb, ub = np.zeros(n), np.zeros(n)
        if hasattr(self.responder, "prefetch"):
            self.responder.prefetch([(pi, i) for pi, p in enumerate(self.F.predicates)
                                     if p.arity == 1 for i in range(n)])
        for pi, p in enumerate(self.F.predicates):
            if p.arity == 1:
                for i in range(n):
                    q = self._id((pi, i))
                    self._ask(q)
                    lo, hi = P.bounds(self.known[q])
                    lb[i] += lo
                    ub[i] += hi
        self.timings["unary"] += time.perf_counter() - t0
        return lb, ub

    def _build_space(self, rel_lb, rel_ub):
        order = np.lexsort((self.rng.random(self.pool.n), -rel_ub))   # ties broken at random
        if self.layout == "independent":
            n_top = (self.pool.n // self.k) * self.k
            items = order[:n_top]
            sets = [sorted(items[i:i + self.k].tolist()) for i in range(0, n_top, self.k)]
            items = np.sort(items)
        else:
            n_top = n_candidate_items(self.k, self.pool.n, self.top_items)
            viable = set(viable_items(rel_lb, rel_ub, self.k, self._n_binary()).tolist())
            kept = [i for i in order.tolist() if i in viable]
            items = np.sort(np.array(kept[:max(n_top, self.k)], dtype=order.dtype))
            sets = None
        for pi, p in enumerate(self.F.predicates):
            if p.arity == 2:
                for i, j in combinations(items.tolist(), 2):
                    self._id((pi, i, j))
        return CandidateSpace(self.F, items, self.k, self.known, self.key_id, sets)

    def _n_binary(self):
        return sum(p.arity == 2 for p in self.F.predicates)

    def _bounds(self, space):
        if self.prob_name == "dep":
            return self._dep()
        return prob_ind(space)

    def run(self) -> RunResult:
        t_start = time.perf_counter()
        space = self._build_space(*self._evaluate_unary())
        self.space = space
        if self.prob_name == "dep":
            self._dep = ProbDep(space, self.dep_samples, seed=int(self.rng.integers(1 << 31)))
        space.prune(self.tau)
        select = SELECTORS[self.selector_name]
        it, best = 0, None
        while True:
            t0 = time.perf_counter()
            lb, ub = self._bounds(space)
            self.timings["prob"] += time.perf_counter() - t0
            idx = space.alive_idx()
            top = int(np.argmax(lb))
            if lb[top] >= self.theta:                 # certified
                best = top
                break
            if not (ub >= self.theta).any() or it >= self.max_iterations:
                break                                 # no candidate set can reach theta
            t0 = time.perf_counter()
            q = select(space, (lb, ub), self.rng, self.theta)
            self.timings["next"] += time.perf_counter() - t0
            if q < 0:                                 # no unknown call is left
                break
            self._ask(q)
            t0 = time.perf_counter()
            space.update(q)
            if self.prob_name == "dep":
                self._dep.update(q)
            space.prune(self.tau)
            self.timings["bounds"] += time.perf_counter() - t0
            it += 1
        best_set = () if best is None else tuple(int(i) for i in space.sets[idx[best]])
        final = float(lb[top])
        return RunResult(best_set, self.responder.calls, it, space.M, len(space.items),
                         time.perf_counter() - t_start, final, dict(self.timings),
                         int((~space.alive).sum()))
