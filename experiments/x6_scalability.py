"""X6: running time of each component vs. the number of candidate sets M.

Candidate sets over 30 items (k = 5, half of the calls known), M from 100
to 100,000.  Measures a bound update, the winning probabilities with ProbInd
and with ProbDep, and the selection of the next call with EntrRed (on the
ProbInd probabilities) and with Greedy; three repeats, results/X6.csv.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bestset import pmf as P                                        # noqa: E402
from bestset.candidates import CandidateSpace                       # noqa: E402
from bestset.nextcall import entr_red, greedy                       # noqa: E402
from bestset.probability import ProbDep, prob_ind                   # noqa: E402
from bestset.scoring import SCORING                                 # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
SIZES = [100, 1000, 10000, 100000]


def build(M, k, rng, known_frac=0.5):
    F = SCORING["F1"]
    n_items = 30                                   # C(30, 5) = 142,506 sets to draw from
    key_id = {}
    for i in range(n_items):
        key_id[(0, i)] = len(key_id)
    for i, j in combinations(range(n_items), 2):
        key_id[(1, i, j)] = len(key_id)
    known = {}
    for q in range(len(key_id)):
        if rng.random() < known_frac:
            known[q] = P.pmf_from_samples(np.clip(rng.normal(rng.random(), 0.05, 3), 0, 1))
    all_sets = list(combinations(range(n_items), k))
    pick = rng.choice(len(all_sets), M, replace=False)
    sets = [all_sets[i] for i in sorted(pick)]
    return CandidateSpace(F, np.arange(n_items), k, known, key_id, sets)


def main(out=RESULTS, repeats=3, k=5):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "X6.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "k", "repeat", "component", "time_s"])
        for M in SIZES:
            for rep in range(repeats):
                rng = np.random.default_rng(rep)
                space = build(M, k, rng)
                dep = ProbDep(space, 2000, seed=rep)
                rows = []
                q = int(rng.choice(space.pred_ids[0]))      # issue one call of the first set
                space.known[q] = P.pmf_from_samples([0.7, 0.8, 0.7])
                t0 = time.perf_counter(); space.update(q); rows.append(("bounds", time.perf_counter() - t0))
                t0 = time.perf_counter(); p_ind = prob_ind(space); rows.append(("probind", time.perf_counter() - t0))
                t0 = time.perf_counter(); dep.update(q); dep(); rows.append(("probdep", time.perf_counter() - t0))
                t0 = time.perf_counter(); entr_red(space, p_ind, rng); rows.append(("entrred", time.perf_counter() - t0))
                t0 = time.perf_counter(); greedy(space, p_ind, rng); rows.append(("greedy", time.perf_counter() - t0))
                for comp, t in rows:
                    w.writerow([M, k, rep, comp, round(t, 6)])
                print(M, rep, {c: round(t, 4) for c, t in rows}, flush=True)


if __name__ == "__main__":
    main()
