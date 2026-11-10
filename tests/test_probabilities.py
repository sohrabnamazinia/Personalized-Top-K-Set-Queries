"""ProbInd and ProbDep against a Monte-Carlo estimate on a small candidate space."""
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bestset import pmf as P                                # noqa: E402
from bestset.candidates import CandidateSpace               # noqa: E402
from bestset.probability import prob_dep, prob_ind          # noqa: E402
from bestset.scoring import SCORING                         # noqa: E402


def test_winning_probabilities(seed=0, draws=200000):
    rng = np.random.default_rng(seed)
    F, k, items = SCORING["F1"], 3, np.arange(7)
    key_id = {}
    for i in items:
        key_id[(0, int(i))] = len(key_id)
    for i, j in combinations(items.tolist(), 2):
        key_id[(1, i, j)] = len(key_id)
    known = {}
    for q in key_id.values():
        if rng.random() < 0.6:
            known[q] = P.pmf_from_samples(np.clip(rng.normal(rng.random(), 0.08, 3), 0, 1))
    space = CandidateSpace(F, items, k, known, key_id)
    ind, dep = prob_ind(space), prob_dep(space, 20000)

    # joint distribution: a call shared by several sets is drawn once
    grid = np.arange(P.GRID_M) * P.STEP
    values = np.stack([rng.choice(grid, draws, p=known.get(q, P.uniform_pmf())) for q in range(len(key_id))], 1)
    scores = np.round(np.stack([values[:, space.pred_ids[c]].sum(1) for c in range(space.M)], 1), 6)
    win = scores >= scores.max(1, keepdims=True) - 1e-9
    mc = (win / win.sum(1, keepdims=True)).mean(0)          # ties split equally

    assert abs(dep.sum() - 1) < 1e-6
    assert np.abs(dep - mc).max() < 0.02
    # the sets share calls, so ProbDep is closer to the joint distribution than ProbInd
    assert np.abs(dep - mc).max() <= np.abs(ind - mc).max()


if __name__ == "__main__":
    test_winning_probabilities()
    print("ok")
