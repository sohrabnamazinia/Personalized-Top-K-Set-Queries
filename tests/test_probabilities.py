"""Winning-probability bounds of ProbInd and ProbDep on a small candidate space."""
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bestset import pmf as P                                # noqa: E402
from bestset.candidates import CandidateSpace               # noqa: E402
from bestset.probability import prob_dep, prob_ind          # noqa: E402
from bestset.scoring import SCORING                         # noqa: E402


def _space(rng, known_frac, n_items=7, k=3):
    F, items = SCORING["F1"], np.arange(n_items)
    key_id = {}
    for i in items:
        key_id[(0, int(i))] = len(key_id)
    for i, j in combinations(items.tolist(), 2):
        key_id[(1, i, j)] = len(key_id)
    known = {}
    for q in key_id.values():
        if rng.random() < known_frac:
            known[q] = P.pmf_from_samples(np.clip(rng.normal(rng.random(), 0.08, 3), 0, 1))
    return CandidateSpace(F, items, k, known, key_id), key_id


def _strict_wins(space, values):
    scores = np.round(np.stack([values[:, space.pred_ids[c]].sum(1) for c in range(space.M)], 1), 6)
    top2 = np.sort(scores, axis=1)[:, -2:]
    return (scores >= top2[:, 1:] - 1e-9) & (top2[:, 1:] > top2[:, :1] + 1e-9)


def test_bounds_enclose_the_winning_probability(seed=0, draws=100000):
    rng = np.random.default_rng(seed)
    space, key_id = _space(rng, 0.6)
    lb, ub = prob_dep(space, 20000)
    assert (lb <= ub + 1e-9).all() and lb.sum() <= 1 + 1e-6

    # whatever the unknown calls turn out to be, the winning probability stays within the bounds
    grid = np.arange(P.GRID_M) * P.STEP
    for fill in (0.0, 0.5, 1.0, None):
        cols = []
        for q in range(len(key_id)):
            if q in space.known:
                cols.append(rng.choice(grid, draws, p=space.known[q]))
            else:
                cols.append(np.full(draws, fill) if fill is not None else rng.choice(grid, draws))
        p = _strict_wins(space, np.stack(cols, 1)).mean(0)
        assert (p >= lb - 0.02).all() and (p <= ub + 0.02).all()

    ind_lb, ind_ub = prob_ind(space)
    assert (ind_lb <= ind_ub + 1e-9).all()


def test_bounds_meet_under_full_information(seed=1, draws=200000):
    rng = np.random.default_rng(seed)
    space, key_id = _space(rng, 1.0)
    lb, ub = prob_dep(space, 20000)
    assert np.abs(lb - ub).max() < 1e-9
    grid = np.arange(P.GRID_M) * P.STEP
    values = np.stack([rng.choice(grid, draws, p=space.known[q]) for q in range(len(key_id))], 1)
    assert np.abs(lb - _strict_wins(space, values).mean(0)).max() < 0.02
    ind_lb, ind_ub = prob_ind(space)
    assert np.abs(ind_lb - ind_ub).max() < 1e-9


if __name__ == "__main__":
    test_bounds_enclose_the_winning_probability()
    test_bounds_meet_under_full_information()
    print("ok")
