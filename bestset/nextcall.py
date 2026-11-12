"""Choosing the next predicate call.

EntrRed  : a candidate set c is unresolved when LB(c) < theta <= UB(c).  Each
           unknown call q is scored with
               PEF(q) = sum of UB(c) - LB(c) over the unresolved sets c that contain q
           and the call with the largest score is issued next.  When no
           unresolved set has an unknown call, a call of another alive set is
           issued, since it still tightens the bounds of the unresolved sets.
Greedy   : a random unknown call of the candidate set with the largest upper bound.
Random   : a random unknown call among the alive candidate sets.

Every selector takes the candidate space, the bounds (LB, UB) of the alive
sets, the random generator and theta, and returns a call id (-1 if no
unknown call is left).
"""
from __future__ import annotations

import numpy as np


def entr_red(space, bounds, rng, theta) -> int:
    lb, ub = bounds
    idx = space.alive_idx()
    unresolved = (lb < theta) & (ub >= theta)
    score = {}
    for r in np.flatnonzero(unresolved):
        width = ub[r] - lb[r]
        for q in space.unknown_preds(int(idx[r])):
            score[q] = score.get(q, 0.0) + width
    if not score:
        return random_call(space, bounds, rng, theta)
    top = max(score.values())
    return int(rng.choice(sorted(q for q, s in score.items() if s >= top - 1e-12)))


def greedy(space, bounds, rng, theta) -> int:
    idx = space.alive_idx()
    for ci in idx[np.argsort(-space.ub[idx])]:
        unknown = space.unknown_preds(int(ci))
        if unknown:
            return int(rng.choice(unknown))
    return -1


def random_call(space, bounds, rng, theta) -> int:
    idx = space.alive_idx()
    pool = set()
    for ci in idx:
        pool.update(space.unknown_preds(int(ci)))
    return int(rng.choice(sorted(pool))) if pool else -1


SELECTORS = {"entrred": entr_red, "greedy": greedy, "random": random_call}
