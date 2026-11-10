"""Choosing the next predicate call.

EntrRed  : take the candidate set c+ with the largest winning probability,
           consider its unknown predicate calls P_U', and score each call q
           with PEF(q) = sum_{c in C_q} sum_{c' not in C_q} |P(c) - P(c')|
           where C_q are the candidate sets that contain q.  The call with the
           largest PEF is issued next.  If c+ has no unknown call, the next
           most probable set is used instead.
Greedy   : a random unknown call of the candidate set with the largest upper bound.
Random   : a random unknown call among the alive candidate sets.
"""
from __future__ import annotations

import numpy as np


def _abs_diff_sum(a: np.ndarray, b: np.ndarray) -> float:
    """sum_{i,j} |a_i - b_j| in O((|a|+|b|) log)"""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    b = np.sort(b)
    cb = np.concatenate([[0.0], np.cumsum(b)])
    pos = np.searchsorted(b, a, side="left")
    below = a * pos - cb[pos]
    above = (cb[-1] - cb[pos]) - a * (len(b) - pos)
    return float((below + above).sum())


def entr_red(space, probs: np.ndarray, rng) -> int:
    idx = space.alive_idx()
    alive_pos = {int(ci): r for r, ci in enumerate(idx)}
    for r in np.argsort(-probs):
        ci = int(idx[r])
        unknown = space.unknown_preds(ci)
        if not unknown:
            continue
        best, best_score = [], -1.0
        for q in unknown:
            cq = [alive_pos[int(c)] for c in space.cands_of[q] if space.alive[c]]
            mask = np.zeros(len(idx), dtype=bool)
            mask[cq] = True
            s = _abs_diff_sum(probs[mask], probs[~mask])
            if s > best_score + 1e-12:
                best, best_score = [q], s
            elif abs(s - best_score) <= 1e-12:
                best.append(q)
        return int(rng.choice(best))
    return -1


def greedy(space, probs, rng) -> int:
    idx = space.alive_idx()
    for ci in idx[np.argsort(-space.ub[idx])]:
        unknown = space.unknown_preds(int(ci))
        if unknown:
            return int(rng.choice(unknown))
    return -1


def random_call(space, probs, rng) -> int:
    idx = space.alive_idx()
    pool = set()
    for ci in idx:
        pool.update(space.unknown_preds(int(ci)))
    return int(rng.choice(sorted(pool))) if pool else -1


SELECTORS = {"entrred": entr_red, "greedy": greedy, "random": random_call}
