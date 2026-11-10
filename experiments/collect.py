"""Collect the responses of every predicate call on a pool (Exhaustive):

    python -m experiments.collect --F F1 --query 0

Issues the x responses of every unary and every binary call of the scoring
function and records them in responses/, where experiments.run reads them.
Calls that are already recorded are not sent again.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bestset.baselines import exhaustive                                   # noqa: E402
from bestset.data import load_pool                                         # noqa: E402
from bestset.responses import ModelClient, RecordedResponder, ResponseStore  # noqa: E402
from bestset.scoring import DATASET_OF, SCORING                            # noqa: E402
from experiments.configs import DEFAULT, LLMS, VLMS                        # noqa: E402
from experiments.run import store_path                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", required=True, choices=sorted(SCORING))
    ap.add_argument("--query", type=int, nargs="+", default=[0])
    ap.add_argument("--n", type=int, default=DEFAULT["n"])
    ap.add_argument("--x", type=int, default=DEFAULT["x"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--llm", default="gpt-6-luna", choices=sorted(LLMS))
    ap.add_argument("--vlm", default="qwen3-vl", choices=sorted(VLMS))
    ap.add_argument("--workers", type=int, default=16, help="concurrent calls")
    args = ap.parse_args()
    F, dataset = SCORING[args.F], DATASET_OF[args.F]
    models = {"text": ModelClient(*LLMS[args.llm]), "image": ModelClient(*VLMS[args.vlm])}
    for q in args.query:
        pool = load_pool(dataset, args.n, q)
        store = ResponseStore(store_path(dataset, q, args.n))
        responder = RecordedResponder(F, pool, models, store, args.x, args.workers)
        best, calls, _ = exhaustive(F, pool, responder, args.k)
        print(f"{args.F} query {q}: {calls} calls, best set {best}", flush=True)


if __name__ == "__main__":
    main()
