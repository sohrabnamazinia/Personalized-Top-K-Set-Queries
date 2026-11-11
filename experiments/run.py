"""Run an experiment: ``python -m experiments.run --exp X1``.

One row per job is appended to results/<exp>.csv; finished jobs are skipped
when the command is run again.  The raw responses of the predicate calls on
a pool are recorded in responses/<dataset>_q<query>_n<n>_<llm>_<vlm>.json and
shared by every algorithm and every experiment on that pool and these models.

Recall compares the returned set with the ground truth of the query: the best
candidate set of the framework (the candidate space built from the bounds
of the unary calls) on the recorded responses, with every call of these sets
issued.  LOTUS and TourRank select their own items, so their sets are scored
against the same ground truth.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from itertools import groupby
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bestset.baselines import BASELINES, Reference, set_score                      # noqa: E402
from bestset.data import load_pool                                                 # noqa: E402
from bestset.framework import BestSetRetrieval                                     # noqa: E402
from bestset.responses import ModelClient, RecordedResponder, ResponseStore, VariabilityResponder  # noqa: E402
from bestset.scoring import DATASET_OF, SCORING                                    # noqa: E402
from experiments.configs import EXPERIMENTS, LLMS, VLMS                            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
RESPONSES = os.path.join(ROOT, "responses")
COLUMNS = ["exp", "dataset", "F", "ell", "k", "x", "eps", "theta", "tau", "layout", "llm", "vlm", "M", "n_items", "algo",
           "seed", "calls", "recall", "time_s", "iterations", "final_prob", "best_set", "ref_set", "gap"]


def store_path(dataset, query, n, llm, vlm):
    return os.path.join(RESPONSES, f"{dataset}_q{query}_n{n}_{llm}_{vlm}.json")


def ground_truth_items(F, pool, responder, job):
    """Items of the framework's candidate sets for this query."""
    b = BestSetRetrieval(F, pool, responder, job["k"], top_items=job["top_items"], seed=job["seed"])
    return b._build_space(*b._evaluate_unary()).items


def make_responder(F, pool, store, job, models):
    r = RecordedResponder(F, pool, models, store, job["x"])
    return r if job["eps"] is None else VariabilityResponder(r, job["eps"], job["seed"])


def run_job(job, pool, store, models):
    F = SCORING[job["F"]]
    responder = make_responder(F, pool, store, job, models)
    t0 = time.perf_counter()
    algo, sets = job["algo"], None
    if algo in BASELINES:
        best, calls, _ = BASELINES[algo](F, pool, responder, job["k"], job["top_items"], seed=job["seed"])
        M, it, p = 0, 0, 1.0
        items = ground_truth_items(F, pool, make_responder(F, pool, store, job, models), job)
    else:
        selector, prob = algo.split("+") if "+" in algo else (algo, "probind")   # Random and Greedy stop with ProbInd
        b = BestSetRetrieval(F, pool, responder, job["k"], theta=job["theta"], prob=prob.replace("prob", ""),
                             selector=selector, top_items=job["top_items"], tau=job["tau"],
                             seed=job["seed"], layout=job["layout"])
        r = b.run()
        best, calls, M, it, p = r.best_set, r.calls, r.M, r.iterations, r.final_prob
        items = b.space.items
        if job["layout"] == "independent":
            sets = b.space.sets
    elapsed = time.perf_counter() - t0
    # recall: the ground truth on the same raw responses
    ref = Reference(F, pool, make_responder(F, pool, store, job, models), job["k"], items, sets)
    return dict(exp=job["exp"], dataset=pool.dataset, F=job["F"], ell=F.ell, k=job["k"], x=job["x"],
                eps="" if job["eps"] is None else job["eps"], theta=job["theta"],
                tau="" if job["tau"] is None else job["tau"], layout=job["layout"], llm=job["llm"], vlm=job["vlm"],
                M=M, n_items=len(items),
                algo=algo, seed=job["seed"], calls=calls, recall=int(ref.agrees(best)),
                time_s=round(elapsed, 3), iterations=it, final_prob=round(p, 4),
                best_set=" ".join(map(str, sorted(best))), ref_set=" ".join(map(str, ref.best_set)),
                gap=round(ref.best_score - set_score(ref.rel, ref.div, best), 4))


def run_pool(args):
    """All jobs on one pool and pair of models, in one process, so that the pool has one writer of its responses."""
    jobs = args
    dataset, query, n = DATASET_OF[jobs[0]["F"]], jobs[0]["seed"], jobs[0]["n"]
    llm, vlm = jobs[0]["llm"], jobs[0]["vlm"]
    pool = load_pool(dataset, n, query)
    store = ResponseStore(store_path(dataset, query, n, llm, vlm))
    models = {"text": ModelClient(*LLMS[llm]), "image": ModelClient(*VLMS[vlm])}
    return [run_job(job, pool, store, models) for job in jobs]


def _key(row):
    return tuple(str(row[c]) for c in ("exp", "F", "k", "x", "eps", "theta", "tau", "layout", "llm", "vlm", "algo", "seed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--jobs", type=int, default=4, help="pools processed in parallel")
    ap.add_argument("--n", type=int, default=None, help="pool size (default: configs.DEFAULT)")
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.exp}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {_key(row) for row in csv.DictReader(f)}
    jobs = EXPERIMENTS[args.exp]()
    if args.n:
        jobs = [dict(j, n=args.n) for j in jobs]
    blank = lambda j: dict(j, eps="" if j["eps"] is None else j["eps"], tau="" if j["tau"] is None else j["tau"])
    jobs = [j for j in jobs if _key(blank(j)) not in done]
    by_pool = lambda j: (DATASET_OF[j["F"]], j["seed"], j["llm"], j["vlm"])
    groups = [list(g) for _, g in groupby(sorted(jobs, key=by_pool), key=by_pool)]
    print(f"{args.exp}: {len(jobs)} jobs on {len(groups)} pools ({len(done)} done)", flush=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f, Pool(args.jobs) as mp:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for i, rows in enumerate(mp.imap_unordered(run_pool, groups), 1):
            w.writerows(rows); f.flush()
            print(f"  {i}/{len(groups)} pools", flush=True)


if __name__ == "__main__":
    main()
