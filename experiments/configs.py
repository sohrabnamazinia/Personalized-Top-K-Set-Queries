"""Settings of the experiments.

An experiment is a list of jobs; a job is one run of one algorithm on one
query.  Defaults: pools of n = 200 items, x = 3, theta = 0.999, candidate
sets over the 16 items with the largest upper-bound score.
"""
from __future__ import annotations

from itertools import product

# modality -> (model, provider); the figures use the first model of each family
LLMS = {"gpt-6-luna": ("gpt-6-luna", "openai"),
        "llama-4-scout": ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "deepinfra")}
VLMS = {"qwen3-vl": ("Qwen/Qwen3-VL-235B-A22B-Instruct", "deepinfra"),
        "mistral-large": ("mistral-large-latest", "mistral")}

DEFAULT = dict(n=200, x=3, theta=0.999, eps=None, top_items=16, tau=None, layout="dependent")
QUERIES = list(range(10))
QUERIES5 = list(range(5))

OURS = ["entrred+probind", "entrred+probdep"]
ALL_ALGOS = ["random", "greedy", "lotus", "tourrank"] + OURS


def _jobs(exp, F_list, k_list, algos, queries=QUERIES, **over):
    jobs = []
    for F, k, algo, q in product(F_list, k_list, algos, queries):
        job = dict(DEFAULT, exp=exp, F=F, k=k, algo=algo, seed=q)
        job.update(over)
        jobs.append(job)
    return jobs


def x1():   # predicate calls vs k
    return _jobs("X1", ["F1", "F3", "F5"], [3, 5, 7], ALL_ALGOS)


def x3():   # number of predicates
    return _jobs("X3", ["F1", "F1+", "F1++", "F5", "F5+", "F5++"], [5],
                 ["greedy", "lotus"] + OURS, queries=QUERIES5)


def x4():   # variability of the responses
    jobs = []
    for eps in [0.0, 0.02, 0.05, 0.1, 0.2]:
        jobs += _jobs("X4", ["F1", "F5"], [5], ["greedy", "entrred+probdep"], queries=QUERIES5, eps=eps)
    return jobs


def x5():   # threshold theta
    jobs = []
    for theta in [0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
        jobs += _jobs("X5", ["F1", "F3"], [7], OURS, queries=QUERIES5, theta=theta)
    return jobs


def x8():   # bound-based vs threshold pruning
    jobs = []
    for tau in [None, 0.96, 0.92, 0.88, 0.82]:
        jobs += _jobs("X8", ["F1", "F3", "F5"], [5], ["entrred+probdep"], queries=QUERIES5, tau=tau)
    return jobs


EXPERIMENTS = {"X1": x1, "X3": x3, "X4": x4, "X5": x5, "X8": x8}
