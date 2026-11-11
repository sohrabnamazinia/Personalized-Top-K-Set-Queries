"""Print the numbers reported in Section V from results/*.csv: ``python -m experiments.numbers``."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.plots import LLM_PRICE, VLM_PRICE, exhaustive_calls, load   # noqa: E402

pd.set_option("display.width", 200)


def x1(d):
    d = d[d.k <= 7]
    print("\n== X1: mean calls per dataset / k / algo")
    print(d.groupby(["dataset", "k", "algo"]).calls.mean().unstack().round(1).to_string())
    print("\n== X1: recall per algo / k")
    print(d.groupby(["algo", "k"]).recall.mean().unstack().round(3).to_string())
    ours = d[d.algo == "entrred+probdep"].groupby(["dataset", "k"]).calls.mean()
    ex = exhaustive_calls()
    lot = d[d.algo == "lotus"].calls.mean(); tr = d[d.algo == "tourrank"].calls.mean()
    print(f"\nExhaustive {ex}, LOTUS {lot:.0f}, TourRank {tr:.0f}")
    print("ratios vs ours: Exhaustive", (ex / ours).round(1).to_dict())
    print("ratios LOTUS", (lot / ours).round(2).to_dict())
    print("ratios TourRank", (tr / ours).round(2).to_dict())
    g = d[d.dataset == "hotels"].groupby(["k", "algo"]).calls.mean().unstack()
    print("\nHotels EntrRed+ProbDep vs Greedy:", g[["entrred+probdep", "greedy"]].round(1).to_dict("index"))
    print("recall all algos (k<=7):", d.groupby("algo").recall.mean().round(4).to_dict())


def x3(d):
    print("\n== X3: calls per ell (k=5)")
    print(d.groupby(["dataset", "ell", "algo"]).calls.mean().unstack().round(1).to_string())


def x4(d):
    print("\n== X4: calls per eps (k=5)")
    print(d.groupby(["dataset", "algo", "eps"]).calls.mean().unstack().round(1).to_string())
    print("\n== X4: recall per eps")
    print(d.groupby(["algo", "eps"]).recall.mean().unstack().round(3).to_string())


def x5(d):
    print("\n== X5: calls and recall per theta (k=7)")
    print(d.groupby(["dataset", "algo", "theta"]).calls.mean().unstack().round(1).to_string())
    print(d.groupby(["dataset", "algo", "theta"]).recall.mean().unstack().round(3).to_string())


def x8(d):
    print("\n== X8: recall and calls saved per tau (k=5, EntrRed+ProbDep)")
    d = d.copy(); d["tau"] = d["tau"].fillna(0).astype(float)
    base = d[d.tau == 0].groupby("dataset").calls.mean()
    for ds, g in d.groupby("dataset"):
        row = []
        for tau, h in g.groupby("tau"):
            saved = 100 * (base[ds] - h.calls.mean()) / base[ds]
            row.append(f"tau={tau}: recall {100*h.recall.mean():.0f}% saved {saved:.0f}%")
        print(ds, " | ".join(row))


def x6(d):
    print("\n== X6: mean time per component / M")
    print(d.groupby(["component", "M"]).time_s.mean().unstack().to_string())


def cost(d1, text_tokens=900, image_tokens=2000, out=8):
    print("\n== cost per query ($), rows F,k; columns per model: Greedy, LOTUS, EntrRed+ProbDep")
    for F, toks, price in [("F1", text_tokens, LLM_PRICE), ("F5", image_tokens, VLM_PRICE)]:
        for k in [3, 5, 7]:
            cells = []
            for model, (pin, pout) in price.items():
                for algo in ["greedy", "lotus", "entrred+probdep"]:
                    c = d1[(d1.F == F) & (d1.k == k) & (d1.algo == algo)].calls.mean()
                    cells.append(f"{c * (toks * pin + out * pout) / 1e6:.3f}")
            print(F, k, " ".join(cells))


def main():
    d = {e: load(e) for e in ["X1", "X3", "X4", "X5", "X6", "X8"]}
    for e, f in [("X1", x1), ("X3", x3), ("X4", x4), ("X5", x5), ("X8", x8), ("X6", x6)]:
        if d[e] is not None:
            f(d[e])
    if d["X1"] is not None:
        cost(d["X1"])


if __name__ == "__main__":
    main()
