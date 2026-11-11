"""Figures and tables from results/*.csv: ``python -m experiments.plots``.

Writes figures/fig1_calls_vs_k.pdf, fig3_predicates.pdf, fig4_uncertainty.pdf,
fig5_theta.pdf, fig6_scalability.pdf, fig7_recall_theta.pdf and prints the
rows of the cost, model and pruning tables as LaTeX.
"""
from __future__ import annotations

import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS, FIGURES = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

LABEL = {"random": "Random", "greedy": "Greedy", "lotus": "LOTUS", "tourrank": "TourRank",
         "entrred+probind": "EntrRed+ProbInd", "entrred+probdep": "EntrRed+ProbDep", "exhaustive": "Exhaustive"}
# grayscale: (color, marker, linestyle) per algorithm; bars also use HATCH
STYLE = {"random": ("0.75", "x", ":"), "greedy": ("0.55", "s", "--"), "lotus": ("0.35", "^", "-."),
         "tourrank": ("0.5", "v", (0, (3, 1, 1, 1))), "entrred+probind": ("0.2", "o", "-"),
         "entrred+probdep": ("black", "D", "-")}
HATCH = {"random": "", "greedy": "//", "lotus": "..", "tourrank": "xx", "entrred+probind": "\\\\", "entrred+probdep": ""}
BARCOL = {"lotus": "0.85", "greedy": "0.65", "entrred+probind": "0.4", "entrred+probdep": "black"}
ORDER = ["random", "lotus", "tourrank", "greedy", "entrred+probind", "entrred+probdep"]
DS = {"hotels": "Hotels", "movies": "Movies", "yelp": "Yelp"}
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "pdf.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "errorbar.capsize": 1.5})
COL = (3.45, 1.23)      # one-column figure (inches)
WIDE = (7.0, 1.65)      # two-column figure
def _save(fig, name):
    fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(os.path.join(FIGURES, name), bbox_inches="tight", pad_inches=0.02); plt.close(fig)



def load(exp):
    path = os.path.join(RESULTS, f"{exp}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "calls" in df and "F" in df:
        # The unary calls are issued by every algorithm to build the candidate
        # space; "calls" counts the calls issued after them.
        df["total_calls"] = df["calls"]
        df["calls"] = df["calls"] - df["x"] * N_POOL * df["F"].map(N_UNARY)
    return df


N_POOL = 200
N_UNARY = {"F1": 1, "F2": 1, "F3": 1, "F4": 1, "F5": 1, "F6": 1, "F1+": 2, "F1++": 2, "F5+": 2, "F5++": 2}


def agg(df, by):
    g = df.groupby(by + ["algo"])["calls"].agg(["mean", "std"]).reset_index()
    return g


def exhaustive_calls(n=200, x=3):
    return x * comb(n, 2)


K_MAX = 7


def fig1(df):
    df = df[df.k <= K_MAX]
    panels = [F for F in ["F1", "F3", "F5", "F6"] if (df.F == F).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=WIDE)
    for ax, (ds, sub) in zip(axes, [(df[df.F == F].dataset.iloc[0], df[df.F == F]) for F in panels]):
        g = agg(sub, ["k"])
        for algo in [a for a in ORDER if a not in ("lotus", "tourrank", "entrred+probind")]:
            r = g[g.algo == algo]
            c, m, ls = STYLE[algo]
            ax.plot(r.k, r["mean"], color=c, marker=m, ls=ls, label=LABEL[algo], ms=4)
        # TourRank, LOTUS and Exhaustive are off the chart: their counts are annotated
        lot = sub[sub.algo == "lotus"].calls.mean(); tr = sub[sub.algo == "tourrank"].calls.mean()
        ax.text(0.97, 0.04, f"TourRank: {tr:,.0f}\nLOTUS: {lot:,.0f}\nExhaustive: {exhaustive_calls():,}", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=6.5, color="0.25")
        ax.set_xlabel("k"); ax.set_xticks(sorted(sub.k.unique()))
        ax.set_title(f"{DS[ds]} ({sub.F.iloc[0] if len(sub) else ''})"); ax.grid(alpha=.3)
    axes[0].set_ylabel("#predicate calls")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=7)
    _save(fig, "fig1_calls_vs_k.pdf")


def fig3(df):
    fig, axes = plt.subplots(1, 2, figsize=COL)
    algos = ["lotus", "greedy", "entrred+probdep"]
    for ax, ds in zip(axes, ["hotels", "yelp"]):
        sub = df[df.dataset == ds]
        g = sub.groupby(["ell", "algo"])["calls"].agg(["mean", "std"]).reset_index()
        ells = sorted(sub.ell.unique()); w = 0.8 / len(algos)
        for i, algo in enumerate(algos):
            r = g[g.algo == algo].set_index("ell").reindex(ells)
            ax.bar(np.arange(len(ells)) + (i - 1) * w, r["mean"], w, color=BARCOL[algo],
                   hatch=HATCH[algo], edgecolor="black", linewidth=0.6, label=LABEL[algo])
        ax.set_xticks(range(len(ells))); ax.set_xticklabels([f"ℓ={e}" for e in ells])
        ax.set_title(DS[ds]); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("#predicate calls")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=6.5, handlelength=1.4, columnspacing=1.5)
    _save(fig, "fig3_predicates.pdf")


def fig4(df):
    """Predicate calls vs. the variability eps of the responses (x = 3)."""
    fig, axes = plt.subplots(1, 2, figsize=(COL[0], COL[1] + 0.3))
    look = {"greedy": ("0.6", "s", "--"), "entrred+probdep": ("black", "D", "-")}
    for ax, ds in zip(axes, ["hotels", "yelp"]):
        sub = df[(df.dataset == ds) & (df.x == 3)]
        for algo, (c, m, ls) in look.items():
            r = sub[sub.algo == algo].groupby("eps")["calls"].agg(["mean", "std"]).reset_index()
            ax.plot(range(len(r)), r["mean"], color=c, ls=ls, marker=m, ms=3.5, label=LABEL[algo])
        ax.set_xticks(range(len(r))); ax.set_xticklabels([f"{v:g}" for v in r.eps])
        ax.set_xlabel("response variability ε"); ax.set_title(DS[ds]); ax.grid(alpha=.3)
    axes[0].set_ylabel("#predicate calls")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.14), handlelength=2.2, columnspacing=2)
    _save(fig, "fig4_uncertainty.pdf")


def fig5(df):
    fig, axes = plt.subplots(1, 2, figsize=(COL[0], COL[1] + 0.3))
    look = {"entrred+probind": ("0.6", "s", "--", -0.004), "entrred+probdep": ("black", "D", "-", 0.004)}
    for ax, ds in zip(axes, ["hotels", "movies"]):
        sub = df[df.dataset == ds]
        for algo in ["entrred+probind", "entrred+probdep"]:
            r = sub[sub.algo == algo].groupby("theta")["calls"].agg(["mean", "std"]).reset_index()
            c, m, ls, dx = look[algo]
            ax.plot(range(len(r)), r["mean"], color=c, marker=m, ls=ls, label=LABEL[algo], ms=3.5)
        ax.set_xticks(range(len(r))); ax.set_xticklabels([f"{v:g}" for v in r.theta], fontsize=5.8)
        ax.set_xlabel("θ"); ax.set_title(DS[ds]); ax.grid(alpha=.3)
    axes[0].set_ylabel("#predicate calls")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=6.8, handlelength=2.2, columnspacing=2)
    _save(fig, "fig5_theta.pdf")


def fig_recall_theta(df):
    """Recall vs. theta (k = 7) of EntrRed+ProbInd and EntrRed+ProbDep."""
    fig, axes = plt.subplots(1, 2, figsize=(COL[0], COL[1] + 0.3))
    for ax, ds in zip(axes, ["hotels", "movies"]):
        sub = df[df.dataset == ds]
        for algo in ["entrred+probind", "entrred+probdep"]:
            r = sub[sub.algo == algo].groupby("theta")["recall"].mean().reset_index()
            c, m, ls = {"entrred+probind": ("0.6", "s", "--"), "entrred+probdep": ("black", "D", "-")}[algo]
            ax.plot(range(len(r)), 100 * r["recall"], color=c, marker=m, ls=ls, label=LABEL[algo], ms=3.5)
        ax.set_xticks(range(len(r))); ax.set_xticklabels([f"{v:g}" for v in r.theta], fontsize=5.8)
        ax.set_ylim(-5, 105); ax.set_yticks([0, 50, 100])
        ax.set_xlabel("θ"); ax.set_title(DS[ds]); ax.grid(alpha=.3)
    axes[0].set_ylabel("recall (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=6.8, handlelength=2.2, columnspacing=2)
    _save(fig, "fig7_recall_theta.pdf")


def fig6(df):
    g = df.groupby(["M", "component"])["time_s"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=COL)
    for comp, lab, c, m in [("bounds", "bound update", "0.6", "s"), ("probind", "ProbInd", "0.3", "o"),
                            ("probdep", "ProbDep", "black", "D")]:
        r = g[g.component == comp]; axes[0].plot(r.M, r.time_s, color=c, marker=m, label=lab, ms=3.5)
    for comp, lab, c, m, ls in [("entrred", "EntrRed (+ProbInd)", "black", "o", "-"), ("greedy", "Greedy", "0.45", "^", "--")]:
        r = g[g.component == comp]; axes[1].plot(r.M, r.time_s, color=c, marker=m, ls=ls, label=lab, ms=3.5)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("#candidate sets M"); ax.grid(alpha=.3, which="both")
    axes[0].set_ylabel("time (s)")
    h0, l0 = axes[0].get_legend_handles_labels(); h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=5, frameon=False, fontsize=6.5, handlelength=1.3, handletextpad=0.4, columnspacing=1.0)
    _save(fig, "fig6_scalability.pdf")


def table_x8(df):
    base = df[df.tau.isna()].groupby("dataset")["calls"].mean()
    print("\n% X8: candidate-set management (recall %, calls saved %)")
    for ds in ["hotels", "movies", "yelp"]:
        sub = df[df.dataset == ds]
        cells = [f"{100 * sub[sub.tau.isna()].recall.mean():.0f}\\%"]
        for tau in [0.96, 0.92, 0.88, 0.82]:
            r = sub[sub.tau == tau]
            cells.append(f"{100 * r.recall.mean():.0f}\\% ({100 * (1 - r.calls.mean() / base[ds]):.0f}\\%)")
        print(f"{DS[ds]} & " + " & ".join(cells) + " \\\\")


LLM_PRICE = {"GPT-6 Luna": (0.10, 0.50), "Llama-4-Scout": (0.11, 0.34)}        # $ per 1M tokens (in, out)
VLM_PRICE = {"Qwen3-VL-235B": (0.20, 0.88), "Mistral Large": (0.50, 1.50)}
LLM_NAME = {"gpt-6-luna": "GPT-6 Luna", "llama-4-scout": "Llama-4-Scout"}
VLM_NAME = {"qwen3-vl": "Qwen3-VL-235B", "mistral-large": "Mistral Large"}


def table_cost(df1, text_tokens=900, image_tokens=2000, out_tokens=8):
    """Cost per query from the X1 calls and the per-token prices: F1 rows with the LLMs, F5 rows with the VLMs."""
    print("\n% cost per query ($); rows F, k; per model: Greedy, LOTUS, EntrRed+ProbDep")
    for F, toks, price in [("F1", text_tokens, LLM_PRICE), ("F5", image_tokens, VLM_PRICE)]:
        for k in [3, 5, 7]:
            cells = []
            for model, (pin, pout) in price.items():
                for algo in ["greedy", "lotus", "entrred+probdep"]:
                    calls = df1[(df1.F == F) & (df1.k == k) & (df1.algo == algo)].calls.mean()
                    cells.append(f"\\${calls * (toks * pin + out_tokens * pout) / 1e6:.3f}")
            print(f"{F} & {k} & " + " & ".join(cells) + " \\\\")


def table_models(dm, text_tokens=900, image_tokens=2000, out_tokens=8):
    """Per query, each model evaluating the calls (XM): calls, cost and recall of EntrRed+ProbDep."""
    print("\n% models: calls, cost per query ($), recall")
    for (F, llm, vlm), g in dm.groupby(["F", "llm", "vlm"]):
        model, toks, (pin, pout) = ((llm, text_tokens, LLM_PRICE[LLM_NAME[llm]]) if F == "F1"
                                    else (vlm, image_tokens, VLM_PRICE[VLM_NAME[vlm]]))
        calls = g.calls.mean()
        print(f"{model} & {calls:.0f} & \\${calls * (toks * pin + out_tokens * pout) / 1e6:.3f} & {100 * g.recall.mean():.0f}\\% \\\\")


def recall_baselines(d):
    print("\n% recall per algorithm and k")
    print((100 * d.groupby(["algo", "k"]).recall.mean().unstack()).round(0).to_string())


def summary(df1):
    print("\n% X1: mean calls and recall per algorithm")
    df1 = df1[df1.k <= K_MAX]
    g = df1.groupby("algo").agg(calls=("calls", "mean"), recall=("recall", "mean"))
    print(g.round(2).to_string())
    print(df1[df1.algo.str.startswith("entrred")].groupby(["algo", "k"])["recall"].mean().unstack().round(2).to_string())
    for ds in ["hotels", "movies", "yelp"]:
        s = df1[df1.dataset == ds].groupby(["k", "algo"]).calls.mean().unstack()
        print(f"\n{ds}\n", s.round(0).to_string())


def main():
    os.makedirs(FIGURES, exist_ok=True)
    d = {e: load(e) for e in ["X1", "XF", "X3", "X4", "X5", "X6", "XM", "X8"]}
    if d["X1"] is not None: fig1(d["X1"]); summary(d["X1"]); table_cost(d["X1"])
    rec = [d[e] for e in ["X1", "XF", "X3"] if d[e] is not None]
    if rec: recall_baselines(pd.concat(rec))
    if d["X3"] is not None: fig3(d["X3"])
    if d["X4"] is not None: fig4(d["X4"])
    if d["X5"] is not None: fig5(d["X5"]); fig_recall_theta(d["X5"])
    if d["XM"] is not None: table_models(d["XM"])
    if d["X6"] is not None: fig6(d["X6"])
    if d["X8"] is not None: table_x8(d["X8"])


if __name__ == "__main__":
    main()
