"""
Regenerates all paper figures with the Greedy baseline added.
Run this AFTER all exp_* scripts have completed.

Output figures saved to greedy/plots/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── colour / hatch conventions ──────────────────────────────────────────────
STYLE = {
    "Max_Prob":  {"color": "black",   "hatch": "",     "label": "EntrRed using ProbInd"},
    "Naive":     {"color": "#aaaaaa", "hatch": "....", "label": "Random"},
    "Greedy":    {"color": "#555555", "hatch": "xxxx", "label": "Greedy"},
}
K_VALUES = [2, 4, 6, 8, 10]

# ── helper ───────────────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def bar_chart(ax, df, title, methods_order=("Max_Prob", "Naive", "Greedy")):
    """Draw grouped bar chart of api_calls per k for each method."""
    k_vals = sorted(df["k"].unique())
    n_methods = len(methods_order)
    x = np.arange(len(k_vals))
    width = 0.25

    for idx, method in enumerate(methods_order):
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        calls = [sub[sub["k"] == k]["api_calls"].values[0] if not sub[sub["k"] == k].empty else 0
                 for k in k_vals]
        offset = (idx - (n_methods - 1) / 2) * width
        s = STYLE[method]
        ax.bar(x + offset, calls, width, label=s["label"],
               color=s["color"], hatch=s["hatch"], edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(k_vals)
    ax.set_xlabel("k")
    ax.set_ylabel("#LLM Calls")
    ax.set_title(title, fontsize=9)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.legend(title="Algorithm", title_fontsize=8, fontsize=7, frameon=True)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2  — Hotels F1 + Businesses F5, Discrete + Range
# ═══════════════════════════════════════════════════════════════════════════
def plot_figure2():
    subplots = [
        {
            "csv":  os.path.join(SCRIPT_DIR, "fig2_hotels", "Discrete",
                                 "Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv"),
            "title":  "(a) Hotels - F1 - Discrete",
            "fname":  "fig2a_Hotels_F1_Discrete.png",
        },
        {
            "csv":  os.path.join(SCRIPT_DIR, "fig2_businesses", "Discrete",
                                 "Results_Businesses_REL_Location_Around_New_York_DIV_Cost.csv"),
            "title":  "(b) Businesses - F5 - Discrete",
            "fname":  "fig2b_Businesses_F5_Discrete.png",
        },
        {
            "csv":  os.path.join(SCRIPT_DIR, "fig2_hotels", "Range",
                                 "Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv"),
            "title":  "(c) Hotels - F1 - Range",
            "fname":  "fig2c_Hotels_F1_Range.png",
        },
        {
            "csv":  os.path.join(SCRIPT_DIR, "fig2_businesses", "Range",
                                 "Results_Businesses_REL_Location_Around_New_York_DIV_Cost.csv"),
            "title":  "(d) Businesses - F5 - Range",
            "fname":  "fig2d_Businesses_F5_Range.png",
        },
    ]

    for sp in subplots:
        df = load(sp["csv"])
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        if df is None:
            ax.set_title(f"{sp['title']}\n(data missing)", fontsize=8)
        else:
            bar_chart(ax, df, sp["title"])
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, sp["fname"])
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6  — Movies F3, Discrete + Range
# ═══════════════════════════════════════════════════════════════════════════
def plot_figure6():
    subplots = [
        {
            "csv":   os.path.join(SCRIPT_DIR, "fig6_movies", "Discrete",
                                  "Results_Movies_REL_Brief_plot_DIV_Different_years.csv"),
            "fname": "fig6a_Movies_F3_Discrete.png",
        },
        {
            "csv":   os.path.join(SCRIPT_DIR, "fig6_movies", "Range",
                                  "Results_Movies_REL_Brief_plot_DIV_Different_years.csv"),
            "fname": "fig6b_Movies_F3_Range.png",
        },
    ]

    for sp in subplots:
        df = load(sp["csv"])
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        if df is None:
            ax.set_title("(data missing)", fontsize=8)
        else:
            bar_chart(ax, df, "")   # no title
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, sp["fname"])
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4  — Time to determine next question vs #candidates
# ═══════════════════════════════════════════════════════════════════════════
# n_actual -> #candidates (for k=2: C(n,2))
import math
N_MAPPING = {n: math.comb(n, 2) for n in [45, 101, 142, 173, 201]}

def plot_figure4():
    subplots = [
        {
            "csv":   os.path.join(SCRIPT_DIR, "fig4_scalability",
                                  "Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv"),
            "title": "(a) Hotels - Scoring function $\mathcal{F}_1$",
            "fname": "fig4a_Hotels_F1_scalability.png",
        },
        {
            "csv":   os.path.join(SCRIPT_DIR, "fig4_scalability",
                                  "Results_Movies_REL_Brief_plot_DIV_Different_years.csv"),
            "title": "(b) Movies - Scoring function $\mathcal{F}_3$",
            "fname": "fig4b_Movies_F3_scalability.png",
        },
    ]

    line_styles = {
        "Max_Prob": {"color": "black",   "ls": "-",  "marker": "o", "label": "EntrRed using ProbInd"},
        "Greedy":   {"color": "#444444", "ls": "--", "marker": "s", "label": "Greedy"},
    }

    for sp in subplots:
        df = load(sp["csv"])
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        if df is None:
            ax.set_title(sp["title"] + "\n(data missing)", fontsize=8)
        else:
            df["n_candidates"] = df["n"].map(N_MAPPING)
            for method, sty in line_styles.items():
                sub = df[df["method"] == method].sort_values("n_candidates")
                if sub.empty:
                    continue
                ax.plot(sub["n_candidates"], sub["total_time_determine_next_question"],
                        marker=sty["marker"], color=sty["color"],
                        linestyle=sty["ls"], label=sty["label"])
            ax.set_xlabel("#Candidates")
            ax.set_ylabel("Time - Determine next question (Seconds)")
            ax.legend(title="Algorithm", title_fontsize=8, fontsize=7, frameon=True)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, sp["fname"])
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8  — PDF computation time vs #candidates (Greedy = 0)
# ═══════════════════════════════════════════════════════════════════════════
def plot_figure8():
    """
    Figure 8 compares ProbInd vs ProbDep PDF computation time.
    For Greedy, total_time_compute_pdf = 0 (no probabilistic model).
    We overlay a Greedy flat-zero line on top of the existing Fig8 data.
    """
    scalability_dir = os.path.join(SCRIPT_DIR, "..", "Results_Scalibility")
    greedy_fig4_dir = os.path.join(SCRIPT_DIR, "fig4_scalability")

    datasets_conf = [
        ("hotels", "Rating_of_the_hotel", "Physical_distance_of_the_hotels", "Hotels - F1"),
        ("businesses", "Location_Around_New_York", "Cost", "Businesses - F3"),
        ("businesses", "Location_Around_New_York", "Cost", "Businesses - F5"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (ds, rel, div, title) in zip(axes, datasets_conf):
        fname = f"Results_{ds.capitalize()}_REL_{rel}_DIV_{div}.csv"

        # EntrRed (ProbInd) from existing scalability results
        existing_path = os.path.join(scalability_dir, fname)
        df_existing = load(existing_path)
        if df_existing is not None:
            df_mp = df_existing[df_existing["method"] == "Max_Prob"].copy()
            df_mp["n_candidates"] = df_mp["n"].map(N_MAPPING)
            df_mp = df_mp.sort_values("n_candidates")
            ax.plot(df_mp["n_candidates"], df_mp["total_time_compute_pdf"],
                    marker="o", color="black", linestyle="-", label="ProbInd (EntrRed)")

        # Greedy — pdf time is 0
        greedy_path = os.path.join(greedy_fig4_dir, fname)
        df_greedy = load(greedy_path)
        if df_greedy is not None:
            df_g = df_greedy[df_greedy["method"] == "Greedy"].copy()
            df_g["n_candidates"] = df_g["n"].map(N_MAPPING)
            df_g = df_g.sort_values("n_candidates")
            ax.plot(df_g["n_candidates"], df_g["total_time_compute_pdf"],
                    marker="s", color="#444444", linestyle="--", label="Greedy (no PDF)")
        else:
            # Draw flat zero line over candidate range
            x_range = sorted(N_MAPPING.values())
            ax.plot(x_range, [0] * len(x_range),
                    marker="s", color="#444444", linestyle="--", label="Greedy (no PDF)")

        ax.set_xlabel("#Candidates")
        ax.set_ylabel("Time - Compute PDF (s)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Figure 8: PDF computation time — ProbInd vs Greedy (no PDF)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "figure8_with_greedy.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# COST TABLES  (Tables 6 & 7)
# ═══════════════════════════════════════════════════════════════════════════
# Prices per LLM call (approximate, based on existing table values)
GPT_COST_PER_CALL  = 0.00032   # $ per diversity/relevance call for GPT-4-turbo
LLAMA_COST_PER_CALL = 0.000060  # $ per call for Llama-3

def compute_cost_tables():
    """
    Compute estimated monetary cost tables for Greedy vs existing methods.
    Reads api_calls from greedy results and applies cost-per-call rate.
    """
    out_dir = os.path.join(SCRIPT_DIR, "cost_tables")
    os.makedirs(out_dir, exist_ok=True)

    # Hotels F1 (used for Table 6 F1 column) + Businesses F5 (F5 column)
    sources = {
        "Hotels_F1": {
            "disc": os.path.join(SCRIPT_DIR, "fig2_hotels", "Discrete",
                                 "Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv"),
            "range": os.path.join(SCRIPT_DIR, "fig2_hotels", "Range",
                                  "Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv"),
        },
        "Businesses_F5": {
            "disc": os.path.join(SCRIPT_DIR, "fig2_businesses", "Discrete",
                                 "Results_Businesses_REL_Location_Around_New_York_DIV_Cost.csv"),
            "range": os.path.join(SCRIPT_DIR, "fig2_businesses", "Range",
                                  "Results_Businesses_REL_Location_Around_New_York_DIV_Cost.csv"),
        },
    }

    for label, paths in sources.items():
        rows = []
        for setting, path in paths.items():
            df = load(path)
            if df is None:
                continue
            for _, row in df.iterrows():
                rows.append({
                    "k": int(row["k"]),
                    "setting": setting,
                    "method": row["method"],
                    "api_calls": int(row["api_calls"]),
                    "cost_GPT_$":   round(row["api_calls"] * GPT_COST_PER_CALL, 2),
                    "cost_Llama_$": round(row["api_calls"] * LLAMA_COST_PER_CALL, 2),
                })

        if rows:
            df_out = pd.DataFrame(rows).sort_values(["setting", "k", "method"])
            out_path = os.path.join(out_dir, f"Cost_{label}.csv")
            df_out.to_csv(out_path, index=False)
            print(f"Saved cost table: {out_path}")

            # Print Table-6 style summary for GPT
            print(f"\n  {label} — GPT cost ($)")
            pivot = df_out[df_out["setting"] == "disc"].pivot(
                index="k", columns="method", values="cost_GPT_$")
            print(pivot.to_string())


if __name__ == "__main__":
    print("=== Generating Figure 2 ===")
    plot_figure2()
    print("\n=== Generating Figure 6 ===")
    plot_figure6()
    print("\n=== Generating Figure 4 ===")
    plot_figure4()
    print("\n=== Generating Figure 8 ===")
    plot_figure8()
    print("\n=== Generating Cost Tables ===")
    compute_cost_tables()
    print("\nAll done. Check greedy/plots/ and greedy/cost_tables/")
