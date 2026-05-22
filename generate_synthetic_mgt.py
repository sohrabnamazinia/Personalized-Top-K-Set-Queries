"""
Generates synthetic MGT (Mock Ground Truth) CSV files for all dataset/config combos.
Uses reproducible random scores — no LLM calls needed.
Covers both MGT_Results/ (Discrete) and MGT_Range_Results/ (Range).

FIC candidates only reference document indices 0-29 (n=30 generation),
so we need at most C(30,2)=435 diversity rows for FIC experiments.
For scalability (no FIC, n up to 201) we need C(201,2)=20100 diversity rows.
We generate n=1000 files with 20100 diversity rows each — the utility auto-subsets.
"""
import os, numpy as np, csv

np.random.seed(42)

CONFIGS = [
    ("hotels",     "Rating_of_the_hotel",       "Physical_distance_of_the_hotels"),
    ("hotels",     "Distance_from_city_center",  "Star_rating"),
    ("businesses", "Location_Around_New_York",   "Cost"),
    ("businesses", "Type_of_food",               "Open_hours"),
    ("movies",     "Brief_plot",                 "Different_years"),
    ("movies",     "Popularity",                 "Genre_and_movie_periods"),
]

N_REL  = 1000     # relevance rows (covers all experiments up to n=1000)
N_DIV  = 20100    # diversity rows — covers C(201,2)=20100 (max n in scalability)

def round1(x):
    return round(float(x), 1)

def gen_range_relevance(out_path, n):
    """value_lower, value_upper in [0,1] with small gap."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = np.clip(np.random.normal(0.6, 0.15, n), 0.0, 0.9)
    gap  = np.clip(np.random.uniform(0.0, 0.2, n), 0.0, 1.0 - base)
    lower = np.round(base, 1)
    upper = np.round(np.minimum(lower + gap, 1.0), 1)
    times = np.clip(np.random.normal(0.57, 0.1, n), 0.1, 1.1)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d", "value_lower", "value_upper", "time_rel"])
        for d in range(n):
            w.writerow([d, round1(lower[d]), round1(upper[d]), round(times[d], 4)])

def gen_range_diversity(out_path, num_rows):
    """Pairs (d1 < d2). Rows ordered by: d2=0,1,2,... d1<d2."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = np.clip(np.random.normal(0.5, 0.2, num_rows), 0.0, 0.9)
    gap  = np.clip(np.random.uniform(0.0, 0.2, num_rows), 0.0, 1.0 - base)
    lower = np.round(base, 1)
    upper = np.round(np.minimum(lower + gap, 1.0), 1)
    times = np.clip(np.random.normal(0.57, 0.1, num_rows), 0.1, 1.1)
    # generate pairs in the order the code expects: for d2 in range(n), for d1 in range(d2)
    pairs = [(d1, d2) for d2 in range(1, 10000) for d1 in range(d2) if ((d2*(d2-1))//2 + d1) < num_rows]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d2", "d1", "value_lower", "value_upper", "time_div"])
        for idx, (d1, d2) in enumerate(pairs):
            w.writerow([d2, d1, round1(lower[idx]), round1(upper[idx]), round(times[idx], 4)])

def gen_discrete_relevance(out_path, n):
    """Single value per document."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vals  = np.round(np.clip(np.random.normal(0.6, 0.15, n), 0.0, 1.0), 1)
    times = np.clip(np.random.normal(0.57, 0.1, n), 0.1, 1.1)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d", "value", "time_rel"])
        for d in range(n):
            w.writerow([d, round1(vals[d]), round(times[d], 4)])

def gen_discrete_diversity(out_path, num_rows):
    """Single value per pair."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vals  = np.round(np.clip(np.random.normal(0.5, 0.2, num_rows), 0.0, 1.0), 1)
    times = np.clip(np.random.normal(0.57, 0.1, num_rows), 0.1, 1.1)
    pairs = [(d1, d2) for d2 in range(1, 10000) for d1 in range(d2) if ((d2*(d2-1))//2 + d1) < num_rows]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d2", "d1", "value", "time_div"])
        for idx, (d1, d2) in enumerate(pairs):
            w.writerow([d2, d1, round1(vals[idx]), round(times[idx], 4)])

# Extra n values needed by scalability experiment (exp_fig4_scalability.py)
SCALABILITY_N = [45, 101, 142, 173, 201]

if __name__ == "__main__":
    # Per-n files for scalability experiments (Discrete mode, varies n)
    for dataset, rel_def, div_def in [
        ("hotels", "Rating_of_the_hotel", "Physical_distance_of_the_hotels"),
        ("movies",  "Brief_plot",          "Different_years"),
    ]:
        for n in SCALABILITY_N:
            n_div_rows = n * (n - 1) // 2
            print(f"  Scalability MGT n={n}: {dataset}")
            gen_discrete_relevance(f"MGT_Results/MGT_{dataset}_{n}_Rel_{rel_def}.csv", n)
            gen_discrete_diversity(f"MGT_Results/MGT_{dataset}_{n}_Div_{div_def}.csv", n_div_rows)

    for dataset, rel_def, div_def in CONFIGS:
        print(f"Generating MGT for {dataset} | {rel_def} / {div_def} ...")

        # ── Range (MGT_Range_Results) ──────────────────────────────────
        rr_dir = "MGT_Range_Results"
        gen_range_relevance(f"{rr_dir}/MGT_{dataset}_{N_REL}_Rel_{rel_def}.csv", N_REL)
        gen_range_diversity(f"{rr_dir}/MGT_{dataset}_{N_REL}_Div_{div_def}.csv", N_DIV)

        # ── Discrete (MGT_Results) ─────────────────────────────────────
        dr_dir = "MGT_Results"
        gen_discrete_relevance(f"{dr_dir}/MGT_{dataset}_{N_REL}_Rel_{rel_def}.csv", N_REL)
        gen_discrete_diversity(f"{dr_dir}/MGT_{dataset}_{N_REL}_Div_{div_def}.csv", N_DIV)

    print("Done. All synthetic MGT files written.")
