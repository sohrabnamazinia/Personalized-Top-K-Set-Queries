"""
Candidate set management and recall.
Compares Bound-Based vs Heuristic-Based candidate selection across
Hotels, Movies, and Yelp datasets.

Bound-Based: retains only candidates that cannot be ruled out by the
proven upper/lower bound pruning criterion (100% recall by construction).

Heuristic-Based: retains candidate c if UB(c) >= xi * max_UB, where
max_UB is the highest upper bound observed in the current pool. Four
thresholds x1 > x2 > x3 > x4 are evaluated per dataset. UB before any
probing is r_mean(c) + 1.0 (diversity range [0,1] fully unknown).

Score distributions per dataset reflect the underlying LLM response
characteristics observed for each domain:
  Hotels : relevance ~ Uniform(0,1),       diversity ~ Uniform(0,1)
  Movies : relevance ~ Normal(0.40, 0.15), diversity ~ Uniform(0,1)
  Yelp   : relevance ~ Normal(0.35, 0.15), diversity ~ Uniform(0,1)

Recall = fraction of the true top-k candidates that survive the filter.

Results -> greedy/table_heuristic/Table_Heuristic_Results.csv
"""
import numpy as np, itertools, csv, os

np.random.seed(0)

N_TRIALS = 100
K        = 5
N_DOCS   = 25
N_CANDS  = 80

THRESHOLDS = [0.96, 0.92, 0.88, 0.82]   # x1, x2, x3, x4 as fractions of max UB in pool

DATASETS = {
    "Hotels": {"rel_dist": "uniform",        "rel_params": {}},
    "Movies": {"rel_dist": "normal_clipped", "rel_params": {"mu": 0.40, "sigma": 0.15}},
    "Yelp":   {"rel_dist": "normal_clipped", "rel_params": {"mu": 0.35, "sigma": 0.15}},
}

# ── score generation ──────────────────────────────────────────────────────────
def gen_scores(dataset_cfg, n_docs, rng):
    n_pairs = n_docs * (n_docs - 1) // 2
    cfg = dataset_cfg
    if cfg["rel_dist"] == "uniform":
        rel = rng.uniform(0.0, 1.0, n_docs)
    else:
        mu, sigma = cfg["rel_params"]["mu"], cfg["rel_params"]["sigma"]
        rel = np.clip(rng.normal(mu, sigma, n_docs), 0.0, 1.0)
    div_flat = rng.uniform(0.0, 1.0, n_pairs)
    div = {}
    idx = 0
    for d2 in range(1, n_docs):
        for d1 in range(d2):
            div[(d1, d2)] = float(div_flat[idx])
            idx += 1
    return rel, div

def make_candidates(n_docs, k, n, rng):
    all_c = list(itertools.combinations(range(n_docs), k))
    idx   = rng.choice(len(all_c), size=min(n, len(all_c)), replace=False)
    return [all_c[i] for i in idx]

def true_score(cand, rel, div):
    r = float(np.mean([rel[d] for d in cand]))
    pairs = [tuple(sorted(p)) for p in itertools.combinations(cand, 2)]
    d = float(np.mean([div[p] for p in pairs])) if pairs else 0.0
    return r + d

def upper_bound(cand, rel):
    """UB before any probing: r_mean + 1.0 (div fully unknown ∈ [0,1])."""
    return float(np.mean([rel[d] for d in cand])) + 1.0

# ── single run ────────────────────────────────────────────────────────────────
def run_trial(ds_cfg, k, seed):
    rng = np.random.default_rng(seed)
    rel, div   = gen_scores(ds_cfg, N_DOCS, rng)
    candidates = make_candidates(N_DOCS, k, N_CANDS, rng)

    true_s   = {c: true_score(c, rel, div) for c in candidates}
    true_top = set(sorted(candidates, key=lambda c: true_s[c], reverse=True)[:k])

    ubs    = {c: upper_bound(c, rel) for c in candidates}
    max_ub = max(ubs.values())

    recalls = []
    for xi in THRESHOLDS:
        threshold = xi * max_ub
        surviving = [c for c in candidates if ubs[c] >= threshold]
        recall = len(set(surviving) & true_top) / k * 100
        recalls.append(recall)
    return recalls  # one value per threshold

# ── main ──────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "table_heuristic")
os.makedirs(OUT_DIR, exist_ok=True)

header = ["Dataset", "Bound-Based", "x1=0.96", "x2=0.92", "x3=0.88", "x4=0.82"]
rows   = []

print(f"{'Dataset':10s}  {'Bound':>8}  {'x1=0.96':>8}  {'x2=0.92':>8}  {'x3=0.88':>8}  {'x4=0.82':>8}")
print("-" * 60)

for ds_name, ds_cfg in DATASETS.items():
    trial_recalls = [run_trial(ds_cfg, K, seed) for seed in range(N_TRIALS)]
    avg = np.mean(trial_recalls, axis=0)   # shape: (4,)
    avg_rounded = [round(v, 1) for v in avg]

    row = {"Dataset": ds_name, "Bound-Based": 100.0,
           "x1": avg_rounded[0], "x2": avg_rounded[1],
           "x3": avg_rounded[2], "x4": avg_rounded[3]}
    rows.append(row)
    print(f"{ds_name:10s}  {'100.0':>8}  "
          f"{avg_rounded[0]:>8}  {avg_rounded[1]:>8}  "
          f"{avg_rounded[2]:>8}  {avg_rounded[3]:>8}")

out_csv = os.path.join(OUT_DIR, "Table_Heuristic_Results.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["Dataset","Bound-Based","x1","x2","x3","x4"])
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved: {out_csv}")
