"""
Table 8: Effect of LLM score distribution on probing calls and recall.
Compares Uniform(0,1) vs Normal(mu=0.5, sigma=0.10) for k = 3, 5, 7.

Self-contained simulation. Relevance scores are drawn once per trial and
treated as known; diversity scores are probed pair-by-pair. The algorithm
maintains lower/upper bounds on each candidate's score and prunes candidates
that cannot be in the top-k. For Normal scores the bounds for unprobed pairs
are initialised to the 3-sigma interval [0.2, 0.8]; for Uniform the full
range [0.0, 1.0] is used.

Pruning criterion: a candidate is removed only when at least k other active
candidates have a lower bound exceeding its upper bound (safe top-k pruning).
The simulation stops once WIN_THRESHOLD of the initial pool has been pruned.

Recall is the fraction of the true top-k that remains in the active set
when the algorithm terminates.

Results -> greedy/table8_distributions/Table8_Results.csv
"""
import numpy as np, random, itertools, csv, os

np.random.seed(0)
random.seed(0)

N_DOCS        = 20      # |D| per run
N_CANDIDATES  = 50      # initial pool
WIN_THRESHOLD = 0.78    # stop when 78 % eliminated  (22 % ≈ 11 remain)
N_TRIALS      = 30
K_VALUES      = [3, 5, 7]
DISTRIBUTIONS = ["Uniform", "Normal"]

MU, SIGMA = 0.50, 0.10
N_UNK_LB = max(0.0, MU - 3 * SIGMA)
N_UNK_UB = min(1.0, MU + 3 * SIGMA)

# ── helpers ───────────────────────────────────────────────────────────────────
def gen_scores(dist, n_docs, rng):
    n_pairs = n_docs * (n_docs - 1) // 2
    rel = rng.uniform(0.0, 1.0, n_docs)
    div_flat = (rng.uniform(0.0, 1.0, n_pairs) if dist == "Uniform"
                else np.clip(rng.normal(MU, SIGMA, n_pairs), 0.0, 1.0))
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

def get_bounds(cand, rel, probed_div, unk_lb, unk_ub):
    pairs  = [tuple(sorted(p)) for p in itertools.combinations(cand, 2)]
    n_p    = len(pairs)
    r_mean = float(np.mean([rel[d] for d in cand]))
    if n_p == 0:
        return r_mean, r_mean
    known_sum, n_known = 0.0, 0
    for p in pairs:
        if p in probed_div:
            known_sum += probed_div[p]
            n_known   += 1
    n_unk = n_p - n_known
    return (r_mean + (known_sum + n_unk * unk_lb) / n_p,
            r_mean + (known_sum + n_unk * unk_ub) / n_p)

# ── simulation ────────────────────────────────────────────────────────────────
def run_sim(dist, k, seed):
    rng = np.random.default_rng(seed)
    random.seed(int(seed))

    unk_lb = 0.0 if dist == "Uniform" else N_UNK_LB
    unk_ub = 1.0 if dist == "Uniform" else N_UNK_UB

    rel, div   = gen_scores(dist, N_DOCS, rng)
    candidates = make_candidates(N_DOCS, k, N_CANDIDATES, rng)
    if not candidates:
        return 0, 100.0

    true_s   = {c: true_score(c, rel, div) for c in candidates}
    true_top = set(sorted(candidates, key=lambda c: true_s[c], reverse=True)[:k])

    probed_div = {}
    asked      = set()
    active     = list(candidates)
    probes     = 0
    # stop when WIN_THRESHOLD fraction pruned; always keep at least k active
    target     = max(k, int(len(active) * (1 - WIN_THRESHOLD)))

    while len(active) > target:
        bds = {c: get_bounds(c, rel, probed_div, unk_lb, unk_ub) for c in active}

        # Choose pair from highest-UB candidate
        chosen_pair = None
        for cand in sorted(active, key=lambda c: bds[c][1], reverse=True):
            pairs = [tuple(sorted(p)) for p in itertools.combinations(cand, 2)]
            avail = [p for p in pairs if p not in asked]
            if avail:
                chosen_pair = avail[0]
                break
        if chosen_pair is None:
            break  # all pairs exhausted

        asked.add(chosen_pair)
        probed_div[chosen_pair] = div[chosen_pair]
        probes += 1

        bds    = {c: get_bounds(c, rel, probed_div, unk_lb, unk_ub) for c in active}
        sorted_lbs = sorted([b[0] for b in bds.values()], reverse=True)
        kth_lb     = sorted_lbs[k - 1] if len(sorted_lbs) >= k else 0.0
        active     = [c for c in active if bds[c][1] >= kth_lb]

    recall = len(set(active) & true_top) / k * 100
    return probes, recall

# ── main ──────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "table8_distributions")
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
print(f"{'Dist':10s}  {'k':>3}  {'Probes':>8}  {'Recall':>8}")
print("-" * 40)
for dist in DISTRIBUTIONS:
    for k in K_VALUES:
        probes_list, recall_list = [], []
        for seed in range(N_TRIALS):
            p, r = run_sim(dist, k, seed * 11 + 3)
            probes_list.append(p)
            recall_list.append(r)
        avg_probes = round(np.mean(probes_list))
        avg_recall = round(np.mean(recall_list), 1)
        rows.append({"distribution": dist, "k": k,
                     "avg_probes": avg_probes, "recall_pct": avg_recall})
        print(f"{dist:10s}  {k:>3}  {avg_probes:>8}  {avg_recall:>7.1f}%")

out_csv = os.path.join(OUT_DIR, "Table8_Results.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["distribution","k","avg_probes","recall_pct"])
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved: {out_csv}")
