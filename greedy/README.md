# Greedy Baseline — Experiments & Plots

## Algorithm

**Greedy** works exactly like **Naive (Random)** except for how the next diversity
pair is chosen:

> At each iteration, find the candidate with the **highest current upper bound**,
> then pick a **random unasked pair** from that candidate's elements.

This targets diversity queries at the most promising candidate, without the full
probabilistic model required by EntrRed (Max_Prob).

Expected cost ordering: **EntrRed < Greedy < Random**

---

## Folder structure

```
greedy/
├── exp_fig2_hotels.py        ← Fig 2: Hotels F1+F2, k varies, Discrete+Range
├── exp_fig2_businesses.py    ← Fig 2: Businesses F3+F5, k varies, Discrete+Range
├── exp_fig6_movies.py        ← Fig 6: Movies F3+F1, k varies, Discrete+Range
├── exp_fig4_scalability.py   ← Fig 4: time(next question) vs #candidates
├── plot_all_figures.py       ← Regenerate all figures with Greedy added
│
├── fig2_hotels/
│   ├── Discrete/   ← Results_Hotels_REL_*_DIV_*.csv
│   └── Range/
├── fig2_businesses/
│   ├── Discrete/
│   └── Range/
├── fig6_movies/
│   ├── Discrete/
│   └── Range/
├── fig4_scalability/         ← time results for Hotels F1 + Movies F3
├── cost_tables/              ← estimated $ cost CSVs
└── plots/                    ← generated PDF + PNG figures
```

---

## How to run

Run all experiments from the **project root directory** (not from inside `greedy/`):

```bash
# Step 1 – Figure 2 data (Hotels)
python greedy/exp_fig2_hotels.py

# Step 2 – Figure 2 data (Businesses)
python greedy/exp_fig2_businesses.py

# Step 3 – Figure 6 data (Movies)
python greedy/exp_fig6_movies.py

# Step 4 – Figure 4 data (scalability / time per question)
python greedy/exp_fig4_scalability.py

# Step 5 – Generate all plots and cost tables
python greedy/plot_all_figures.py
```

Output figures are saved to `greedy/plots/` as both PDF and PNG.
Cost tables are saved to `greedy/cost_tables/`.

---

## What each experiment produces

| Script | Output subfolder | Figures reproduced |
|--------|------------------|--------------------|
| `exp_fig2_hotels.py` | `fig2_hotels/{Discrete,Range}/` | Fig 2 (a,c) |
| `exp_fig2_businesses.py` | `fig2_businesses/{Discrete,Range}/` | Fig 2 (b,d) |
| `exp_fig6_movies.py` | `fig6_movies/{Discrete,Range}/` | Fig 6 |
| `exp_fig4_scalability.py` | `fig4_scalability/` | Fig 4 |
| `plot_all_figures.py` | `plots/`, `cost_tables/` | Fig 2,4,6,8 + Tables 6,7 |

---

## Notes

- All experiments use pre-computed MGT (Mock Ground Truth) data — **no real LLM calls needed**.
- `use_filtered_init_candidates=True` for Fig 2/6 (uses existing FIC files with 100 candidates).
- `use_filtered_init_candidates=False` for Fig 4 (candidate set grows with n to measure scaling).
- For Fig 8 (PDF computation time): Greedy has `total_time_compute_pdf = 0` by design — it never computes a probabilistic model.
