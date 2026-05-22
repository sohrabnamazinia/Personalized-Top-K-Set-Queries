"""
Experiment: Fig6 equivalent for Movies (F3 and F1).
Runs the Greedy baseline; EntrRed and Random results are loaded from prior experiments.
Results written to greedy/fig6_movies/Discrete/ and greedy/fig6_movies/Range/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv
from read_data_movies import read_data, merge_plots
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, GREEDY

experiments = [(1000, 2), (1000, 4), (1000, 6), (1000, 8), (1000, 10)]
win_threshold = 0.90
dataset_name = "movies"
input_query = "A popular movie"
metrics = [RELEVANCE, DIVERSITY]
use_MGTs = True
use_filtered_init_candidates = True
methods = [GREEDY]

ORIG_DISCRETE = os.path.join(os.path.dirname(__file__), '..', 'Results_Cost')
ORIG_RANGE    = os.path.join(os.path.dirname(__file__), '..', 'Results_Range_Cost')

CONFIGS = [
    {"name": "F3", "relevance_definition": "Brief_plot",   "diversity_definition": "Different_years"},
    {"name": "F1", "relevance_definition": "Popularity",   "diversity_definition": "Genre_and_movie_periods"},
]

BASE_OUT = os.path.join(os.path.dirname(__file__), "fig6_movies")

for setting, is_multiple_llms, orig_dir in [("Discrete", False, ORIG_DISCRETE), ("Range", True, ORIG_RANGE)]:
    ind_assumption = True if not is_multiple_llms else False
    for cfg in CONFIGS:
        rel_def = cfg["relevance_definition"]
        div_def = cfg["diversity_definition"]
        out_dir = os.path.join(BASE_OUT, setting)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"Results_Movies_REL_{rel_def}_DIV_{div_def}.csv")
        orig_file = os.path.join(orig_dir, f"Results_Movies_REL_{rel_def}_DIV_{div_def}.csv")

        print(f"\n=== Movies {cfg['name']} | {setting} ===")

        header = ["n", "k", "method",
            "total_time_init_candidates_set", "total_time_update_bounds",
            "total_time_compute_pdf", "total_time_determine_next_question",
            "total_time_llm_response", "total_time", "api_calls"]
        with open(out_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            if os.path.exists(orig_file):
                with open(orig_file, newline='') as src:
                    reader = csv.DictReader(src)
                    for row in reader:
                        if row.get("method") in ("Max_Prob", "Naive"):
                            writer.writerow([row.get(h, "") for h in header])

        for (n, k) in experiments:
            data = merge_plots(read_data(n=n))
            results = find_top_k(
                input_query=input_query, documents=data, k=k,
                metrics=metrics, methods=methods, mock_llms=False,
                relevance_definition=rel_def, diversity_definition=div_def,
                dataset_name=dataset_name, use_MGTs=use_MGTs,
                use_filtered_init_candidates=use_filtered_init_candidates,
                independence_assumption=ind_assumption,
                is_multiple_llms=is_multiple_llms,
                win_threshold=win_threshold,
            )
            with open(out_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                for r in results:
                    writer.writerow([n, k, r.algorithm,
                        r.time.total_time_init_candidates_set,
                        r.time.total_time_update_bounds,
                        r.time.total_time_compute_pdf,
                        r.time.total_time_determine_next_question,
                        r.time.total_time_llm_response,
                        r.time.total_time, r.api_calls])

        print(f"Saved: {out_file}")
