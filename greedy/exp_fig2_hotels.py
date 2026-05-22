"""
Experiment: Fig2 equivalent for Hotels (F1 and F2).
Runs the Greedy baseline; EntrRed and Random results are loaded from prior experiments.
Results written to greedy/fig2_hotels/Discrete/ and greedy/fig2_hotels/Range/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv
from read_data_hotels import read_data, merge_descriptions
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, GREEDY

experiments = [(1000, 2), (1000, 4), (1000, 6), (1000, 8), (1000, 10)]
win_threshold = 0.90
dataset_name = "hotels"
input_query = "Affordable hotel"
metrics = [RELEVANCE, DIVERSITY]
use_MGTs = True
use_filtered_init_candidates = True
methods = [GREEDY]

ORIG_DISCRETE = os.path.join(os.path.dirname(__file__), '..', 'Results_Cost')
ORIG_RANGE    = os.path.join(os.path.dirname(__file__), '..', 'Results_Range_Cost')

CONFIGS = [
    {"name": "F1", "relevance_definition": "Rating_of_the_hotel",       "diversity_definition": "Physical_distance_of_the_hotels"},
    {"name": "F2", "relevance_definition": "Distance_from_city_center",  "diversity_definition": "Star_rating"},
]

BASE_OUT = os.path.join(os.path.dirname(__file__), "fig2_hotels")

for setting, is_multiple_llms, orig_dir in [("Discrete", False, ORIG_DISCRETE), ("Range", True, ORIG_RANGE)]:
    ind_assumption = True if not is_multiple_llms else False
    for cfg in CONFIGS:
        rel_def = cfg["relevance_definition"]
        div_def = cfg["diversity_definition"]
        out_dir = os.path.join(BASE_OUT, setting)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"Results_Hotels_REL_{rel_def}_DIV_{div_def}.csv")
        orig_file = os.path.join(orig_dir, f"Results_Hotels_REL_{rel_def}_DIV_{div_def}.csv")

        print(f"\n=== Hotels {cfg['name']} | {setting} ===")

        # Copy Max_Prob + Naive rows from original results
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

        # Run Greedy and append
        for (n, k) in experiments:
            data = merge_descriptions(read_data(n=n))
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
