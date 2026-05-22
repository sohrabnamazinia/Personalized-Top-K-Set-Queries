"""
Experiment: Fig4 equivalent — time to determine next question as #candidates varies.
Runs the Greedy baseline; EntrRed results are loaded from prior scalability experiments.
Results written to greedy/fig4_scalability/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv
from read_data_hotels import read_data, merge_descriptions
from read_data_movies import read_data as read_movies, merge_plots
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, GREEDY

n_values = [45, 101, 142, 173, 201]
k = 2
win_threshold = 0.90
use_MGTs = True
use_filtered_init_candidates = False
methods = [GREEDY]

ORIG_SCALABILITY = os.path.join(os.path.dirname(__file__), '..', 'Results_Scalibility')

DATASETS = [
    {
        "name": "hotels",
        "input_query": "Affordable hotel",
        "relevance_definition": "Rating_of_the_hotel",
        "diversity_definition": "Physical_distance_of_the_hotels",
        "reader": lambda n: merge_descriptions(read_data(n=n)),
        "out_name": "Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv",
    },
    {
        "name": "movies",
        "input_query": "A popular movie",
        "relevance_definition": "Brief_plot",
        "diversity_definition": "Different_years",
        "reader": lambda n: merge_plots(read_movies(n=n)),
        "out_name": "Results_Movies_REL_Brief_plot_DIV_Different_years.csv",
    },
]

BASE_OUT = os.path.join(os.path.dirname(__file__), "fig4_scalability")
os.makedirs(BASE_OUT, exist_ok=True)

metrics = [RELEVANCE, DIVERSITY]
header = ["n", "k", "method",
    "total_time_init_candidates_set", "total_time_update_bounds",
    "total_time_compute_pdf", "total_time_determine_next_question",
    "total_time_llm_response", "total_time", "api_calls"]

for ds in DATASETS:
    out_file = os.path.join(BASE_OUT, ds["out_name"])
    orig_file = os.path.join(ORIG_SCALABILITY, ds["out_name"])
    print(f"\n=== Fig4 Scalability: {ds['name']} ===")

    # Pre-populate with original Max_Prob rows
    with open(out_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        if os.path.exists(orig_file):
            with open(orig_file, newline='') as src:
                reader = csv.DictReader(src)
                for row in reader:
                    if row.get("method") == "Max_Prob":
                        writer.writerow([row.get(h, "") for h in header])

    # Run Greedy and append
    for n in n_values:
        data = ds["reader"](n)
        results = find_top_k(
            input_query=ds["input_query"], documents=data, k=k,
            metrics=metrics, methods=methods, mock_llms=False,
            relevance_definition=ds["relevance_definition"],
            diversity_definition=ds["diversity_definition"],
            dataset_name=ds["name"], use_MGTs=use_MGTs,
            use_filtered_init_candidates=use_filtered_init_candidates,
            independence_assumption=True,   # ProbInd required for Discrete mode
            is_multiple_llms=False,
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
        print(f"  n={n} done")

    print(f"Saved: {out_file}")
