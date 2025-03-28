import csv
from read_data_movies import read_data, merge_plots
from Ranking import find_top_k, store_results
from utilities import RELEVANCE, DIVERSITY, NAIVE, MAX_PROB, EXACT_BASELINE, get_unique_filename

experiments = [(1000, 2), (1000, 4), (1000, 6), (1000, 8), (1000, 10)] 
#experiments = [(45, 2), (64, 2), (90, 2), (142, 2), (201, 2)]  
win_threshold = 0.90
dataset_name = "movies"
input_query = "A scary movie"
relevance_definition = "Popularity" 
diversity_definition = "Genre_and_movie_periods"
# relevance_definition = "Brief_plot" 
# diversity_definition = "Different_years"
use_filtered_init_candidates = True
independence_assumption = True
report_entropy_in_naive = False
is_multiple_llms = True
use_MGTs = True
metrics = [RELEVANCE, DIVERSITY]
methods = [MAX_PROB, NAIVE, EXACT_BASELINE]  
output_name = "Results_Movies_REL_" + relevance_definition + "_DIV_" + diversity_definition
output_file = get_unique_filename(output_name+ ".csv")


with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "n", "k", "method", 
        "total_time_init_candidates_set", 
        "total_time_update_bounds", 
        "total_time_compute_pdf", 
        "total_time_determine_next_question", 
        "total_time_llm_response",
        "total_time",
        "api_calls"  
    ])

all_results = []
for (n, k) in experiments:
    data = merge_plots(read_data(n=n))
    results = find_top_k(input_query=input_query, documents=data, k=k, metrics=metrics, methods=methods, mock_llms=False, relevance_definition=relevance_definition, diversity_definition=diversity_definition, dataset_name=dataset_name, use_MGTs=use_MGTs, report_entropy_in_naive=report_entropy_in_naive, use_filtered_init_candidates=use_filtered_init_candidates, independence_assumption=independence_assumption, is_multiple_llms=is_multiple_llms, win_threshold=win_threshold)
    all_results.extend(results)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for result in results:
            writer.writerow([
                n, k, result.algorithm,
                result.time.total_time_init_candidates_set,
                result.time.total_time_update_bounds,
                result.time.total_time_compute_pdf,
                result.time.total_time_determine_next_question,
                result.time.total_time_llm_response,
                result.time.total_time,
                result.api_calls  
            ])

store_results(all_results, output_name=output_name)
print("Experiment completed. Combined results saved in:", output_file)
