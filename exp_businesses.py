import csv
from read_data_movies import read_data, merge_plots
from Ranking import find_top_k, store_results
from utilities import RELEVANCE, DIVERSITY, NAIVE, MAX_PROB, EXACT_BASELINE

experiments = [(50, 3), (50, 5), (50, 7)] 
dataset_name = "businesses"
input_query = "Affordable restaurant"
relevance_definition = "Type_of_food"
diversity_definition = "Open_hours"
# relevance_definition = "Location_Around_New_York"
# diversity_definition = "Cost"
use_filtered_init_candidates = True
report_entropy_in_naive = False
use_MGTs = True
metrics = [RELEVANCE, DIVERSITY]
methods = [MAX_PROB, NAIVE, EXACT_BASELINE]  
output_name = "Results_Businesses_REL_" + relevance_definition + "_DIV_" + diversity_definition
output_file = output_name + ".csv"

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
    results = find_top_k(input_query, data, k, metrics, methods, mock_llms=False, relevance_definition=relevance_definition, diversity_definition=diversity_definition, dataset_name=dataset_name, use_MGTs=use_MGTs, report_entropy_in_naive=report_entropy_in_naive, use_filtered_init_candidates = use_filtered_init_candidates)
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
