import csv
from read_data_movies import read_data, merge_plots
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE

experiments = [(6, 3)] 
input_query = "A scary movie"
relevance_definition = "Popularity of the movie"
diversity_definition = "Genre and movie periods"
metrics = [RELEVANCE, DIVERSITY]
methods = [LOWEST_OVERLAP, EXACT_BASELINE]  
output_file = "experiment_movies.csv"

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

for (n, k) in experiments:
    data = merge_plots(read_data(n=n))
    results = find_top_k(input_query, data, k, metrics, methods, mock_llms=False, relevance_definition=relevance_definition, diversity_definition=diversity_definition)
    
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

print("Experiment completed. Combined results saved in:", output_file)
