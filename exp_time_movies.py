import csv
from read_data_movies import read_data, merge_plots
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE

# List of (n, k) tuples for experimentation
experiments = [(10, 2)]
input_query = "A scary movie"
metrics = [RELEVANCE, DIVERSITY]
methods = [MIN_UNCERTAINTY]
output_file = "experiment_time_movies.csv"

# Open CSV file and write the header for time measurements
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "n", "k", "method", 
        "total_time_init_candidates_set", 
        "total_time_update_bounds", 
        "total_time_compute_pdf", 
        "total_time_determine_next_question", 
        "total_time_llm_response",
        "total_time"
    ])

for (n, k) in experiments:
    plots = merge_plots(read_data(n=n))
    results = find_top_k(input_query, plots, k, metrics, methods, mock_llms=False)
    
    # Append each result's timing details into the CSV
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
                result.time.total_time
            ])

print("Experiment completed. Results saved in", output_file)
