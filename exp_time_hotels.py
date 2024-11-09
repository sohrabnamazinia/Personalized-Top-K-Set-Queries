import csv
from read_data_hotels import read_data, merge_descriptions
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE

# List of (n, k) tuples for experimentation
experiments = [(3, 2)]
dataset_name = "hotels"
input_query = "Affordable hotel"
relevance_definition = "Rating of the hotel"
diversity_definition = "Physical distance of the hotel"
metrics = [RELEVANCE, DIVERSITY]
methods = [EXACT_BASELINE]
output_file = "experiment_time_hotels.csv"

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
    plots = merge_descriptions(read_data(n=n))
    results = find_top_k(input_query, plots, k, metrics, methods, mock_llms=False, relevance_definition=relevance_definition, diversity_definition=diversity_definition, dataset_name=dataset_name)
    
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
