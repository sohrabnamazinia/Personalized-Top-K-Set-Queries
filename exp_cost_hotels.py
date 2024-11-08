import csv
from read_data_hotels import read_data, merge_descriptions
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE

# List of (n, k) tuples for experimentation
experiments = [(10, 2)]  
input_query = "Affordable hotel"
relevance_definition = "Rating of the hotel"
diversity_definition = "Physical distance of the hotel"
metrics = [RELEVANCE, DIVERSITY]
methods = [LOWEST_OVERLAP, MIN_UNCERTAINTY]
output_file = "experiment_cost_hotels.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["n", "k", "method", "api_calls"])

for (n, k) in experiments:
    hotels = merge_descriptions(read_data(n=n))  
    results = find_top_k(input_query, hotels, k, metrics, methods, mock_llms=False, relevance_definition=relevance_definition, diversity_definition=diversity_definition)
    
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for result in results:
            writer.writerow([n, k, result.algorithm, result.api_calls])  

print("Experiment completed. Results saved in: ", output_file)
