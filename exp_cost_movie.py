import csv
from read_data_movies import read_data, merge_plots
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE

# List of (n, k) tuples for experimentation
experiments = [(10, 2), (10, 4)]  
input_query = "A scary movie"
relevance_definition = "Popularity of the movie"
diversity_definition = "Genre and movie periods"
metrics = [RELEVANCE, DIVERSITY]
methods = [LOWEST_OVERLAP]
output_file = "experiment_cost_movies.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["n", "k", "method", "api_calls"])

for (n, k) in experiments:
    plots = merge_plots(read_data(n=n))  
    results = find_top_k(input_query, plots, k, metrics, methods, mock_llms=False, relevance_definition=relevance_definition, diversity_definition=diversity_definition)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for result in results:
            writer.writerow([n, k, result.algorithm, result.api_calls])  

print("Experiment completed. Results saved in", output_file)
