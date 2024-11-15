from read_data_hotels import read_data, merge_descriptions
from Ranking import Metric
from utilities import RELEVANCE, DIVERSITY
import time

# Inputs for MGT
experiments = [4]  
sequential_randomized_length = 2
#sequential_randomized_length = 16667
dataset_name = "hotels"
input_query = "Affordable hotel"
relevance_definition = "Rating_of_the_hotel"
diversity_definition = "Physical_distance_of_the_hotels"
metrics = [RELEVANCE, DIVERSITY]
#  dataset_name = "movies"
#  input_query = "A scary movie"
#  relevance_definition = "Popularity of the movie"
#  diversity_definition = "Genre and movie periods"

for n in experiments:
    data = merge_descriptions(read_data(n=n))

    relevance_table = Metric(metrics[0], 1 ,n, dataset_name)
    diversity_table = Metric(metrics[1], n ,n, dataset_name)
    
    start_time = time.time()
    relevance_table.call_all_randomized_involved(data, input_query, relevance_definition=relevance_definition, sequential_randomized_length = sequential_randomized_length)
    diversity_table.call_all_randomized_involved(data, diversity_definition=diversity_definition, sequential_randomized_length = sequential_randomized_length)
    duration = time.time() - start_time

    print(f"{time.time()}: Experiment for dataset {dataset_name} with n={n}, relevance definition '{relevance_definition}', and diversity definition '{diversity_definition}' is done.")
    print(f"Duration: {duration:.2f} seconds")