from read_data_buisinesses import read_data
from Ranking import Metric
from utilities import RELEVANCE, DIVERSITY, CHATGPT, LLAMA
import time

# Inputs for MGT
experiments = [10000]  
sequential_randomized_length_relevance = 50000
sequential_randomized_length_diversity = 80000
# sequential_randomized_length_diversity = 2
#sequential_randomized_length_diversity = 15000
dataset_name = "businesses"
input_query = "Affordable restaurant"
relevance_definition = "Type_of_food"
diversity_definition = "Open_hours"
llm_model = LLAMA
metrics = [RELEVANCE, DIVERSITY]
# relevance_definition = "Location_Around_New_York"
# diversity_definition = "Cost"
images_directory = "dataset_businesses/businesses_photos/"

for n in experiments:
    buisinesses_info, buisinesses_photos = read_data(n=n)

    relevance_table = Metric(metrics[0], 1 ,n, dataset_name)
    diversity_table = Metric(metrics[1], n ,n, dataset_name)
    
    start_time = time.time()
    relevance_table.call_all_randomized_involved_range(buisinesses_photos, input_query, relevance_definition=relevance_definition, sequential_randomized_length = sequential_randomized_length_relevance, is_image_type=True, images_directory=images_directory, llm_model=llm_model)
    diversity_table.call_all_randomized_involved_range(buisinesses_info, diversity_definition=diversity_definition, sequential_randomized_length = sequential_randomized_length_diversity, llm_model=llm_model)
    duration = time.time() - start_time

    print(f"{time.time()}: Experiment for dataset {dataset_name} with n={n}, relevance definition '{relevance_definition}', and diversity definition '{diversity_definition}' is done.")
    print(f"Duration: {duration:.2f} seconds")