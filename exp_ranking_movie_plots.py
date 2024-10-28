from read_data_movies import read_data, merge_plots
from Ranking import find_top_k, store_results
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE


movies_count_count = 5
input_query = "A scary movie"
k = 2
metrics = [RELEVANCE, DIVERSITY]
methods = [LOWEST_OVERLAP, EXACT_BASELINE]

plots = merge_plots(read_data(n=movies_count_count))
#print(plots)
#print(reviews)
results = find_top_k(input_query, plots, k, metrics, methods, mock_llms=False)
store_results(results)
