from read_data_hotel_reviews import read_data, merge_reviews
from Ranking import find_top_k, store_results
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE


reviews_count = 6
input_query = "A hotel with low price and nice view"
k = 3
metrics = [RELEVANCE, DIVERSITY]
methods = [NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE]

reviews = merge_reviews(read_data(n=reviews_count))
#print(reviews)
results = find_top_k(input_query, reviews, k, metrics, methods, mock_llms=False)
store_results(results)
