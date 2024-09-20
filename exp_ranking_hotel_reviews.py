from read_data_hotel_reviews import read_data, merge_reviews
from Ranking import find_top_k
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE


reviews_count = 4
input_query = ""
k = 2
metrics = [RELEVANCE, DIVERSITY]
methods = [NAIVE]

reviews = merge_reviews(read_data(n=reviews_count))
results = find_top_k(input_query, reviews, k, metrics, methods, mock_llms=False)
print(results)
