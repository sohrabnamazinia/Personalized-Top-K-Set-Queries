from Ranking import store_all_top_k_candidates, store_top_k_candidates

n_values = [500, 500, 500, 500, 500]
k_values = [2, 4, 6, 8, 10]
number_of_final_candidates = 500
random_selection = True

store_all_top_k_candidates(n_values, k_values, number_of_final_candidates, radnom_selection=random_selection)

# The higher the n values, the higher the cost difference between EntrRed and Random
# The higher the number of candidates, the higher the cost difference between EntrRed and Random