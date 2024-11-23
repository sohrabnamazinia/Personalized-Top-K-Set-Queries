from Ranking import store_all_top_k_candidates, store_top_k_candidates

n_values = [50, 40, 30, 20, 15]
k_values = [2, 4, 6, 8, 10]
number_of_final_candidates = 100
random_selection = True

store_all_top_k_candidates(n_values, k_values, number_of_final_candidates, radnom_selection=random_selection)
