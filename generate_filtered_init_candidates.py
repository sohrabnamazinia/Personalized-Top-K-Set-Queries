from Ranking import store_all_top_k_candidates

n_values = [50, 40, 30]
k_values = [3, 5, 7]
number_of_final_candidates = 100
random_selection = True

store_all_top_k_candidates(n_values, k_values, number_of_final_candidates, radnom_selection=random_selection)
