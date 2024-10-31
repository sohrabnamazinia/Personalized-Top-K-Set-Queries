from enum import Enum
import itertools

LOWEST_OVERLAP, MIN_UNCERTAINTY, EXACT_BASELINE, NAIVE = "Lowest_Overlap", "Min_Uncertainty", "Exact_Baseline", "Naive"
RELEVANCE, DIVERSITY = "relevance", "diversity"


class TopKResult:
    def __init__(self, algorithm, candidates_set, time, api_calls, entropydep) -> None:
        self.algorithm = algorithm
        self.candidates_set = candidates_set
        self.time = time
        self.api_calls = api_calls
        # self.entropy = entropy
        self.entropydep = entropydep

class Dataset(Enum):
    HOTEL_REVIEWS : 1
    MOVIE_PLOTS : 2
    

def read_documents(input_file=None, n=4, mock_llms=False):
    if mock_llms:
        return [""] * n
    with open(input_file, 'r') as file:
        result = file.read().splitlines()
    return result

def init_candidates_set(n, k, lb_init_value, ub_init_value):
    combinations = itertools.combinations(range(n), k)
    candidates_set = {(combination): (lb_init_value, ub_init_value) for combination in combinations}
    return candidates_set

def check_pair_exist(candidate, pair):
    return (pair[0] in candidate and pair[1] in candidate)

def choose_2(k):
    if k < 2:
        return 0  
    return k * (k - 1) // 2

def compute_exact_scores_baseline(metrics, candidates_set):
    relevance_table = metrics[0].table
    diversity_table = metrics[1].table
    k = len(next(iter(candidates_set)))
    result = {}

    for candidate, _ in candidates_set.items():
        relevance_scores = [relevance_table[0, doc] for doc in candidate]
        relevance_score = sum(relevance_scores) / k
        
        diversity_scores = 0
        candidate_list = list(candidate)
        for i in range(k):
            for j in range(i + 1, k):
                diversity_scores += (diversity_table[candidate_list[i], candidate_list[j]])
        diversity_score = diversity_scores / choose_2(k)
        
        total_score = relevance_score + diversity_score
        
        result[candidate] = total_score
    
    return result

def check_prune(tuple_1, tuple_2):
    candidate_1, bounds_1 = tuple_1[0], tuple_1[1]
    candidate_2, bounds_2 = tuple_2[0], tuple_2[1]
    if bounds_1[0] >= bounds_2[1]: return candidate_2
    if bounds_2[0] >= bounds_1[1]: return candidate_1
    return None  

