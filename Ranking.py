import numpy as np
import itertools

class Metric:
    def __init__(self, name: str, n: int, m: int):
        self.name = name
        self.table = np.zeros((n, m), dtype=float)
        self.n = n
        self.m = m

    def Set(self, i: int, j: int, value: float):
        if 0 <= i < self.n and 0 <= j < self.m:
            self.table[i, j] = value
        else:
            raise IndexError("Index out of bounds")
    
    def update_bounds(UBs, LBs, index):
        pass
        
    def __str__(self):
        return f"Table(name={self.name}, shape=({self.n}, {self.m}))\n{self.table}"


# NOTE
def read_documents(path=None, n=4):
    result = []
    for i in range(n):
        result.append("")
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

def check_prune(tuple_1, tuple_2):
    candidate_1, bounds_1 = tuple_1[0], tuple_1[1]
    candidate_2, bounds_2 = tuple_2[0], tuple_2[1]
    if bounds_1[0] >= bounds_2[1]: return candidate_2
    if bounds_2[0] >= bounds_1[1]: return candidate_1
    return None  

# NOTE
def call_llm_relevance(query, d):
    pass

# NOTE
def call_llm_diversity(d1, d2):
    pass

# NOTE
def find_top_k(documents, k):
    pass

def update_lb_ub_relevance(candidates_set, d, value, k):
    updated_candidates = []
    for candidate in candidates_set:
        if d in candidate:
            updated_candidates.append(candidate)
            new_lb = candidates_set[candidate][0] + (value / k)
            new_ub = candidates_set[candidate][1] - ((1 - value) / k)
            candidates_set[candidate] = (new_lb, new_ub)

    return candidates_set, updated_candidates

def update_lb_ub_diversity(candidates_set, pair, value, k):
    updated_candidates = []
    for candidate in candidates_set:
        if check_pair_exist(candidate, pair):
            updated_candidates.append(candidate)
            new_lb = candidates_set[candidate][0] + (value / choose_2(k))
            new_ub = candidates_set[candidate][1] - ((1 - value) / choose_2(k))
            candidates_set[candidate] = (new_lb, new_ub)

    return candidates_set, updated_candidates

def prune(candidates_set, updated_keys):
    for updated_key in updated_keys:
        # check if it has already been pruned
        if updated_key not in candidates_set.keys(): continue
        pruned_keys = set()
        for key in candidates_set:
            if key != updated_key:
                pruned_key = check_prune((updated_key, candidates_set[updated_key]), (key, candidates_set[key]))
                if pruned_key != None: pruned_keys.add(pruned_key)
        for key in pruned_keys:
            candidates_set.pop(key)
    return candidates_set

def compute_all_bounds_baseline():
    pass

def mock_table_diversity(table):
    table.table = np.round(np.random.random((table.n, table.m)), 1)
    mask = np.triu(np.ones((table.n, table.m)), k=1)
    table.table = np.where(mask, table.table, -1)

def mock_table_relevance(table):
    table.table = np.round(np.random.random((table.n, table.m)), 1)

def find_top_k_naive(input_query, documents, k, metrics):
    # init candidate set 
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    print(candidates_set)
    # create tables
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)

    # fill tables by mocking OR calling LLM for each cell
    mock_table_diversity(diversity_table)
    mock_table_relevance(relevance_table)
    print(relevance_table)
    print(diversity_table)

    for d in range(relevance_table.m):
        # update bounds
        value = relevance_table.table[0][d]
        candidates_set, updated_keys = update_lb_ub_relevance(candidates_set, d, value, k)

        # prune
        candidates_set = prune(candidates_set, updated_keys)

    for i in range(diversity_table.n):
        for j in range(i + 1, diversity_table.m):
            # update bounds
            pair = (i, j)
            value = diversity_table.table[i][j]
            candidates_set, updated_keys = update_lb_ub_diversity(candidates_set, pair, value, k)

            # prune
            candidates_set = prune(candidates_set, updated_keys)
    
    return candidates_set

input_query = ""
n = 4
k = 3
metrics = ["relevance", "diversity"]
documents = read_documents(n=n)
result = find_top_k_naive(input_query, documents, k, metrics)
print(result)

