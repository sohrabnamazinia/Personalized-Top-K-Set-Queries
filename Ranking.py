import numpy as np
import itertools

class Metric:
    def __init__(self, name: str, n: int, m: int):
        self.name = name
        self.table = np.zeros((n, m), dtype=float)
        self.n = n
        self.m = m

    def set(self, i: int, j: int, value: float):
        if 0 <= i < self.n and 0 <= j < self.m:
            self.table[i, j] = value
        else:
            raise IndexError("Index out of bounds")
    
    def set_all(self, table):
        if table.shape == (self.n, self.m):
            self.table = table
        else:
            raise IndexError("New Table has dimension conflict")

    def set_all_random(self):
        self.table = np.round(np.random.random((self.n, self.m)), 1)
        if self.n > 1:
            mask = np.triu(np.ones((self.n, self.m)), k=1)
            self.table = np.where(mask, self.table, -1)

    
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

def compute_all_bounds_baseline(metrics, candidates_set):
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

# NOTE
def call_llm_relevance(query, d):
    pass

# NOTE
def call_llm_diversity(d1, d2):
    pass

# NOTE
def find_top_k(input_query, documents, k, metrics):
    # init candidate set 
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    print(candidates_set)
    # create tables
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)

    # relevance 
    for d in range(n):
        value = call_llm_relevance(input_query, d)
        relevance_table.set(d, 1, value)
        candidates_set, updated_candidates = update_lb_ub_relevance(candidates_set, d, value, k)
    
    while len(candidates_set) > 1:
        i, j = choose_next_llm_diversity(diversity_table, candidates_set)
        value = call_llm_diversity(i, j)
        diversity_table.set(i, j, value)
        candidates_set, updated_candidates = update_lb_ub_relevance(candidates_set, d, value, k)
        candidates_set = prune(candidates_set, updated_candidates)

    return candidates_set

def choose_next_llm_diversity(diversity_table, candidates_set):
    return 0, 0

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

def find_top_k_naive(input_query, documents, k, metrics):
    # init candidate set 
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    print(candidates_set)
    # create tables
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)

    # fill tables by mocking OR calling LLM for each cell
    diversity_table.set_all_random()
    relevance_table.set_all_random()

    print("*****************************")
    print(relevance_table)
    print(diversity_table)
    print("*****************************")

    result = compute_all_bounds_baseline([relevance_table, diversity_table], candidates_set)
    print("Baseline Approach - Exact scores:\n", result)
    print("*****************************")

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
n = 3
k = 2
metrics = ["relevance", "diversity"]
documents = read_documents(n=n)
result = find_top_k_naive(input_query, documents, k, metrics)
print(result)

