import numpy as np
import itertools
import copy

class Metric:
    def __init__(self, name: str, n: int, m: int):
        self.name = name
        self.table = np.full((n, m), None, dtype=object)
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
            self.table = np.where(mask, self.table, None)
    
    def call_all(self, documents, query=None):
        # relevance table
        if query != None:
            for i in range(self.m):
                d = documents[i]
                value = call_llm_relevance(query, d)
                self.set(1, i, value)

        # diversity table
        else:
            for i in range(self.n):
                d1 = documents[i]
                for j in range(self.m):
                    d2 = documents[j]
                    value = call_llm_diversity(d1, d2)
                    self.set(i, j, value)

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

def call_all_llms_relevance(input_query, relevance_table, candidates_set, k, mocked_table = False):
    # relevance 
    n = relevance_table.m
    mock_table = mocked_table != None

    for d in range(n):
        # check if llm should be mocked or not and get value based on this condition
        value = call_llm_relevance(input_query, d, mocked_table) if mock_table else call_llm_relevance(input_query, d)
        relevance_table.set(d, 1, value)
        candidates_set, updated_candidates = update_lb_ub_relevance(candidates_set, d, value, k)
    return candidates_set, updated_candidates

# NOTE
def call_llm_relevance(query, d, relevance_table = None):
    # Case: Mocked LLM - d is integer
    if relevance_table != None:
        return relevance_table[d]
    
    # Case: Real LLM - d is the string document

# NOTE
def call_llm_diversity(d1, d2, diversity_table = None):
    # Case: Mocked LLM - d is integer
    if diversity_table != None:
        return diversity_table[d1][d2]
    
    # Case: Real LLM - d is the string document

def find_top_k_Min_Uncertainty(input_query, documents, k, metrics, mocked_tables = None):
    # init candidates set and tables
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    print(candidates_set)
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)
    mock_tables = mocked_tables != None
    if mock_tables:
        relevance_table.set_all(mocked_tables[0])
        diversity_table.set_all(mocked_tables[1])

    # use all relevance llm calls
    candidates_set, _ = call_all_llms_relevance(input_query, relevance_table, candidates_set, k, mocked_tables[0])
    
    # algorithm
    while len(candidates_set) > 1:
        i, j = choose_next_llm_diversity(diversity_table, candidates_set)
        value = call_llm_diversity(i, j, mocked_tables[1])
        diversity_table.set(i, j, value)
        candidates_set, updated_candidates = update_lb_ub_diversity(candidates_set, (i, j), value, k)
        candidates_set = prune(candidates_set, updated_candidates)

    return candidates_set

def find_top_k_Exact_Baseline(input_query, documents, k, metrics, mocked_tables = None):
    # init candidate set and tables
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    print(candidates_set)
    mock_tables = mocked_tables != None
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)
    if mock_tables:
        relevance_table.set_all(mocked_tables[0])
        diversity_table.set_all(mocked_tables[1])
    else:
        relevance_table.call_all(documents, input_query)
        diversity_table.call_all(documents)
    
    print("*****************************")
    print(relevance_table)
    print(diversity_table)
    print("*****************************")

    result = compute_exact_scores_baseline([relevance_table, diversity_table], candidates_set)
    print("Baseline Approach - Exact scores:\n", result)
    print("*****************************")

    return result

def find_top_k_Naive(input_query, documents, k, metrics, mocked_tables = None):
    # init candidate set and tables
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    print(candidates_set)
    mock_tables = mocked_tables != None
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)
    if mock_tables:
        relevance_table.set_all(mocked_tables[0])
        diversity_table.set_all(mocked_tables[1])
    else:
        relevance_table.call_all(documents, input_query)
        diversity_table.call_all(documents)
    
    print("*****************************")
    print(relevance_table)
    print(diversity_table)
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

def choose_next_llm_diversity(diversity_table, candidates_set):
    pair_uncertainty_scores = {}

    for candidate, bounds in candidates_set.items():
        candidate_pairs = list(itertools.combinations(candidate, 2))
        for pair in candidate_pairs:
            i, j = pair
            # Only consider pairs with a None value in the diversity table
            if diversity_table.table[i, j] is None:
                shared_interval_sum = 0
                for other_candidate, other_bounds in candidates_set.items():
                    if other_candidate == candidate: continue
                    lower_bound_shared = max(bounds[0], other_bounds[0])
                    upper_bound_shared = min(bounds[1], other_bounds[1])
                    if lower_bound_shared < upper_bound_shared:  # There is a common interval
                        shared_interval_sum += (upper_bound_shared - lower_bound_shared)

                if pair in pair_uncertainty_scores:
                    pair_uncertainty_scores[pair] += shared_interval_sum
                else:
                    pair_uncertainty_scores[pair] = shared_interval_sum

    # Find the pair with the maximum uncertainty score
    if pair_uncertainty_scores:
        max_pair = max(pair_uncertainty_scores, key=pair_uncertainty_scores.get)
        return max_pair
    else:   
        return None  # In case no valid pair is found

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

def find_top_k(input_query, documents, k, metrics, methods, mock_llms = False):
    # fill tables by mocking OR calling LLM for each cell
    if mock_llms:
        relevance_table = Metric(metrics[0], 1 ,n)
        diversity_table = Metric(metrics[1], n ,n)
        relevance_table.set_all_random()
        diversity_table.set_all_random()
        mocked_tables = [relevance_table.table, diversity_table.table] if mock_llms else None 
    
    if "Exact_Baseline" in methods:
        find_top_k_Exact_Baseline(input_query, documents, k, metrics, mocked_tables=mocked_tables)

    if "Naive" in methods:
        find_top_k_Naive(input_query, documents, k, metrics, mocked_tables=mocked_tables)

    if "Min_Uncertainty" in methods:
        find_top_k_Min_Uncertainty(input_query, documents, k, metrics, mocked_tables=mocked_tables)

# set inputs
input_query = ""
n = 3
k = 2
metrics = ["relevance", "diversity"]
methods = ["Exact_Baseline", "Naive", "Min_Uncertainty"]
mock_llms = True

documents = read_documents(n=n)
result = find_top_k(input_query, documents, k, metrics, methods, mock_llms)
print(result)

