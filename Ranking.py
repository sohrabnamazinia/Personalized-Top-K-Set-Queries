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


# NOTE: Mock version
def read_documents(path=None, n=4):
    result = []
    for i in range(n):
        result.append("")
    return result

def init_candidates_set(n, k, lb_init_value, ub_init_value):
    combinations = itertools.combinations(range(n), k)
    candidates_set = {(combination): (lb_init_value, ub_init_value) for combination in combinations}
    return candidates_set

# NOTE
def call_llm_relevance(d):
    pass

# NOTE
def call_llm_diversity(d1, d2):
    pass

# NOTE
def find_top_k(documents, k):
    pass

# NOTE
def update_lb_ub_relevance(d, value):
    pass

# NOTE
def update_lb_ub_diversity(d1, d2, value):
    pass

def mock_table_diversity(table):
    table.table = np.round(np.random.random((table.n, table.m)), 1)

def mock_table_relevance(table):
    table.table = np.round(np.random.random((table.n, table.m)), 1)

def find_top_k_naive(documents, k, metrics):
    # init candidate set and bounds
    candidates_list = init_candidates_set(n, k, 0, len(metrics))
    UBs = 0
    LBs = 0
    # create tables
    n = len(documents)
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)

    # fill tables
    mock_table_diversity(diversity_table)
    mock_table_relevance(relevance_table)

    return None

# input_query = ""
# n = 3
# k = 2
#metrics = ["relevance", "diversity"]
# documents = read_documents(n)
# result = find_top_k_naive(documents, k, metrics)
# print(result)