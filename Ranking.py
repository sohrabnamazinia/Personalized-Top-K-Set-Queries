import numpy as np
import itertools
import time
from LLMApi import LLMApi
from copy import deepcopy
import math
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE, TopKResult, read_documents, init_candidates_set, check_pair_exist, choose_2, compute_exact_scores_baseline, check_prune

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

    def set_all_random(self, seed):
        np.random.seed(seed)
        self.table = np.round(np.random.random((self.n, self.m)), 1)
        if self.n > 1:
            mask = np.triu(np.ones((self.n, self.m)), k=1)
            self.table = np.where(mask, self.table, None)
    
    def call_all(self, documents, query=None):
        # relevance table
        if query != None:
            for d in range(self.m):
                value = call_llm_relevance(query, d, documents)
                self.set(0, d, value)

        # diversity table
        else:
            for d1 in range(self.n):
                for d2 in range(self.m):
                    value = call_llm_diversity(d1, d2, documents)
                    self.set(d1, d2, value)

    def peek_value(self, i, j=0):
        if self.name == "relevance": return self.table[0, i]
        if self.name == "diversity": return self.table[i, j]

    def __str__(self):
        return f"Table(name={self.name}, shape=({self.n}, {self.m}))\n{self.table}"

def call_all_llms_relevance(input_query, documents, relevance_table, candidates_set, k, mocked_table = None):
    # relevance 
    n = relevance_table.m
    mock_table = mocked_table is not None

    for d in range(n):
        # check if llm should be mocked or not and get value based on this condition
        value = call_llm_relevance(input_query, d, documents, mocked_table) if mock_table else call_llm_relevance(input_query, d, documents)
        relevance_table.set(0, d, value)
        candidates_set, updated_candidates = update_lb_ub_relevance(candidates_set, d, value, k)
    return candidates_set, updated_candidates

def call_llm_relevance(query, d, documents, relevance_table = None):
    # Case: Mocked LLM - d is integer
    if relevance_table is not None:
        return relevance_table[0][d]
    
    # Case: Real LLM - d is the string document
    api = LLMApi()
    result = api.call_llm_relevance(query, documents[d])
    return result

def call_llm_diversity(d1, d2, documents, diversity_table = None):
    # Case: Mocked LLM - d is integer
    if diversity_table is not None:
        return diversity_table[d1][d2]
    
    # Case: Real LLM - d is the string document
    api = LLMApi()
    result = api.call_llm_diversity(documents[d1], documents[d2])
    return result

# def prob_score(bound, other_bound):
    c_lb, c_ub = bound
    o_lb, o_ub = other_bound
    # edge cases
    if c_lb > o_ub: 
        return 1
    if c_ub < o_lb:
        return 0
    if c_lb == o_lb and c_ub == o_ub:
        return 0.5
    # usual case if c and o partially overlap with c range > o range
    if c_ub > o_ub and c_lb > o_lb:
        return 0.5 + 0.5**3 + 0.5**2
    # usual case if c and o partially overlap with c range < o range
    if (c_ub < o_ub and c_lb < o_lb):
        return 0.5**3
    # other cases of complete overlap with equal bound on one end
    if (c_ub < o_ub and c_lb > o_lb) or (c_ub > o_ub and c_lb < o_lb):
        return 1/3 + 1/3*1/2
    if (c_lb == o_lb and c_ub > o_ub) or (c_ub == o_ub and c_lb > o_lb):
        return 0.5 + 0.5**2
    if (c_lb == o_lb and c_ub < o_ub) or (c_ub == o_ub and c_lb < o_lb):
        return 0.5**2

# def call_entropy(candidates_set, algorithm= None):
    if algorithm == NAIVE or algorithm == EXACT_BASELINE:
        return 0
    if len(candidates_set) == 1:
        return 0  # When only 1 candidate is left, it is clearly the winner now so entropy becomes 0 automatically
    probabilities_candidate = {}
    mock_set = deepcopy(candidates_set)
    for cand, bound in candidates_set.items():
        mock_set.pop(cand)
        for other_cand, other_bound in mock_set.items():
            prob_score_cand = prob_score(bound, other_bound)
            prob_score_other = 1 - prob_score_cand
            probabilities_candidate[cand] = probabilities_candidate.get(cand, 1) * prob_score_cand
            probabilities_candidate[other_cand] = probabilities_candidate.get(other_cand, 1) * prob_score_other
    # print(probabilities_candidate)
    entropy = -sum(map(lambda p: 0 if p==0 else p * math.log2(p), probabilities_candidate.values()))
    # print(candidates_set, entropy)
    return round(entropy, 4)

# def prob_score_dep(bound, other_bound):
    """A probability scoring function that checks if candidata > other candidate w.r.t their bounds. The bounds are update
    accordingly to indicate the range in which they are greater than the other candidate. It also calculates the complement
    probability and returns the bounds of the other candidate where it is greater than the candidate. It returns the 2 probabilities
    and 2 bounds indicating where cand > other and other > cand respectively."""
    c_lb, c_ub = bound
    o_lb, o_ub = other_bound
    
    # edge cases
    if c_lb > o_ub: 
        return 1, bound
    if c_ub < o_lb:
        return 0, None
    if c_lb == o_lb and c_ub == o_ub:
        return 0.5, bound
    # usual case if c and o partially overlap with c range > o range
    if c_ub > o_ub and c_lb > o_lb:
        prob = 0.5 + 0.5**3 + 0.5**2
        return prob, bound
    # usual case if c and o partially overlap with c range < o range
    if (c_ub < o_ub and c_lb < o_lb):
        prob = 0.5**3
        return prob, (o_lb, c_ub)
    # other cases of complete overlap with equal bound on one end
    if (c_ub < o_ub and c_lb > o_lb):
        prob = 1/3 + 1/3*1/2
        return prob, bound
    if (c_ub > o_ub and c_lb < o_lb):
        prob = 1/3 + 1/3*1/2
        return prob, (o_lb, c_ub)
    if (c_lb == o_lb and c_ub > o_ub):
        prob = 0.5 + 0.5**2
        return prob, bound
    if (c_ub == o_ub and c_lb > o_lb):
        prob = 0.5 + 0.5**2
        return prob, bound
    if (c_lb == o_lb and c_ub < o_ub): 
        prob = 0.5**2
        return prob, bound
    if (c_ub == o_ub and c_lb < o_lb):
        prob = 0.5**2
        return prob, (o_lb, c_ub)
    
# def call_entropy_dependence(candidates_set, algorithm= None):
    if algorithm == NAIVE or algorithm == EXACT_BASELINE:
        return 0
    if len(candidates_set) == 1:
        return 0  # When only 1 candidate is left, it is clearly the winner now so entropy becomes 0 automatically
    probabilities_candidate = {}
    # mock_set = deepcopy(candidates_set)
    for cand, bound in candidates_set.items():
        cand_poss_bound = bound #if cand_poss == None else cand_poss[1] # possible candidate bound initialized with its original bound if it does not already exist in prob_cand list
        for other_cand, other_bound in candidates_set.items():
            if other_cand == cand: continue
            prob_score_cand, cand_poss_bound = prob_score_dep(cand_poss_bound, other_bound)  # get the prob score and the complement score along with the possible bounds for each case
            x_val = probabilities_candidate.get(cand, 1) 
            if prob_score_cand == 0:
                probabilities_candidate[cand] = 0
                break  # if a candidate ever gets 0 probability w.r.t. other candidates, it's probability score of being the winner becomes 0, so it does not need to be computed any further
            probabilities_candidate[cand] = x_val * prob_score_cand
            
    entropy = -sum(map(lambda p: 0 if p==0 else p * math.log2(p), probabilities_candidate.values()))
    print(candidates_set, entropy)
    return round(entropy, 4)

def gen_2d(grouped_pairs:list):
    all_tables = {}
    for pairs in grouped_pairs:
        if len(pairs) == 2:
            table_name = tuple([p[0] for p in pairs])
            all_tables[table_name] = []
            bounds = [p[1] for p in pairs]
            c_lb, c_ub = bounds[0]
            o_lb, o_ub = bounds[1]
            c_lb, c_ub, o_lb, o_ub = int(c_lb*10), int(c_ub*10), int(o_lb*10), int(o_ub*10)
            for c in range(c_lb, c_ub+1):
                for o in range(o_lb, o_ub+1):
                    if c >= o:
                        all_tables[table_name].append((c,o, 1))
                    else:
                        all_tables[table_name].append((c,o, 0))
            # print(all_tables[table_name])
        elif len(pairs) == 1:
            table_name = tuple([p[0] for p in pairs])
            bounds = [p[1] for p in pairs]
            all_tables[table_name] = []
            c_lb, c_ub = bounds[0]
            c_lb, c_ub = int(c_lb*10), int(c_ub*10)
            for c in range(c_lb, c_ub+1):
                all_tables[table_name].append((c,1))
    return all_tables

# def scoring_func(cand, all_2d: dict):
    req_table = None
    req_table_vals = None
    remember_comp = []
    for table_names in all_2d.keys():
        if cand in table_names:
            req_table = table_names
            req_table_vals = all_2d[table_names]
            break
    flag = 1
    if cand == req_table[0]:
        num = [vals[:-1] for vals in req_table_vals if 1 in vals]
    else:
        num = [vals[:-1] for vals in req_table_vals if 0 in vals]
        flag = 2
    # c_vals = [v[0] for v in num]
    # remember_comp.extend(num)
    # print("req",  req_table)
    numer = len(num)
    denom = len(req_table_vals)
    prob = numer/denom
    if prob == 0: return 0
    mock_table = deepcopy(all_2d)
    mock_table.pop(req_table)
    for table_v in mock_table.values():
        count = 0
        temp = []
        # print("num", num)
        for n in num:
            for vals in table_v:    
                if flag == 1:
                    if n[0] == max(n[0], max(vals[:-1])):
                        count += 1
                        # remember_comp.append((n, vals[:-1]))
                        temp.append(n + vals[:-1])
                elif flag == 2:
                    if n[1] == max(n[1], max(vals[:-1])):
                        count += 1
                        # remember_comp.append((n, vals[:-1]))
                        temp.append(n + vals[:-1])    
        num = temp
        # print("tab", tables)
        denom = denom * len(table_v)
        numer = count
        print(count, denom)
        prob = count/denom
        if prob == 0: return 0
    return prob

def common_ele(cand, cand_bound, other, other_bound, div_tab:Metric, rel_tab:Metric):
    '''Checks for common elements between 2 candidates and accordingly calculates conditional probability'''
    common = []
    k = len(cand)
    denom_div = choose_2(k)  # denominator when the bound was calculated using diversity
    c1lb, c1ub = cand_bound
    c2lb, c2ub = other_bound
    # print(rel_tab)
    rel_c1 = sum(map(lambda doc: rel_tab.peek_value(doc), cand))/k # relevance score for c1
    rel_c2 = sum(map(lambda doc: rel_tab.peek_value(doc), other))/k # relevance score for c2
    rel_c1, rel_c2 = int(rel_c1*10), int(rel_c2*10)
    for docs in cand:
        if docs in other: common.append(docs)
    # print(k, c1, c2, common)
    if len(common) > 1:
        for i in range(len(common)):
            x = common[i]
            for y in common[i+1:]:
                # print(denom_div)
                if div_tab.peek_value(x,y) is None:
                    val = 0
                    # print(x,y,c1lb,c2lb, rel_c1, rel_c2)
                    # subtracting the rel score from lb and ub, then multiplying them with the denominator for 
                    # normalized div score to get the sum of diversity scores, then subtracting the value, after which 
                    # dividing the newly obtained sum of div scores without val with the denominator - 1 (Accounting for the val being
                    # removed) and then finally adding the rel score again to obtain the new lb without the common element
                    # print("here",((c1lb - rel_c1)*denom_div - val),((c2lb - rel_c2)*denom_div - val))
                    c1lb = (((c1lb - rel_c1)*denom_div - val)/(denom_div - 1)) + rel_c1
                    c2lb = (((c2lb - rel_c2)*denom_div - val)/(denom_div - 1)) + rel_c2
                    val = 10
                    c1ub = (((c1ub - rel_c1)*denom_div - val)/(denom_div - 1)) + rel_c1
                    c2ub = (((c2ub - rel_c2)*denom_div - val)/(denom_div - 1)) + rel_c2
                # else:
                #     val = div_tab.peek_value(x,y)
                #     print(x,y,val)
                #     c1lb = (((c1lb - rel_c1)*denom_div - val)/(denom_div - 1)) + rel_c1
                #     c2lb = (((c2lb - rel_c2)*denom_div - val)/(denom_div - 1)) + rel_c2
                #     c1ub = (((c1ub - rel_c1)*denom_div - val)/(denom_div - 1)) + rel_c1
                #     c2ub = (((c2ub - rel_c2)*denom_div - val)/(denom_div - 1)) + rel_c2
                denom_div = denom_div -1
    new_c1bnd = (c1lb, c1ub)
    new_c2bnd = (c2lb, c2ub)
    # print(cand_bound, other_bound, new_c1bnd, new_c2bnd)
    return new_c1bnd, new_c2bnd

def gen_1d(candidates_set:dict):
    oned_table = {}
    for cand, bound in candidates_set.items():
        lb, ub = bound
        lb, ub = int(lb*10), int(ub*10)
        oned_table[cand] = [i for i in range(lb, ub+1)]
    # print(oned_table)
    return oned_table

def scoring_func2(cand, all_tables: dict, diversity_table:Metric,relevance_table:Metric):
    for table_names in all_tables.keys():
        # print(cand, table_names)
        if cand == table_names:
            req_table = table_names
            req_table_vals = all_tables[table_names]
            break
    # flag = 1
    # if cand == req_table[0]:
    #     num = [vals[:-1] for vals in req_table_vals if 1 in vals]
    # else:
    #     num = [vals[:-1] for vals in req_table_vals if 0 in vals]
    #     flag = 2
    # signals = {bin(i): num[i] for i in range(len(num))}
    signals = {bin(i): req_table_vals[i] for i in range(len(req_table_vals))}
    # print(signals)
    source_node = {vals:[keys] for keys, vals in signals.items()}
    denom = len(req_table_vals)
    numer = 1
    cand_bound = (min(req_table_vals), max(req_table_vals))
    for table_n, table_v in all_tables.items():
        if table_n == req_table: continue
        inter_node = {}
        signal_counter = {}
        # print(source_node.keys())
        other_bound = (min(table_v), max(table_v))
        cand_bound_new, other_bound_new = common_ele(cand, cand_bound, table_n, other_bound, diversity_table, relevance_table)
        for sigs in source_node.values():  # Iterating through all the nodes and taking the list of signals in them
            # print(len(sigs))
            for sig in sigs: # iterating through the signals at a particular node
                for vals in table_v:  # checking if the signal is entering any of the nodes of the next table
                    # print(signals[sig], vals, cand_bound_new, other_bound_new)
                    if min(cand_bound_new) <= signals[sig] <= max(cand_bound_new) and min(other_bound_new) <= vals <= max(other_bound_new):
                        if signals[sig] >= vals: 
                            signal_counter[sig] = signal_counter.get(sig, 0) + 1
                            # print("here")
                    if signals[sig] >= vals: # if feasible signal, i.e., signal value is higher than the node value
                        # print(sig, signals[sig][0], vals[:-1])
                        # print(signals[sig], vals)
                        if vals not in inter_node: inter_node[vals] = [sig]  # then the signal enters that node
                        else: 
                            if sig not in inter_node[vals]: inter_node[vals].append(sig)
                    # if flag == 2 and signals[sig][1] >= max(vals[:-1]): # flag == 1 or 2 decides which col of the 2d table to consider as the candidate
                    #     # print(sig, signals[sig][1], vals[:-1])
                    #     signal_counter[sig] = signal_counter.get(sig, 0) + 1
                    #     if vals not in inter_node: inter_node[vals[:-1]] = [sig]
                    #     else: inter_node[vals[:-1]].append(sig)
        source_node = inter_node  # the next node becomes the source node for the next table of nodes
        
        if len(source_node) == 0: return 0 # if at any point, no source node is there then end the iteration. Can happen for 0 probability of winning
        # print(signal_counter)
        numer = sum(signal_counter.values())
        # print(cand, numer, denom)
        if numer == 0: return 0
        
        denom = denom * len(table_v)
        # print(cand, numer, denom)
    return numer/denom

def call_entropy_discrete_2D(candidates_set:dict, diversity_table:Metric,relevance_table:Metric, algorithm=None):
    if algorithm == NAIVE or algorithm == EXACT_BASELINE:
        return 0
    if len(candidates_set) == 1:
        return 0  # When only 1 candidate is left, it is clearly the winner now so entropy becomes 0 automatically
    probabilities_candidate = {}
    # print(candidates_set)
    all_1d = gen_1d(candidates_set)
    # pairs = list(candidates_set.items())
    # grouped_pairs = [pairs[i:i+2] for i in range(0, len(pairs), 2)]
    # # print(grouped_pairs)
    # all_2d = gen_2d(grouped_pairs) 
    # print(all_2d)
    ckeys = list(candidates_set.keys())
    for cand in ckeys:
        prob_score = scoring_func2(cand, all_1d, diversity_table,relevance_table)
        probabilities_candidate[cand] = prob_score
    normaliser = sum(probabilities_candidate.values())
    probabilities_candidate = {key:vals/normaliser for key,vals in probabilities_candidate.items()}
    entropy = -sum(map(lambda p: 0 if p==0 else p * math.log2(p), probabilities_candidate.values()))
    # print(probabilities_candidate)
    # print(candidates_set, entropy)
    return round(entropy, 4)

def find_top_k_lowest_overlap(input_query, documents, k, metrics, mocked_tables = None):
    # init candidates set and tables
    algorithm = LOWEST_OVERLAP
    start_time = time.time()
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)
    entropy_over_time = []
    entropy_discrete_2D = []
    # entropy_over_time.append(call_entropy(candidates_set))
    # entropy_dep_over_time.append(call_entropy_dependence(candidates_set))
    # use all relevance llm calls
    candidates_set, _ = call_all_llms_relevance(input_query, documents, relevance_table, candidates_set, k, mocked_tables[0] if mocked_tables is not None else None)
    
    # algorithm
    count = 0
    while len(candidates_set) > 1:
        # entropy = call_entropy(candidates_set)
        entropy_dep = call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table)
        # print(f"Entropy at iteration {count}: ",entropy)
        print(f"Entropy (dep) at iteration {count}: ",entropy_dep)
        # entropy_over_time.append(entropy)
        entropy_discrete_2D.append(entropy_dep)
        pair = choose_next_llm_diversity_lowest_overlap(diversity_table, candidates_set)
        if pair is not None: i, j = pair
        else: break 
        value = call_llm_diversity(i, j, documents, mocked_tables[1] if mocked_tables is not None else None)
        count += 1
        diversity_table.set(i, j, value)
        candidates_set, updated_candidates = update_lb_ub_diversity(candidates_set, (i, j), value, k)
        candidates_set = prune(candidates_set, updated_candidates)

    # entropy_over_time.append(call_entropy(candidates_set))
    entropy_discrete_2D.append(call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table))
    print("*************************************")
    print("Result - Lowest Overlap: \n", candidates_set)
    print("Total number of calls: " , count)
    # print("Final entropy: ", entropy_over_time[-1])
    duration = time.time() - start_time
    return TopKResult(algorithm, candidates_set, duration, count, entropy_discrete_2D) 


def find_top_k_Min_Uncertainty(input_query, documents, k, metrics, mocked_tables = None):
    # init candidates set and tables
    algorithm = MIN_UNCERTAINTY
    start_time = time.time()
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)
    entropy_over_time = []
    entropy_discrete_2D = []
    # entropy_over_time.append(call_entropy(candidates_set))
    # entropy_dep_over_time.append(call_entropy_dependence(candidates_set))
    # use all relevance llm calls
    candidates_set, _ = call_all_llms_relevance(input_query, documents, relevance_table, candidates_set, k, mocked_tables[0] if mocked_tables is not None else None)
    
    # algorithm
    count = 0
    while len(candidates_set) > 1:
        # entropy = call_entropy(candidates_set)
        entropy_dep = call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table)
        # print(f"Entropy at iteration {count}: ",entropy)
        print(f"Entropy (dep) at iteration {count}: ",entropy_dep)
        # entropy_over_time.append(entropy)
        entropy_discrete_2D.append(entropy_dep)
        pair = choose_next_llm_diversity(diversity_table, candidates_set)
        if pair is not None: i, j = pair
        else: break 
        value = call_llm_diversity(i, j, documents, mocked_tables[1] if mocked_tables is not None else None)
        count += 1
        diversity_table.set(i, j, value)
        candidates_set, updated_candidates = update_lb_ub_diversity(candidates_set, (i, j), value, k)
        candidates_set = prune(candidates_set, updated_candidates)
        
    # entropy_over_time.append(call_entropy(candidates_set))
    entropy_discrete_2D.append(call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table))
    print("*************************************")
    print("Result - Min Uncertainty: \n", candidates_set)
    print("Total number of calls: " , count)
    # print("Final entropy: ", entropy_over_time[-1])
    duration = time.time() - start_time
    return TopKResult(algorithm, candidates_set, duration, count, entropy_discrete_2D) 

def find_top_k_Exact_Baseline(input_query, documents, k, metrics, mocked_tables = None):
    # init candidate set and tables
    algorithm = EXACT_BASELINE
    start_time = time.time()
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    print(candidates_set)
    mock_tables = mocked_tables is not None
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)
    if mock_tables:
        relevance_table.set_all(mocked_tables[0])
        diversity_table.set_all(mocked_tables[1])
    else:
        relevance_table.call_all(documents, input_query)
        diversity_table.call_all(documents)
    
    # entropy = call_entropy(candidates_set, algorithm)
    entropy_dep = call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table, algorithm)
    print("*****************************")
    #print(relevance_table)
    #print(diversity_table)
    #print("*****************************")

    result = compute_exact_scores_baseline([relevance_table, diversity_table], candidates_set)
    print("Baseline Approach - Exact scores:\n", result)
    # print("Final entropy: ",entropy, entropy_dep)
    print("*****************************")

    duration = time.time() - start_time
    return TopKResult(algorithm, result, duration, choose_2(n), entropy_dep)

def find_top_k_Naive(input_query, documents, k, metrics, mocked_tables = None):
    # init candidate set and tables
    entropy_over_time = []
    entropy_dep_over_time = []
    algorithm = NAIVE
    start_time = time.time()
    n = len(documents)
    candidates_set = init_candidates_set(n, k, 0, len(metrics))
    # entropy_over_time.append(call_entropy(candidates_set))
    # entropy_dep_over_time.append(call_entropy_discrete_2D(candidates_set))
    mock_tables = mocked_tables is not None
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)
    if mock_tables:
        relevance_table.set_all(mocked_tables[0])
        diversity_table.set_all(mocked_tables[1])
    else:
        relevance_table.call_all(documents, input_query)
        diversity_table.call_all(documents)
    
    for d in range(relevance_table.m):
        # update bounds
        value = relevance_table.table[0][d]
        candidates_set, updated_keys = update_lb_ub_relevance(candidates_set, d, value, k)
        # prune
        candidates_set = prune(candidates_set, updated_keys)
        # entropy_over_time.append(call_entropy(candidates_set))
        # entropy_dep_over_time.append(call_entropy_discrete_2D(candidates_set))
        # print(entropy_over_time[-1])

    for i in range(diversity_table.n):
        for j in range(i + 1, diversity_table.m):
            # update bounds
            pair = (i, j)
            value = diversity_table.table[i][j]
            candidates_set, updated_keys = update_lb_ub_diversity(candidates_set, pair, value, k)
            # prune
            candidates_set = prune(candidates_set, updated_keys)
            # entropy_over_time.append(call_entropy(candidates_set))
            entropy_dep_over_time.append(call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table))
            # print(entropy_over_time[-1])
    
    # entropy_over_time.append(call_entropy(candidates_set, algorithm))
    entropy_dep_over_time.append(call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table, algorithm))
    print("The best candidate - Naive approach: \n", candidates_set)
    # print("Final entropy: ",entropy_over_time[-1])
    duration = time.time() - start_time
    return TopKResult(algorithm, candidates_set, duration, choose_2(n), entropy_dep_over_time)

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

def choose_next_llm_diversity_lowest_overlap(diversity_table, candidates_set):
    lbs, ubs, cands = [], [], []
    mock_set = deepcopy(candidates_set)
    for candidates in candidates_set.keys():
        candidate_pairs = list(itertools.combinations(candidates, 2))
        unfilled = 0
        for pair in candidate_pairs:
            i,j = pair
            if diversity_table.table[i, j] is None:
                unfilled = 1
                break
        if unfilled == 0: mock_set.pop(candidates) # Popping candidates where all possible tuples are already filled
    for cand, bound in mock_set.items():
        lbs.append(bound[0])
        ubs.append(bound[1])
        cands.append(cand)
    bounds = (lbs[0], ubs[0])
    lowest_overlap = 10000000
    highest_non_overlap = 0
    winner_cand = cands[0]
    # runner_up = cands[1]
    # print(winner_cand, runner_up)
    for i in range(len(lbs)):
        lb = lbs[i]
        ub = ubs[i]
        for r_ub in ubs:
            if r_ub == ub: continue   # if other ub matches with cand's ub its a complete overlap so we can skip it
            x = r_ub - lb   # Overlapping length
            y = ub - r_ub   # Non-overlapping length
            if x <= lowest_overlap and y >= highest_non_overlap:
                lowest_overlap = x
                highest_non_overlap = y
                # runner_up = winner_cand
                winner_cand = cands[i]
                bounds = (lb, ub)
                
    pair_uncertainty_scores = {}
    candidate_pairs = list(itertools.combinations(winner_cand, 2))
    
    # runner_up_pairs = list(itertools.combinations(runner_up, 2))
    # print(f"candidate_pairs:{candidate_pairs}")
    # req_pair = list(map(lambda x: x if x in runner_up_pairs else None, candidate_pairs))
    # req_pair = list(filter(lambda x: x is not None, req_pair))
    # print(req_pair)
    # if req_pair: return req_pair[0]     #returns the document pair that is in 2nd potential winner
    for pair in candidate_pairs:        #uses heuristic 1 if there is no common document tuple between winner and runner-up
        i,j = pair
        if diversity_table.table[i, j] is None:
            #Heuristic 1
            shared_interval_sum = 0
            for other_candidate, other_bounds in candidates_set.items(): 
                if other_candidate == winner_cand: continue
                lower_bound_shared = max(bounds[0], other_bounds[0])
                upper_bound_shared = min(bounds[1], other_bounds[1])
                if lower_bound_shared < upper_bound_shared:  # There is a common interval
                    shared_interval_sum += (upper_bound_shared - lower_bound_shared)
            pair_uncertainty_scores[pair] = pair_uncertainty_scores.get(pair, 0) + shared_interval_sum
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
                if pruned_key is not None: pruned_keys.add(pruned_key)
        for key in pruned_keys:
            candidates_set.pop(key)
    return candidates_set

def find_top_k(input_query, documents, k, metrics, methods, seed, mock_llms = False, is_output_discrete=True):
    results = []
    mocked_tables = None
    n = len(documents)
    # fill tables by mocking OR calling LLM for each cell
    if mock_llms:
        relevance_table = Metric(metrics[0], 1 ,n)
        diversity_table = Metric(metrics[1], n ,n)
        print(seed)
        relevance_table.set_all_random(seed)
        diversity_table.set_all_random(seed)
        mocked_tables = [relevance_table.table, diversity_table.table] if mock_llms else None 
        print(relevance_table)
        print(diversity_table)
    
    if EXACT_BASELINE in methods:
        results.append(find_top_k_Exact_Baseline(input_query, documents, k, metrics, mocked_tables=mocked_tables))

    if NAIVE in methods:
        results.append(find_top_k_Naive(input_query, documents, k, metrics, mocked_tables=mocked_tables))

    if MIN_UNCERTAINTY in methods:
        results.append(find_top_k_Min_Uncertainty(input_query, documents, k, metrics, mocked_tables=mocked_tables))
    
    if LOWEST_OVERLAP in methods:
        results.append(find_top_k_lowest_overlap(input_query, documents, k, metrics, mocked_tables=mocked_tables))
    
    return results

def store_results(results):
    filename = "results.txt"
    with open(filename, 'w') as file:
        file.write("Experiment Results\n")
        file.write("==================\n\n")

        for i, result in enumerate(results, 1):
            file.write(f"Algorithm {i}: {result.algorithm}\n")
            file.write("-" * (12 + len(result.algorithm)) + "\n")
            file.write(f"Candidates Set: {result.candidates_set}\n")
            file.write(f"Execution Time: {result.time:.4f} seconds\n")
            file.write(f"API Calls: {result.api_calls}\n")
            # file.write(f"Entropy over time: {result.entropy}\n")
            file.write(f"Entropy (dep) over time: {result.entropydep}\n")
            file.write("\n")

        # file.write("Summary\n")
        # file.write("-------\n")
        # file.write(f"Total Algorithms: {len(results)}\n")
        # file.write(f"Total Execution Time: {sum(r.time for r in results):.4f} seconds\n")
        # file.write(f"Total API Calls: {sum(r.api_calls for r in results)}\n")

    print(f"Results have been stored in {filename}")

# # inputs
input_query = "I need a phone which is iPhone and has great storage"
input_path = "documents.txt"
n = 10
k = 3
metrics = [RELEVANCE, DIVERSITY]
methods = [MIN_UNCERTAINTY, LOWEST_OVERLAP, EXACT_BASELINE, NAIVE]
#methods = [MIN_UNCERTAINTY]
#methods = ["Exact_Baseline", "Naive"]
mock_llms = True
seed = 42

# run
documents = read_documents(input_path, n, mock_llms)
results = find_top_k(input_query, documents, k, metrics, methods, seed, mock_llms)

# store results
store_results(results)