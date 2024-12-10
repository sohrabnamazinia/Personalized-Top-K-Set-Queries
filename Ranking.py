import numpy as np
import itertools
import time
from LLMApi import LLMApi
from copy import deepcopy
import math
import csv
import random
import pandas as pd
import os
from utilities import RELEVANCE, DIVERSITY, NAIVE, MIN_UNCERTAINTY, MAX_PROB, EXACT_BASELINE, TopKResult, ComponentsTime, read_documents, init_candidates_set, check_pair_exist, choose_2, compute_exact_scores_baseline, check_prune, find_mgt_csv, load_init_filtered_candidates
from read_data_hotels import read_data, merge_descriptions

class Metric:
    def __init__(self, name: str, n: int, m: int, dataset_name = None):
        self.name = name
        self.dataset_name = dataset_name
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
    
    def call_all_randomized_involved(self, documents, query=None, relevance_definition=None, diversity_definition=None, sequential_randomized_length = 16668, is_image_type=False, images_directory=None):
        mean = 0.57
        std_dev = 0.1
        values_choices = [i * 0.1 for i in range(11)]

        # Relevance table
        if query is not None:
            relevance_csv = f"MGT_{self.dataset_name}_{self.m}_Rel_{relevance_definition}.csv"
            with open(relevance_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['d', 'value', 'time_rel'])

                if not is_image_type:
                    counter = 0
                    for d in range(self.m):
                        if counter % 1000 == 0:
                            print("Rel call for document: " + str(d))
                        if counter % sequential_randomized_length == 0:
                            start_time_rel = time.time()
                            value = call_llm_relevance(query, d, documents, relevance_definition=relevance_definition)
                            time_rel = time.time() - start_time_rel
                            self.set(0, d, value)
                            writer.writerow([d, value, time_rel])
                        else:
                            value = round(random.choice(values_choices), 1)
                            time_rel = np.random.normal(mean, std_dev)
                            time_rel = min(max(0.1, time_rel), 1.1)
                            self.set(0, d, value)
                            writer.writerow([d, value, time_rel])
                        counter += 1
                else:
                    counter = 0
                    for d in range(self.m):
                        if counter % 1000 == 0:
                            print("Rel call for document: " + str(d))
                        if counter % sequential_randomized_length == 0:
                            print("Calling LLM for relevance of document: " + str(d))
                            start_time_rel = time.time()
                            # NOTE: IMAGE
                            value = call_llm_image(query, d, documents, relevance_definition=relevance_definition, images_directory=images_directory)
                            time_rel = time.time() - start_time_rel
                            self.set(0, d, value)
                            writer.writerow([d, value, time_rel])
                        else:
                            value = round(random.choice(values_choices), 1)
                            time_rel = np.random.normal(mean, std_dev)
                            time_rel = min(max(0.1, time_rel), 1.1)
                            self.set(0, d, value)
                            writer.writerow([d, value, time_rel])
                        counter += 1

        # Diversity table
        else:
            diversity_csv = f"MGT_{self.dataset_name}_{self.n}_Div_{diversity_definition}.csv"
            with open(diversity_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['d2', 'd1', 'value', 'time_div'])

                counter = 0
                for d1 in range(self.n):
                    for d2 in range(d1):
                        if counter % 10000 == 0:
                            print("Div call for documents: " + str(d2) + ", " + str(d1))
                        if counter % sequential_randomized_length == 0:
                            print("calling lLM for documents: " + str(d2) + ", " + str(d1))
                            start_time_div = time.time()
                            value = call_llm_diversity(d1, d2, documents, diversity_definition=diversity_definition)
                            time_div = time.time() - start_time_div
                            self.set(d2, d1, value)
                            writer.writerow([d2, d1, value, time_div])
                        else:
                            value = round(random.choice(values_choices), 1)
                            time_div = np.random.normal(mean, std_dev)
                            time_div = min(max(0.1, time_div), 1.1)
                            self.set(d2, d1, value)
                            writer.writerow([d2, d1, value, time_div])
                        counter += 1

    
    def call_all(self, documents, query=None, relevance_definition=None, diversity_definition=None, is_image_type = False, images_directory=None):
        # Relevance table
        if query is not None:
            relevance_csv = f"MGT_{self.dataset_name}_{self.m}_Rel_{relevance_definition}.csv"
            with open(relevance_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['d', 'value', 'time_rel'])

                if not is_image_type:
                    for d in range(self.m):
                        start_time_rel = time.time()
                        value = call_llm_relevance(query, d, documents, relevance_definition=relevance_definition)
                        time_rel = time.time() - start_time_rel
                        self.set(0, d, value)
                        writer.writerow([d, value, time_rel])

                else:
                    for d in range(self.m):
                        print("Calling LLM for relevance of document: " + str(d))
                        start_time_rel = time.time()
                        # NOTE: IMAGE
                        value = call_llm_image(query, d, documents, relevance_definition=relevance_definition, images_directory=images_directory)
                        time_rel = time.time() - start_time_rel
                        self.set(0, d, value)
                        writer.writerow([d, value, time_rel])

        # Diversity table
        else:
            diversity_csv = f"MGT_{self.dataset_name}_{self.n}_Div_{diversity_definition}.csv"
            with open(diversity_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['d2', 'd1', 'value', 'time_div'])

                for d1 in range(self.n):
                    for d2 in range(d1):
                        start_time_div = time.time()
                        value = call_llm_diversity(d1, d2, documents, diversity_definition=diversity_definition)
                        time_div = time.time() - start_time_div
                        self.set(d2, d1, value)
                        writer.writerow([d2, d1, value, time_div])


    def peek_value(self, i, j=0):
        if self.name == "relevance": return self.table[0, i]
        if self.name == "diversity": return self.table[i, j]

    def __str__(self):
        return f"Table(name={self.name}, shape=({self.n}, {self.m}))\n{self.table}"

def call_all_llms_relevance(input_query, documents, relevance_table, candidates_set, k, mocked_table = None, relevance_definition=None):
    # relevance 
    n = relevance_table.m
    mock_table = mocked_table is not None

    for d in range(n):
        # check if llm should be mocked or not and get value based on this condition
        value = call_llm_relevance(input_query, d, documents, relevance_table=mocked_table) if mock_table else call_llm_relevance(input_query, d, documents, relevance_definition=relevance_definition)
        relevance_table.set(0, d, value)
        candidates_set, updated_candidates = update_lb_ub_relevance(candidates_set, d, value, k)
    return candidates_set, updated_candidates

def call_all_llms_relevance_MGT(dataset_name, relevance_table, candidates_set, k, relevance_definition=None):
    n = relevance_table.m
    mgt_df_rel = find_mgt_csv(dataset_name=dataset_name, n=n, relevance_definition=relevance_definition)
    total_time_rel = 0
    for d in range(n):
        row = mgt_df_rel.iloc[d]  
        value, time_rel = row['value'], row["time_rel"]
        total_time_rel += time_rel
        relevance_table.set(0, d, value)
        candidates_set, updated_candidates = update_lb_ub_relevance(candidates_set, d, value, k)
    return candidates_set, updated_candidates, total_time_rel


def call_llm_relevance(query, d, documents, relevance_definition=None, relevance_table = None):
    # Case: Mocked LLM - d is integer
    if relevance_table is not None:
        return relevance_table[0][d]
    
    # Case: Real LLM - d is the string document
    api = LLMApi(relevance_definition=relevance_definition)
    result = api.call_llm_relevance(query, documents[d])
    return result

def call_llm_diversity(d1, d2, documents, diversity_table = None, diversity_definition = None):
    # Case: Mocked LLM - d is integer
    if diversity_table is not None:
        return diversity_table[d1][d2]
    
    # Case: Real LLM - d is the string document
    api = LLMApi(diversity_definition=diversity_definition)
    result = api.call_llm_diversity(documents[d1], documents[d2])
    return result

def call_llm_diversity_MGT(d1, d2, mgt_df_div):
    row_index = int(((d2 * (d2 - 1)) / 2) + (d1))
    row = mgt_df_div.iloc[row_index]
    value = row['value']
    time_div = row['time_div']

    return value, time_div

def call_llm_image(query, d, images, relevance_definition=None, relevance_table = None, images_directory=None):
    # Case: Mocked LLM - d is integer
    if relevance_table is not None:
        return relevance_table[0][d]
    
    # Case: Real LLM - d is the string document
    image_path = images_directory + str(images[d].photo_id)
    api = LLMApi(relevance_definition=relevance_definition)
    result = api.call_llm_image(query, image_path)
    return result

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
    rel_c1 = rel_c1 * 10
    rel_c2 = rel_c2 *10
    # if math.ceil(rel_c1*10) > c1lb: rel_c1 = c1lb
    # else: rel_c1 = math.ceil(rel_c1*10)
    # if math.ceil(rel_c2*10) > c2lb: rel_c2 = c2lb
    # else: rel_c2 = math.ceil(rel_c2*10)
    for docs in cand:
        if docs in other: common.append(docs)
    # print(k, c1, c2, common)
    div_c1_lb = 0
    div_c1_ub = 0
    denom_c1 = 0
    div_c2_lb = 0
    div_c2_ub = 0
    denom_c2 = 0
    old_c1lb = c1lb
    old_c1ub = c1ub
    old_c2lb = c2lb
    old_c2ub = c2ub
    if len(common) > 1:
        # print(common)
        for i in range(len(cand)):
            x = cand[i]
            for y in cand[i+1:]:
                if x in common and y in common and div_tab.peek_value(x,y) is None:
                    continue
                
                div_c1_lb += 0 if div_tab.peek_value(x,y) is None else div_tab.peek_value(x,y)*10
                div_c1_ub += 10 if div_tab.peek_value(x,y) is None else div_tab.peek_value(x,y)*10
                denom_c1 += 1
                # print("Here3",(x,y), div_c1_lb, div_c1_ub, denom_c1)
        c1lb = rel_c1 + div_c1_lb/denom_c1
        c1ub = rel_c1 + div_c1_ub/denom_c1
        for i in range(len(other)):
            x = other[i]
            for y in other[i+1:]:
                if x in common and y in common and div_tab.peek_value(x,y) is None:
                    continue
                
                div_c2_lb += 0 if div_tab.peek_value(x,y) is None else div_tab.peek_value(x,y)*10
                div_c2_ub += 10 if div_tab.peek_value(x,y) is None else div_tab.peek_value(x,y)*10
                denom_c2 += 1
                # print("Here4",(x,y), div_c2_lb, div_c2_ub, denom_c2)
        c2lb = rel_c2 + div_c2_lb/denom_c2
        c2ub = rel_c2 + div_c2_ub/denom_c2
        # if old_c1lb > c1lb or old_c1ub < c1ub: print(cand_bound, c1lb, c1ub)
        # if old_c2lb > c2lb or old_c2ub < c2ub: print(other_bound, c2lb, c2ub)
        # for i in range(len(common)):
        #     x = common[i]
        #     for y in common[i+1:]:
        #         # print(denom_div)
        #         if div_tab.peek_value(x,y) is None:
        #             val = 0
        #             # print(x,y,c1lb,c2lb, rel_c1, rel_c2)
        #             # subtracting the rel score from lb and ub, then multiplying them with the denominator for 
        #             # normalized div score to get the sum of diversity scores, then subtracting the value, after which 
        #             # dividing the newly obtained sum of div scores without val with the denominator - 1 (Accounting for the val being
        #             # removed) and then finally adding the rel score again to obtain the new lb without the common element
        #             # print("here",((c1lb - rel_c1)*denom_div - val),((c2lb - rel_c2)*denom_div - val))
        #             c1lb = (((c1lb - rel_c1)*denom_div - val)/(denom_div - 1)) + rel_c1
        #             c2lb = (((c2lb - rel_c2)*denom_div - val)/(denom_div - 1)) + rel_c2
        #             val = 10
        #             c1ub = (((c1ub - rel_c1)*denom_div - val)/(denom_div - 1)) + rel_c1
        #             c2ub = (((c2ub - rel_c2)*denom_div - val)/(denom_div - 1)) + rel_c2
        #             # if c1lb > c1ub or c2lb > c2ub:
        #             #     print(cand, other, cand_bound, other_bound, c1lb, c1ub, c2lb, c2ub ,rel_c1, rel_c2)
        #             if other == (1,2,3,5,8):
        #                 print(other_bound, c2lb, c2ub, denom_div, rel_c2)
        #             # if cand == (1,2,3,5,7):
        #             #     print(cand_bound, c1lb, c1ub, denom_div, rel_c2)
        #             denom_div = denom_div -1
                # else:
                #     val = div_tab.peek_value(x,y)
                #     print(x,y,val)
                #     c1lb = (((c1lb - rel_c1)*denom_div - val)/(denom_div - 1)) + rel_c1
                #     c2lb = (((c2lb - rel_c2)*denom_div - val)/(denom_div - 1)) + rel_c2
                #     c1ub = (((c1ub - rel_c1)*denom_div - val)/(denom_div - 1)) + rel_c1
                #     c2ub = (((c2ub - rel_c2)*denom_div - val)/(denom_div - 1)) + rel_c2
                
    if math.ceil(c1lb) == math.ceil(c1ub): # say when c1lb is 9.1 and c1ub is 9.2, both their ceils are 10, so i cannot take the floor of ub in this case
        # print("Here")
        c1lb = math.ceil(c1lb)
        c1ub = math.ceil(c1ub)
    # elif 1 > c1lb - c1ub > 0:
    #     c1lb = math.floor(c1lb)
    #     c1ub = math.ceil(c1ub)
    else:
        c1lb = math.ceil(c1lb)
        c1ub = math.floor(c1ub)
    if math.ceil(c2lb) == math.ceil(c2ub):
        # print("Here2")
        c2lb = math.ceil(c2lb)
        c2ub = math.ceil(c2ub)
    # elif 1 > c2lb - c2ub > 0:
    #     c2lb = math.floor(c2lb)
    #     c2ub = math.ceil(c2ub)
    else:
        c2lb = math.ceil(c2lb)
        c2ub = math.floor(c2ub)
    new_c1bnd = (c1lb, c1ub)
    new_c2bnd = (c2lb, c2ub)
    
    # if c1lb > c1ub or c2lb > c2ub:
    # print("Here1",cand, other, cand_bound, other_bound, new_c1bnd, new_c2bnd,rel_c1, rel_c2)
    if (cand_bound[0] > new_c1bnd[0] or cand_bound[1] < new_c1bnd[1] or other_bound[0] > new_c2bnd[0] or other_bound[1] < new_c2bnd[1]):
        print("Here2",cand, other, cand_bound, other_bound, new_c1bnd, new_c2bnd, rel_c1, rel_c2)
    # if cand == (0,1,2):
    #     if other == (0,1,4) or other == (0,2,4) or other == (1,2,4):
    #         print(cand_bound, other_bound, new_c1bnd, new_c2bnd,rel_c1, rel_c2)
    return new_c1bnd, new_c2bnd

def gen_1d(candidates_set:dict):
    oned_table = {}
    for cand, bound in candidates_set.items():
        lb, ub = bound
        # print(lb,ub, np.round(lb, 1), np.round(ub,1))
        lb, ub = math.floor(lb*10), math.ceil(ub*10)
        # print(lb,ub)
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
    signals = {bin(i): req_table_vals[i] for i in range(len(req_table_vals))}
    # print(signals)
    source_node = {vals:[keys] for keys, vals in signals.items()}
    # numer = 1
    prob = 1
    cand_bound = (min(req_table_vals), max(req_table_vals))
    # denom = len(req_table_vals)
    current_internode_access_count = 1
    next_inter_node_access_count = 1
    for table_n, table_v in all_tables.items():
        if table_n == req_table: continue
        inter_node = {}
        signal_counter = {}
        # print(source_node.keys())
        other_bound = (min(table_v), max(table_v))
        # print(cand_bound, other_bound)
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
                        if vals not in inter_node: inter_node[vals] = [sig]  # then the signal enters that node
                        else: 
                            if sig not in inter_node[vals]: inter_node[vals].append(sig)
        source_node = inter_node  # the next node becomes the source node for the next table of nodes
        next_inter_node_access_count = len(inter_node.keys())
        if len(source_node) == 0: return 0 # if at any point, no source node is there then end the iteration. Can happen for 0 probability of winning
        # print(signal_counter)
        numer = sum(signal_counter.values())
        # print(cand, numer, denom)
        if numer == 0: return 0
        # print(cand_bound_new, other_bound_new)
        denom = (cand_bound_new[1] - cand_bound_new[0]+1) * current_internode_access_count * (other_bound_new[1] - other_bound_new[0]+1) 
        # if cand == (0,1,2):
        #     if table_n == (0,1,4) or table_n == (0,2,4) or table_n == (1,2,4):
        #         print(cand, numer, denom, table_n)
        current_internode_access_count = next_inter_node_access_count
        # print(numer, denom)
        prob *=  numer/denom
        prob = prob
    
    return prob

def call_entropy_discrete_2D(candidates_set:dict, diversity_table:Metric,relevance_table:Metric, algorithm=None):
    if algorithm == NAIVE or algorithm == EXACT_BASELINE:
        return 0, None
    if len(candidates_set) == 1:
        return 0, None  # When only 1 candidate is left, it is clearly the winner now so entropy becomes 0 automatically
    probabilities_candidate = {}
    # print(candidates_set)
    all_1d = gen_1d(candidates_set)
    # print(all_1d)
    ckeys = list(candidates_set.keys())
    counter = 0
    for cand in ckeys:
        print("Computing probability for candidate number: " + str(counter))
        prob_score = scoring_func2(cand, all_1d, diversity_table,relevance_table)
        probabilities_candidate[cand] = prob_score
        counter += 1
    # print(probabilities_candidate)
    normaliser = sum(probabilities_candidate.values())
    if normaliser == 0:
        print("Warning: Normalizer is zero. Cannot normalize probabilities.")
        probabilities_candidate = {key: 0 for key in probabilities_candidate}  # Set all to 0 or handle as needed
    else:
        probabilities_candidate = {key: vals / normaliser for key, vals in probabilities_candidate.items()}
    # print(probabilities_candidate)
    entropy = -sum(map(lambda p: 0 if p==0.0 else p * math.log2(p), probabilities_candidate.values()))
    entropy = 0.0 if entropy == -0.0 else entropy
    # print(round(entropy, 4),probabilities_candidate)
    # print(candidates_set, entropy)
    return round(entropy, 3), probabilities_candidate

def prob_score(bound, other_bound):
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
        return (c_ub-o_ub)/(c_ub-c_lb)*1 + (o_ub-c_lb)/(c_ub-c_lb)*0.5
    # usual case if c and o partially overlap with c range < o range
    if (c_ub < o_ub and c_lb < o_lb) or (c_ub == o_ub and c_lb < o_lb):
        return (c_ub-o_lb)/(c_ub-c_lb)*0.5
    # other cases of complete overlap with equal bound on one end
    if (c_ub < o_ub and c_lb > o_lb) or (c_ub == o_ub and c_lb > o_lb) or (c_lb == o_lb and c_ub < o_ub): 
        return 0.5
    if (c_ub > o_ub and c_lb < o_lb) or (c_lb == o_lb and c_ub > o_ub):
        return (c_ub-o_ub)/(c_ub-c_lb)*1 + (o_ub-o_lb)/(c_ub-c_lb)*0.5 

def call_entropy_ind(candidates_set, algorithm= None):
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

def find_top_k_max_prob(input_query, documents, k, metrics, mocked_tables = None, relevance_definition = None, diversity_definition = None, use_MGTs = False, dataset_name=None, use_filtered_init_candidates = False):
    # init candidates set and tables
    algorithm = MAX_PROB
    n = len(documents)
    start_time_init_candidates_set = time.time()
    total_time_update_bounds = 0
    total_time_determine_next_question = 0
    total_time_llm_response = 0
    total_time_compute_pdf = 0
    if not use_filtered_init_candidates:
        candidates_set = init_candidates_set(n, k, 0, len(metrics))
    else:
        candidates_set = load_init_filtered_candidates(dataset_name=dataset_name, relevance_definition=relevance_definition, diversity_definition=diversity_definition, k=k)
    total_time_init_candidates_set = time.time() - start_time_init_candidates_set
    relevance_table = Metric(metrics[0], 1 ,n)
    diversity_table = Metric(metrics[1], n ,n)
    entropy_ind_over_time = []
    entropy_discrete_2D = []
    # entropy_over_time.append(call_entropy(candidates_set))
    # entropy_dep_over_time.append(call_entropy_dependence(candidates_set))
    # use all relevance llm calls
    #print(mocked_tables[0])
    if not use_MGTs:
        candidates_set, _ = call_all_llms_relevance(input_query, documents, relevance_table, candidates_set, k, mocked_tables[0] if mocked_tables is not None else None, relevance_definition=relevance_definition)
    else:
        mgt_df_div = find_mgt_csv(dataset_name=dataset_name, n=n, diversity_definition=diversity_definition)
        candidates_set, _, total_time_llm_response_rel = call_all_llms_relevance_MGT(dataset_name=dataset_name, relevance_table=relevance_table, candidates_set=candidates_set, relevance_definition=relevance_definition, k=k)
        total_time_llm_response += total_time_llm_response_rel
    # algorithm
    count = n
    determined_qs = []
    its = 1
    # print(candidates_set)
    while len(candidates_set) > 1:
        print("Iteration: " + str(its) + ", No. Candidates: " + str(len(candidates_set)))
        # entropy = call_entropy(candidates_set)
        start_time_compute_pdf = time.time()
        entropy_dep, probabilities_cand = call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table)
        total_time_compute_pdf += time.time() - start_time_compute_pdf

        entropy_ind = call_entropy_ind(candidates_set)
        # print(f"Entropy at iteration {count}: ",entropy)
        #print(f"Entropy (dep) at iteration {its}: ",entropy_dep)
        # entropy_over_time.append(entropy)
        entropy_discrete_2D.append(entropy_dep)
        entropy_ind_over_time.append(entropy_ind)
        start_time_determine_next_question = time.time()
        pair = choose_next_llm_diversity_max_prob(diversity_table, candidates_set, probabilities_cand, determined_qs)
        total_time_determine_next_question += time.time() - start_time_determine_next_question
        if pair is not None: i, j = pair
        else: break 
        if not use_MGTs:
            start_time_llm_response = time.time()
            value = call_llm_diversity(i, j, documents, diversity_table=mocked_tables[1] if mocked_tables is not None else None, diversity_definition=diversity_definition)
            total_time_llm_response += time.time() - start_time_llm_response
        else:
            value, time_div = call_llm_diversity_MGT(i, j, mgt_df_div)
            total_time_llm_response += time_div
        count += 1
        diversity_table.set(i, j, value)
        start_time_update_bounds = time.time()
        candidates_set, updated_candidates = update_lb_ub_diversity(candidates_set, (i, j), value, k)
        total_time_update_bounds += time.time() - start_time_update_bounds
        candidates_set = prune(candidates_set, updated_candidates)
        its+=1

    # entropy_over_time.append(call_entropy(candidates_set))
    entropy_dep, probabilities_cand = call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table)
    entropy_discrete_2D.append(entropy_dep)
    entropy_ind = call_entropy_ind(candidates_set)
    entropy_ind_over_time.append(entropy_ind)
    #print(relevance_table)
    #print(diversity_table)
    print("*************************************")
    print("Result - Max probability: \n", candidates_set)
    print("Total number of calls: " , count)
    # print("Final entropy: ", entropy_over_time[-1])
    componentsTime = ComponentsTime(total_time_init_candidates_set=total_time_init_candidates_set, total_time_update_bounds=total_time_update_bounds, total_time_compute_pdf=total_time_compute_pdf, total_time_determine_next_question=total_time_determine_next_question, total_time_llm_response=total_time_llm_response)
    return TopKResult(algorithm, candidates_set, componentsTime, count, entropy_discrete_2D) 


def find_top_k_Exact_Baseline(input_query, documents, k, metrics, mocked_tables = None, relevance_definition = None, diversity_definition = None, use_MGTs = False, dataset_name=None, use_filtered_init_candidates = False):
    # init candidate set and tables
    algorithm = EXACT_BASELINE
    start_time = time.time()
    total_time_llm_response = 0
    n = len(documents)
    if not use_filtered_init_candidates:
        candidates_set = init_candidates_set(n, k, 0, len(metrics))
    else:
        candidates_set = load_init_filtered_candidates(dataset_name=dataset_name, relevance_definition=relevance_definition, diversity_definition=diversity_definition, k=k)
    #print(candidates_set)
    #mock_tables = mocked_tables is not None
    relevance_table = Metric(metrics[0], 1 ,n, dataset_name)
    diversity_table = Metric(metrics[1], n ,n, dataset_name)
    if not use_MGTs:
        start_time_llm_response = time.time()
        candidates_set, _ = call_all_llms_relevance(input_query, documents, relevance_table, candidates_set, k, mocked_tables[0] if mocked_tables is not None else None, relevance_definition=relevance_definition)
        total_time_llm_response += time.time() - start_time_llm_response
    else:
        mgt_df_div = find_mgt_csv(dataset_name=dataset_name, n=n, diversity_definition=diversity_definition)
        candidates_set, _, total_time_llm_response_rel = call_all_llms_relevance_MGT(dataset_name=dataset_name, relevance_table=relevance_table, candidates_set=candidates_set, relevance_definition=relevance_definition, k=k)
        total_time_llm_response += total_time_llm_response_rel
    count = n
    for i in range(n-1):
        for j in range(i+1,n):
            if not use_MGTs:
                start_time_llm_response = time.time()
                value = call_llm_diversity(i, j, documents, diversity_table=mocked_tables[1] if mocked_tables is not None else None, diversity_definition=diversity_definition)
                total_time_llm_response += time.time() - start_time_llm_response
            else:
                value, time_div = call_llm_diversity_MGT(i, j, mgt_df_div)
                total_time_llm_response += time_div
            diversity_table.set(i, j, value)
            
    # entropy = call_entropy(candidates_set, algorithm)
    entropy_dep = call_entropy_discrete_2D(candidates_set,diversity_table,relevance_table, algorithm)
    entropy_ind = call_entropy_ind(candidates_set, algorithm)
    print("*****************************")
    # print(relevance_table)
    # print(diversity_table)
    #print("*****************************")
    result = compute_exact_scores_baseline([relevance_table, diversity_table], candidates_set)
    #print("Baseline Approach - Exact scores:\n", result)
    # print("Final entropy: ",entropy, entropy_dep)
    print("*****************************")
    total_time_exclude_llm_calls =time.time() - start_time
    duration = ComponentsTime(total_time=total_time_exclude_llm_calls + total_time_llm_response)
    return TopKResult(algorithm, result, duration, choose_2(n) + n, entropy_dep)

def find_top_k_Naive(input_query, documents, k, metrics, mocked_tables = None, relevance_definition = None, diversity_definition = None, use_MGTs = False, dataset_name=None, report_entropy=False, use_filtered_init_candidates = False):
    # init candidate set and tables
    entropy_ind_over_time = []
    entropy_dep_over_time = []
    algorithm = NAIVE
    start_time = time.time()
    n = len(documents)
    count = n
    total_time_llm_response = 0
    #print(n, choose_2(n))
    if not use_filtered_init_candidates:
        candidates_set = init_candidates_set(n, k, 0, len(metrics))
    else:
        candidates_set = load_init_filtered_candidates(dataset_name=dataset_name, relevance_definition=relevance_definition, diversity_definition=diversity_definition, k=k)
    # entropy_over_time.append(call_entropy(candidates_set))
    # entropy_dep_over_time.append(call_entropy_discrete_2D(candidates_set))
    mock_tables = mocked_tables is not None
    relevance_table = Metric(metrics[0], 1 ,n, dataset_name)
    #diversity_table = Metric(metrics[1], n ,n, dataset_name)
    diversity_table2 = Metric(metrics[1], n ,n, dataset_name)
    if not use_MGTs:
        candidates_set, _ = call_all_llms_relevance(input_query, documents, relevance_table, candidates_set, k, mocked_tables[0] if mocked_tables is not None else None, relevance_definition=relevance_definition)
    else:
        mgt_df_div = find_mgt_csv(dataset_name=dataset_name, n=n, diversity_definition=diversity_definition)
        candidates_set, _, total_time_llm_response_rel = call_all_llms_relevance_MGT(dataset_name=dataset_name, relevance_table=relevance_table, candidates_set=candidates_set, relevance_definition=relevance_definition, k=k)
        total_time_llm_response += total_time_llm_response_rel    
    already_qsd = []
    # print(candidates_set)
    if report_entropy:
        entropy, _ = call_entropy_discrete_2D(candidates_set,diversity_table2,relevance_table)
        entropy_dep_over_time.append(entropy)
        entropy_ind = call_entropy_ind(candidates_set)
        entropy_ind_over_time.append(entropy_ind)
    its = 0
    while(len(candidates_set) > 1):
        # print(candidates_set)
        its += 1
        setofdocs = set()
        for cand in candidates_set.keys():
            for docs in cand:
                setofdocs.add(docs)
        i, j = choose_random_qs(setofdocs, already_qsd)
        pair = (i,j)
        # print(already_qsd)
        if not use_MGTs:
            start_time_llm_response = time.time()
            value = call_llm_diversity(i, j, documents, diversity_table=mocked_tables[1] if mocked_tables is not None else None, diversity_definition=diversity_definition)
            total_time_llm_response += time.time() - start_time_llm_response
        else:
            value, time_div = call_llm_diversity_MGT(i, j, mgt_df_div)
            total_time_llm_response += time_div
        diversity_table2.set(i, j, value)
        count += 1
        candidates_set, updated_keys = update_lb_ub_diversity(candidates_set, pair, value, k)
        # prune
        candidates_set = prune(candidates_set, updated_keys)
        if report_entropy:
            entropy, _ = call_entropy_discrete_2D(candidates_set,diversity_table2,relevance_table)
            entropy_dep_over_time.append(entropy)
            entropy_ind = call_entropy_ind(candidates_set)
            entropy_ind_over_time.append(entropy_ind)
        #print(f"Entropy at iteration {its} for Naive approach: ", entropy_dep_over_time[-1])

    # print(relevance_table)
    # print(diversity_table2)
    print("The best candidate - Naive approach: \n", candidates_set)
    # print("Final entropy: ",entropy_over_time[-1])
    total_time_exclude_llm =time.time() - start_time
    duration = ComponentsTime(total_time=total_time_exclude_llm + total_time_llm_response)
    return TopKResult(algorithm, candidates_set, duration, count, entropy_dep_over_time)

def choose_random_qs(setofdocs, already_qsd):
    setofdocs = list(setofdocs)
    temp = deepcopy(setofdocs)
    i = np.random.choice(np.array(temp))
    temp.remove(i)
    j = np.random.choice(np.array(temp))
    # print(i, j)
    pair = (i,j)
    while(i>=j or pair in already_qsd):
        temp = deepcopy(setofdocs)
        i = np.random.choice(np.array(temp))
        temp.remove(i)
        j = np.random.choice(np.array(temp))
        pair = (i, j)
        # print(i, j)
    already_qsd.append(pair)
    return i, j

def choose_next_llm_diversity_max_prob(diversity_table, candidates_set, probabilities_cand, determined_qs):       
    # print(probabilities_cand)
    winner_cand = max(probabilities_cand, key=probabilities_cand.get)     
    candidate_pairs = list(itertools.combinations(winner_cand, 2))
    candidate_pairs_temp = deepcopy(candidate_pairs)
    flag = False
    winner_cand_og = (winner_cand, probabilities_cand[winner_cand], candidates_set[winner_cand])
    while(flag == False):
        for pair in candidate_pairs_temp:
            if pair in determined_qs:
                candidate_pairs.remove(pair)
        if len(candidate_pairs) == 0:
            probabilities_cand.pop(winner_cand)
            if len(probabilities_cand) == 0:
                print("All possible questions already asked, remaining candidates have equal scores!")
                return None
            winner_cand = max(probabilities_cand, key=probabilities_cand.get)     
            candidate_pairs = list(itertools.combinations(winner_cand, 2))
            candidate_pairs_temp = deepcopy(candidate_pairs)
            flag = False
        else: 
            flag = True
    # print(determined_qs)
    #if winner_cand != winner_cand_og: print("Original winner:", winner_cand_og)
    # sorted_set = {k: v for k, v in sorted(candidates_set.items(), key=lambda x: (x[1][1], x[1][0]))}
    # print(sorted_set.popitem())
    #print("Chosen winner: ", winner_cand, probabilities_cand[winner_cand], candidates_set[winner_cand], candidate_pairs)
    pair_uncertainty_scores = {}
    for pair in candidate_pairs:
        i,j = pair
        p_cands_with_pair = 0
        p_cands_without_pair = 0
        for cand in probabilities_cand.keys():
            if check_pair_exist(cand, pair):
                p_cands_with_pair += probabilities_cand[cand]
            else: p_cands_without_pair += probabilities_cand[cand]
        pair_score = np.abs(p_cands_with_pair - p_cands_without_pair)
        pair_uncertainty_scores[pair] = pair_score
    # Find the pair with the maximum uncertainty score
    if pair_uncertainty_scores:
        max_pair = max(pair_uncertainty_scores, key=pair_uncertainty_scores.get)
        determined_qs.append(max_pair)
        # print(max_pair, pair_uncertainty_scores)
        return max_pair
    else:   
        return None  # In case no valid pair is found


def update_lb_ub_relevance(candidates_set, d, value, k):
    updated_candidates = []
    for candidate in candidates_set:
        if d in candidate:
            updated_candidates.append(candidate)
            new_lb = np.round(candidates_set[candidate][0] + (value / k), 3)
            new_ub = np.round(candidates_set[candidate][1] - ((1 - value) / k), 3)
            candidates_set[candidate] = (new_lb, new_ub)

    return candidates_set, updated_candidates

def update_lb_ub_diversity(candidates_set, pair, value, k):
    updated_candidates = []
    for candidate in candidates_set:
        if check_pair_exist(candidate, pair):
            updated_candidates.append(candidate)
            # print(candidate, candidates_set[candidate][0], candidates_set[candidate][1], value)
            new_lb = np.round(candidates_set[candidate][0] + (value / choose_2(k)), 3)
            new_ub = np.round(candidates_set[candidate][1] - ((1 - value) / choose_2(k)), 3)
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

def find_top_k(input_query, documents, k, metrics, methods, seed = 42, mock_llms = False, is_output_discrete=True, relevance_definition = None, diversity_definition = None, dataset_name = None, use_MGTs = False, report_entropy_in_naive=False, use_filtered_init_candidates=False):
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
        # print(mocked_tables)
    
    if EXACT_BASELINE in methods:
        results.append(find_top_k_Exact_Baseline(input_query, documents, k, metrics, mocked_tables=mocked_tables, relevance_definition=relevance_definition, diversity_definition=diversity_definition, dataset_name=dataset_name, use_MGTs=use_MGTs, use_filtered_init_candidates=use_filtered_init_candidates))

    if NAIVE in methods:
        results.append(find_top_k_Naive(input_query, documents, k, metrics, mocked_tables=mocked_tables, relevance_definition=relevance_definition, diversity_definition=diversity_definition, dataset_name=dataset_name, use_MGTs=use_MGTs, report_entropy=report_entropy_in_naive, use_filtered_init_candidates=use_filtered_init_candidates))

    # if MIN_UNCERTAINTY in methods:
        # results.append(find_top_k_Min_Uncertainty(input_query, documents, k, metrics, mocked_tables=mocked_tables, relevance_definition=relevance_definition, diversity_definition=diversity_definition))
    
    if MAX_PROB in methods:
        results.append(find_top_k_max_prob(input_query, documents, k, metrics, mocked_tables=mocked_tables, relevance_definition=relevance_definition, diversity_definition=diversity_definition, dataset_name=dataset_name, use_MGTs=use_MGTs, use_filtered_init_candidates=use_filtered_init_candidates))
    
    return results

def store_results(results, output_name=None):
    if output_name == None:
        filename = "results.txt"
    else:
        filename = output_name + ".txt"
    with open(filename, 'w') as file:
        file.write("Experiment Results\n")
        file.write("==================\n\n")
        for i, result in enumerate(results, 1):
            file.write(f"Algorithm {i}: {result.algorithm}\n")
            file.write("-" * (12 + len(result.algorithm)) + "\n")
            file.write(f"Candidates Set: {result.candidates_set}\n")
            file.write(f"Execution Time: {result.time.total_time:.4f} seconds\n")
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

def compute_exact_scores_MGT(n, k, dataset_name, relevance_definition, diversity_definition):
    candidates_set = init_candidates_set(n, k, 0, 2)
    metrics_names = [RELEVANCE, DIVERSITY]
    relevance_table = Metric(metrics_names[0], 1, n, dataset_name)
    diversity_table = Metric(metrics_names[1], n, n, dataset_name)

    relevance_file = f"MGT_Results/MGT_{dataset_name}_10000_Rel_{relevance_definition}.csv"
    mgt_df_rel = pd.read_csv(relevance_file)
    for i in range(n):
        relevance_value = mgt_df_rel.iloc[i]["value"]
        relevance_table.set(0, i, relevance_value)

    diversity_file = f"MGT_Results/MGT_{dataset_name}_10000_Div_{diversity_definition}.csv"
    mgt_df_div = pd.read_csv(diversity_file)
    for d1 in range(n):
        for d2 in range(d1):
            diversity_value, _ = call_llm_diversity_MGT(d1, d2, mgt_df_div)
            diversity_table.set(d2, d1, diversity_value)

    metrics = [relevance_table, diversity_table]
    candidates_scores = compute_exact_scores_baseline(candidates_set=candidates_set, metrics=metrics)
    return candidates_scores

def filter_init_candidates(n, k, dataset_name, relevance_definition, diversity_definition, number_of_final_candidates, random_selection=False):
    if not random_selection:
        candidates_scores = compute_exact_scores_MGT(n, k, dataset_name, relevance_definition, diversity_definition)
        sorted_candidates = sorted(candidates_scores.items(), key=lambda item: item[1], reverse=True)
        final_candidates = sorted_candidates[:number_of_final_candidates]
        return [candidate[0] for candidate in final_candidates]
    else:
         # Get the candidate scores
        candidates_scores = compute_exact_scores_MGT(n, k, dataset_name, relevance_definition, diversity_definition)
            
        # Get the list of candidates (keys of the dictionary)
        candidates_list = list(candidates_scores.items())
            
        # Randomly choose the specified number of final candidates
        final_candidates = random.sample(candidates_list, number_of_final_candidates)
            
        # Return only the candidate part (without their scores)
        return [candidate[0] for candidate in final_candidates]



def store_top_k_candidates(n, k, dataset_name, relevance_definition, diversity_definition, number_of_final_candidates, random_selection=False):
    os.makedirs("FIC_Results", exist_ok=True)

    final_candidates = filter_init_candidates(
        n, k, dataset_name, relevance_definition, diversity_definition, number_of_final_candidates, random_selection=random_selection
    )

    output_file_path = f"FIC_Results/FIC_{dataset_name}_Rel_{relevance_definition}_Div_{diversity_definition}_{k}.csv"

    with open(output_file_path, mode="w", newline="") as output_file:
        writer = csv.writer(output_file)
        for candidate in final_candidates:
            writer.writerow(candidate[:k])  

    print(f"Results stored in {output_file_path}")

def store_all_top_k_candidates(n_values, k_values, number_of_final_candidates, radnom_selection=False):
    # Define the datasets, relevance, and diversity definitions
    datasets = [
        ("hotels", "Rating_of_the_hotel", "Physical_distance_of_the_hotels"),
        ("hotels", "Distance_from_city_center", "Star_rating"),
        ("movies", "Popularity", "Genre_and_movie_periods"),
        ("movies", "Brief_plot", "Different_years"),
        ("businesses", "Type_of_food", "Open_hours"),
        ("businesses", "Location_Around_New_York", "Cost")
    ]

    # Loop through each dataset and relevance/diversity definition
    for dataset_name, rel_def, div_def in datasets:
        for i in range(len(n_values)):
            n = n_values[i]
            k = k_values[i]
            # Call store_top_k_candidates for each combination
            store_top_k_candidates(
                n=n,
                k=k,
                dataset_name=dataset_name,
                relevance_definition=rel_def,
                diversity_definition=div_def,
                number_of_final_candidates=number_of_final_candidates, random_selection=radnom_selection
            )

# # # inputs
# input_query = "Affordable hotel"
# relevance_definition = "Rating_of_the_hotel"
# diversity_definition = "Physical_distance_of_the_hotels"
# input_path = "documents.txt"
# dataset_name = "hotels"
# n = 100
# k = 3
# metrics = [RELEVANCE, DIVERSITY]
# methods = [MAX_PROB, NAIVE, EXACT_BASELINE]
# mock_llms = False
# use_MGTs = True
# use_filtered_init_candidates = False
# seed = 42

# # run
# #documents = read_documents(input_path, n, mock_llms)
# data = merge_descriptions(read_data(n=n))
# results = find_top_k(input_query=input_query, documents=data, k=k, metrics=metrics, methods=methods, seed=seed, mock_llms=mock_llms, relevance_definition=relevance_definition, diversity_definition=diversity_definition, dataset_name=dataset_name, use_MGTs=True, use_filtered_init_candidates=use_filtered_init_candidates)

# # store results
# store_results(results)