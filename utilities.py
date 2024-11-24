from enum import Enum
import itertools
import os
import csv
import pandas as pd
from math import comb

MAX_PROB, MIN_UNCERTAINTY, EXACT_BASELINE, NAIVE = "Max_Prob", "Min_Uncertainty", "Exact_Baseline", "Naive"
RELEVANCE, DIVERSITY = "relevance", "diversity"

class TopKResult:
    def __init__(self, algorithm, candidates_set, time, api_calls, entropydep) -> None:
        self.algorithm = algorithm
        self.candidates_set = candidates_set
        self.time = time
        self.api_calls = api_calls
        # self.entropy = entropy
        self.entropydep = entropydep

class BusinessPhoto:
    def __init__(self, photo_id, photo_info):
        self.photo_id = photo_id
        self.photo_info = photo_info

    def __repr__(self):
        return f"BusinessPhoto(photo_id={self.photo_id}, photo_info={self.photo_info})"

class ComponentsTime:
    def __init__(self, total_time_init_candidates_set = None, total_time_update_bounds = None, total_time_compute_pdf = None, total_time_determine_next_question = None, total_time_llm_response = None, total_time=None) -> None:
        self.total_time_init_candidates_set = total_time_init_candidates_set
        self.total_time_update_bounds = total_time_update_bounds
        self.total_time_compute_pdf = total_time_compute_pdf
        self.total_time_determine_next_question = total_time_determine_next_question
        self.total_time_llm_response = total_time_llm_response
        if total_time != None:
            self.total_time = total_time
        else:
            self.total_time = total_time_init_candidates_set + total_time_update_bounds + total_time_compute_pdf + total_time_determine_next_question + total_time_llm_response
    
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

def find_mgt_csv(dataset_name, n, relevance_definition=None, diversity_definition=None, create_if_not_exists=True):
    if diversity_definition is None:
        mgt_file_name = f"MGT_{dataset_name}_{n}_Rel_{relevance_definition}.csv"
    if relevance_definition is None:
        mgt_file_name = f"MGT_{dataset_name}_{n}_Div_{diversity_definition}.csv"
    
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "MGT_Results", mgt_file_name)
    
    if os.path.isfile(file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file: {e}")
    elif create_if_not_exists:
        # Look for the file with n=10000
        if relevance_definition is None:
            alt_file_name = f"MGT_{dataset_name}_10000_Div_{diversity_definition}.csv"
        elif diversity_definition is None:
            alt_file_name = f"MGT_{dataset_name}_10000_Rel_{relevance_definition}.csv"
        alt_file_path = os.path.join(current_dir, "MGT_Results", alt_file_name)
        
        if os.path.isfile(alt_file_path):
            try:
                alt_df = pd.read_csv(alt_file_path)
                # Determine subset size
                if diversity_definition is None:
                    subset_size = n
                elif relevance_definition is None:
                    subset_size = comb(n, 2)
                subset_df = alt_df.head(subset_size)
                # Save the new subset file
                subset_file_path = file_path  # Same file path as the original search
                subset_df.to_csv(subset_file_path, index=False)
                print(f"Subset CSV file created: {subset_file_path}")
                return subset_df
            except Exception as e:
                raise RuntimeError(f"Error reading alternative CSV file: {e}")
        else:
            raise FileNotFoundError(f"The file {mgt_file_name} was not found, and an alternative file with n=10000 was also not found.")
    else:
        raise FileNotFoundError(f"The file {mgt_file_name} was not found in the MGT_Results directory.")

import csv
import os

def load_init_filtered_candidates(dataset_name, relevance_definition, diversity_definition, k):
    # Construct the file name based on the provided inputs
    file_name = f"FIC_{dataset_name}_Rel_{relevance_definition}_Div_{diversity_definition}_{k}.csv"
    file_path = os.path.join("FIC_Results", file_name)
    
    # Initialize the candidates_set dictionary
    candidates_set = {}
    
    try:
        # Open and read the CSV file
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            
            # Process each row in the CSV
            for row in csv_reader:
                candidate = tuple(map(int, row))  # Convert row elements to integers
                lb_init_value = 0.0  # Placeholder for initial lower bound
                ub_init_value = 2.0  # Placeholder for initial upper bound
                candidates_set[candidate] = (lb_init_value, ub_init_value)
                
    except FileNotFoundError:
        print(f"Error: File {file_name} not found in the 'FIC_Results' directory.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return candidates_set

def get_unique_filename(base_name):
    name, ext = os.path.splitext(base_name)
    counter = 1
    while os.path.exists(base_name):
        if "_" in name and name.rsplit('_', 1)[-1].isdigit():
            base_name = f"{name.rsplit('_', 1)[0]}_{counter}{ext}"
        else:
            base_name = f"{name}_{counter}{ext}"
        counter += 1
    return base_name




