# Personalized Top-k Set Queries Over Predicted Scores

This repository contains the source code for the experiments presented in *Personalized Top-k Set Queries Over Predicted Scores*. The project introduces a computational framework to efficiently identify personalized top-k sets using minimal oracle (e.g., LLM) calls.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Results](#results)
  - [Cost Analysis](#cost-analysis)
  - [Scalability Analysis](#scalability-analysis)


## Overview

In this work, we introduce a framework for answering personalized top-k queries over multi-modal data. Our approach is centered around two key contributions:

1. **Personalization**: We utilize scoring functions that can be decomposed into individual components, with each component having user-defined definitions. This flexibility allows the scoring function to be tailored to the preferences and needs of each user, making the ranking process more personalized.

2. **Efficiency**: We leverage Large Language Models (LLMs) to evaluate data based on the personalized scoring function. More importantly, we minimize the computational cost associated with LLMs. Here, cost is defined by the number of LLM calls required to return the top-k results, and our framework aims to reduce this number, enhancing efficiency while maintaining accuracy.

## Requirements
- Python 3.11 or above
- Required libraries:
  - LangChain
  - NumPy
  - Matplotlib

## Installation
To get started with the framework, follow these steps:

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/sohrabnamazinia/Personalized-Top-K-Set-Queries
2. **Install the libraries**:
   ```bash
   pip install Langhain
   pip install Numpy
   pip install Matplotlib

3. **Optional: Configure API Keys for LLMs**:  
   Ourframework has the option to either call LLMs in real time or to use the pre-stored results of LLM calls done by us for materializing the ground truth. If you want to use the first option, follow these steps to create and configure the key:

   1. Go to the [OpenAI API Key Creation Page](https://platform.openai.com/account/api-keys).
   2. Sign in or create an account if you don't have one.
   3. Once logged in, click on "Create new secret key" to generate your API key.
   4. Copy the generated key.

   After obtaining your API key, export it as an environment variable by running the following command in your terminal:

   ```bash
   export OPENAI_API_KEY="your_api_key_here"  # On Windows: set OPENAI_API_KEY=your_api_key_here


## Running 
**Run the Environments**:  
   There are three key runnable files corresponding to the different use cases in our experiments:

   1. `hotels.py`
   2. `movies.py`
   3. `businesses.py`

   After executing each file, a CSV file is generated containing the results for that specific run and use case. Each CSV file will include the output data for the respective experiment configuration.


**Input Parameters in Runnable Files**:  
   The following key input parameters can be adjusted for each run of the framework:

   - `experiments = [(1000, 2), (1000, 4), (1000, 6), (1000, 8), (1000, 10)]`: A list of tuples where each tuple represents (n, k) values where n is the number of entities in that experiment, and k is the number of entities to be returned (top-k set). 
   - `dataset_name = "hotels"`: Defines the dataset being used (e.g., hotels). It can be either of the following options: "hotels", "movies", "businesses"
   - `input_query = "Affordable hotel"`: The input query for ranking the data. It is user defined to make the scoring function personalized. 
   - `relevance_definition = "Distance_from_city_center"`: Defines the relevance criterion. It is user defined to make the scoring function personalized. 
   - `diversity_definition = "Star_rating"`: Defines the diversity criterion. It is user defined to make the scoring function personalized. 
   - `metrics = [RELEVANCE, DIVERSITY]`: Specifies the metrics to be used in ranking (e.g., relevance and diversity).
   - `use_MGTs = True`: Whether to use pre-stored materialzed ground truth (set it to True) or to call LLMs in real time (set it to False). 
   - `independence_assumption = False`: Whether to assume independence between candidates (ProbInd) or not (ProbDep).
   - `use_filtered_init_candidates = True`: Whether to use filtered initial candidates alredy stored as the top candidates or not. 
   - `report_entropy_in_naive = False`: Whether to report entropy in the Random algorithm or not (preferred to set to False).
   - `methods = [MAX_PROB]`: Specifies the algorithms to be used separately to extract top-k. It can be a list of the following options: MAX_PROB (which is EntrRed), NAIVE (which is Random), and EXACT_BASELINE (which is Baseline).

   You can modify these parameters based on your specific experiment setup.



   



## Results

### Cost Analysis

Cost in this context refers to the number of LLM calls made during the execution of the framework. The different methods vary in cost:

- **EntrRedDep**: This method has the lowest cost, as it minimizes LLM calls, making it the most efficient.
- **EntrRedInd**: This method has a slightly higher cost than EntrRedDep but still remains much more efficient compared to other methods, especially when compared to Random.
- **Random**: The Random method involves more LLM calls and is significantly more costly than the other methods.
- **Baseline**: This method incurs the maximum cost, as it requires all possible LLM calls to evaluate every candidate in the dataset.

### Scalability Analysis

Out of the four key components of the framework — **update bounds**, **compute PDF**, **determine next question**, and **LLM response** — three components are fully scalable as the number of candidates increases:

1. **Update bounds**
2. **Determine next question**
3. **LLM response**

However, the **compute PDF** component is only scalable when using **ProbInd** with the independence assumption. The **ProbDep** method faces challenges in scalability, especially as the number of candidates grows, which impacts the overall performance when using that method.

