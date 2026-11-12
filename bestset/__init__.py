"""Best set retrieval with uncertain and expensive predicates.

pmf          PMFs on the score grid, sum-convolution, bounds
scoring      scoring functions and their predicates
data         item pools of the Hotels, Movies and Yelp datasets
responses    predicate calls to the generative models, recorded responses
candidates   candidate sets, score bounds, pruning
probability  bounds on the winning probabilities: ProbInd and ProbDep
nextcall     selection of the next call: EntrRed, Greedy, Random
baselines    Exhaustive, LOTUS, TourRank
framework    the retrieval loop
parse_query  natural-language query -> scoring function
"""
from .framework import BestSetRetrieval, RunResult  # noqa: F401
