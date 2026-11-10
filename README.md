# Best Set Retrieval with Uncertain and Expensive Predicates

A query asks for the best set of `k` items under a scoring function

    F(s,u) = Σ_p Σ_{e ∈ s} p(e,u) + Σ_p Σ_{e_i,e_j ∈ s} p(e_i,e_j)

whose unary and binary predicates are evaluated by generative models (LLMs
for text, VLMs for images). A predicate call is issued `x` times and its
responses form a PMF. The framework issues as few calls as possible until
one candidate set is the best set with probability at least θ.

## Layout

```
bestset/
  pmf.py            PMFs on the score grid {0, 0.1, ..., 1}, sum-convolution, bounds
  scoring.py        scoring functions F1..F6, F1+, F1++, F5+, F5++ and their predicates
  data.py           item pools of the Hotels, Movies and Yelp datasets
  responses.py      predicate calls to the models (OpenAI, DeepInfra, Mistral APIs);
                    the raw responses of every call are recorded and shared by all algorithms
  candidates.py     candidate sets, score bounds, bound-based and threshold pruning
  probability.py    winning probabilities: ProbInd and ProbDep
  nextcall.py       next call: EntrRed, Greedy, Random
  baselines.py      Exhaustive, LOTUS, TourRank
  framework.py      the retrieval loop
  parse_query.py    natural-language query -> scoring function
experiments/
  configs.py        models and settings of X1, X3, X4, X5, X8
  run.py            python -m experiments.run --exp X1
  collect.py        python -m experiments.collect --F F1 --query 0   (all calls of a pool)
  x6_scalability.py running time of each component
  x7_variability.py the same call on the four models: variability, latency, cost
  plots.py          figures and LaTeX table rows
  numbers.py        the numbers reported in the text
scripts/
  prepare_yelp.py   builds the Yelp files from the Yelp Open Dataset
  run_all.sh        every experiment in sequence
tests/              checks that need no model call
data/README.md      sources of the three datasets
```

## Setup

```
pip install -r requirements.txt
```

Place the datasets in `data/` (see `data/README.md`) and the API keys in a
`.env` file in the repository root:

```
OPENAI_API_KEY=...
DEEPINFRA_API_KEY=...
MISTRAL_API_KEY=...
```

## Running

```
scripts/run_all.sh                    # all experiments, figures and tables
python -m experiments.run --exp X1    # one experiment
python -m experiments.plots
python -m experiments.numbers
python -m pytest tests                # no model call
```

`--llm` and `--vlm` choose the models (`gpt-6-luna` or `llama-4-scout`,
`qwen3-vl` or `mistral-large`); `--n` sets the pool size.

The first time a predicate call is needed on a pool, its `x` responses are
requested from the model and recorded in `responses/`. Every algorithm and
every experiment on that pool reads the recorded responses, so all methods
are compared on the same raw responses, and a re-run sends no call twice.
`experiments.collect` records every call of a pool in advance (Exhaustive).

## Experiments

| | experiment | setting |
|---|---|---|
| X1 | predicate calls vs. k | F1, F3, F5; k = 3, 5, 7; all algorithms; 10 queries |
| X3 | number of predicates | F1, F1+, F1++, F5, F5+, F5++; k = 5; 5 queries |
| X4 | variability of the responses | ε = 0 to 0.2; F1, F5; k = 5; 5 queries |
| X5 | threshold θ | θ = 0.7 to 0.999; F1, F3; k = 7; 5 queries |
| X6 | running time of the components | M = 100 to 100,000 candidate sets |
| X7 | model comparison | one predicate, 50 Yelp businesses, x = 5, four models |
| X8 | pruning | bound-based vs. threshold pruning (τ); k = 5; 5 queries |

Defaults: pools of 200 items, x = 3, θ = 0.999. Recall is 1 when the returned
set is the one Exhaustive returns on the same raw responses.

Columns of `results/X*.csv`: `exp, dataset, F, ell, k, x, eps, theta, tau,
layout, M, n_items, algo, seed, calls, recall, time_s, iterations,
final_prob, best_set, ref_set, gap`.
