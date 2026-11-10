"""From a natural-language query to a scoring function.

One model call turns a short query (e.g. "five affordable hotels in
different parts of the city") into k and the predicates of the scoring
function, each with its arity, its definition and the modality it reads.
This step is not part of the evaluation, which takes the scoring function
as input.
"""
from __future__ import annotations

import json
import os

from .responses import PROVIDERS, load_env
from .scoring import Predicate, ScoringFunction

INSTRUCTIONS = (
    "Turn the user's query into a scoring function for retrieving a set of items. "
    "Answer with JSON only: {\"k\": <number of items>, \"predicates\": [{\"name\": <short name>, "
    "\"arity\": 1 for a property of one item or 2 for a property of a pair of items, "
    "\"definition\": <one sentence that an item or a pair of items can satisfy>, "
    "\"modality\": \"text\" or \"image\"}]}.")


def parse_query(query: str, model: str = "gpt-6-luna", provider: str = "openai"):
    """Return (ScoringFunction, k) for a natural-language query."""
    from openai import OpenAI
    load_env()
    env, base = PROVIDERS[provider]
    client = OpenAI(api_key=os.environ[env], base_url=base)
    r = client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": query}])
    spec = json.loads(r.choices[0].message.content)
    predicates = tuple(Predicate(p["name"], int(p["arity"]), p["definition"].rstrip("."), p["modality"])
                       for p in spec["predicates"])
    return ScoringFunction(query, predicates), int(spec["k"])
