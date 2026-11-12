"""The retrieval pipeline end to end, with a stand-in for the model.

No model is called: ``FakeClient`` returns a score that is a fixed function
of the prompt, with an occasional deviation, which is enough to exercise
the prompts, the recorded responses, the framework and the baselines.
"""
import hashlib
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bestset import data                                               # noqa: E402
from bestset.baselines import BASELINES, Reference                     # noqa: E402
from bestset.framework import BestSetRetrieval                         # noqa: E402
from bestset.responses import RecordedResponder, ResponseStore, VariabilityResponder  # noqa: E402
from bestset.scoring import SCORING                                    # noqa: E402


class FakeClient:
    model = "fake"

    def __init__(self):
        self.sent = 0

    def score(self, prompt):
        self.sent += 1
        h = int(hashlib.sha1(prompt["text"].encode()).hexdigest(), 16)
        value = (h % 11) / 10
        if self.sent % 9 == 0:
            value = min(1.0, value + 0.1)
        return value, 0.0, len(prompt["text"]) // 4


def _movies(folder, n=60):
    rows = [{"Release Year": 1950 + i, "Title": f"Movie {i}", "Origin/Ethnicity": "American",
             "Director": f"Director {i % 7}", "Cast": "", "Genre": ["drama", "comedy", "horror"][i % 3],
             "Wiki Page": "", "Plot": f"Plot of movie {i}. " * 5} for i in range(n)]
    pd.DataFrame(rows).to_csv(os.path.join(folder, "wiki_movie_plots_deduped.csv"), index=False)


def test_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        data.DATA_DIR = tmp
        data._movies.cache_clear()
        _movies(tmp)
        pool = data.load_pool("movies", n=40, query=1)
        F, k = SCORING["F3"], 3
        client = FakeClient()
        models = {"text": client, "image": client}
        path = os.path.join(tmp, "responses.json")

        def responder():
            return RecordedResponder(F, pool, models, ResponseStore(path), x=3, workers=4)

        prompt = responder().prompt((1, 0, 1))
        assert "Item 1" in prompt["text"] and "Item 2" in prompt["text"] and "Predicate:" in prompt["text"]

        for prob in ("ind", "dep"):
            run = BestSetRetrieval(F, pool, responder(), k, theta=0.999, prob=prob, top_items=8, seed=0)
            r = run.run()
            ref = Reference(F, pool, responder(), k, run.space.items)
            assert len(r.best_set) in (0, k) and ref.agrees(r.best_set, 0.999)
            assert r.calls < 3 * (40 + 40 * 39 // 2)

        for name in ("lotus", "tourrank"):
            best, calls, items = BASELINES[name](F, pool, responder(), k, 8, seed=0)
            assert len(best) == k
            assert 0.0 <= Reference(F, pool, responder(), k, items).winning_probability(best) <= 1.0

        # a second run reads the recorded responses and sends nothing
        sent = client.sent
        BestSetRetrieval(F, pool, responder(), k, theta=0.999, prob="dep", top_items=8, seed=0).run()
        assert client.sent == sent

        # eps = 0: the x responses of a call are identical
        v = VariabilityResponder(responder(), 0.0, seed=0)
        assert len(set(v.call((0, 0)))) == 1 and v.calls == 3


if __name__ == "__main__":
    test_pipeline()
    print("ok")
