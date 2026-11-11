"""One job of every kind through experiments.run, with the stand-in model of test_pipeline."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from bestset import data                                   # noqa: E402
from bestset.responses import ResponseStore                # noqa: E402
from experiments import run                                # noqa: E402
from experiments.configs import DEFAULT                    # noqa: E402
from test_pipeline import FakeClient, _movies              # noqa: E402


def test_jobs_share_one_ground_truth():
    with tempfile.TemporaryDirectory() as tmp:
        data.DATA_DIR = tmp
        data._movies.cache_clear()
        _movies(tmp)
        pool = data.load_pool("movies", n=40, query=1)
        store = ResponseStore(os.path.join(tmp, "responses.json"))
        models = {"text": FakeClient(), "image": FakeClient()}
        refs = set()
        for algo in ["entrred+probdep", "entrred+probind", "greedy", "random", "lotus", "tourrank"]:
            job = dict(DEFAULT, exp="T", F="F3", k=3, algo=algo, seed=1, n=40, top_items=8)
            row = run.run_job(job, pool, store, models)
            assert row["calls"] > 0 and row["recall"] in (0, 1)
            refs.add(row["ref_set"])
        assert len(refs) == 1
