"""Predicate calls.

A predicate call asks a generative model for the score of a predicate on
one item (unary) or on a pair of items (binary).  A call is issued x times
and its x scores form the PMF of the call (``pmf.pmf_from_samples``).

ModelClient         one model behind a chat-completion API
ResponseStore       raw responses of the calls on one pool, kept on disk
RecordedResponder   predicate calls of a scoring function on a pool; a call
                    is sent to the model the first time it is needed and its
                    raw responses are recorded, so every algorithm run on the
                    same pool reads the same raw responses
VariabilityResponder  controls the variability of the responses (experiment
                    on the uncertainty of the responses)
APIResponder        calls with explicit prompts (model comparison)

Responders expose ``call(key) -> list[float]`` and count the calls they issue
in ``calls``.  A key is ``(predicate index, i)`` or ``(predicate index, i, j)``
with i < j.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .pmf import GRID_M

SYSTEM = ("Evaluate how well an item (or how diverse a pair of items is) matches "
          "the user's definition. Output a single float in [0,1] with one decimal.")

# chat-completion endpoints: provider -> (environment variable of the key, base url)
PROVIDERS = {
    "openai": ("OPENAI_API_KEY", None),
    "deepinfra": ("DEEPINFRA_API_KEY", "https://api.deepinfra.com/v1/openai"),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
}


def load_env(path: str | None = None):
    """Read KEY=value lines of .env (repository root by default) into the environment."""
    path = path or os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _round_grid(v: float, m: int = GRID_M) -> float:
    return round(min(max(v, 0.0), 1.0) * (m - 1)) / (m - 1)


def data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


class ModelClient:
    """One generative model.  ``score(prompt)`` issues one call and returns
    (score, wall-clock seconds, prompt tokens); a prompt is
    ``{"text": ..., "images": [url, ...]}``."""

    def __init__(self, model: str, provider: str, temperature: float = 1.0, max_retries: int = 3):
        self.model, self.provider, self.temperature = model, provider, temperature
        self.max_retries = max_retries
        self.temperature_rejected = False
        self._client = None
        self._lock = threading.Lock()

    def _api(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI     # the three providers speak the same protocol
                load_env()
                env, base = PROVIDERS[self.provider]
                self._client = OpenAI(api_key=os.environ[env], base_url=base)
        return self._client

    def score(self, prompt: dict):
        from openai import BadRequestError, RateLimitError
        content = [{"type": "text", "text": prompt["text"]}]
        for url in prompt.get("images", []):
            content.append({"type": "image_url", "image_url": {"url": url}})
        kw = {"max_tokens": 8, "temperature": self.temperature}
        reasoning = self.provider == "openai" and self.model.startswith(("gpt-5", "gpt-6", "o"))
        if reasoning:      # these models take max_completion_tokens; no reasoning, so 8 tokens suffice
            kw = {"max_completion_tokens": 8, "reasoning_effort": "none", "temperature": self.temperature}
        if self.temperature_rejected:
            kw.pop("temperature", None)
        waits, invalid = 0, 0
        while True:
            t0 = time.perf_counter()
            try:
                r = self._api().chat.completions.create(
                    model=self.model, **kw,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": content}])
            except RateLimitError:
                if waits == 7:
                    raise
                time.sleep(2 ** waits)
                waits += 1
                continue
            except BadRequestError as e:
                # some models only accept their default temperature
                if "temperature" in kw and "temperature" in str(e).lower():
                    kw.pop("temperature")
                    self.temperature_rejected = True
                    continue
                raise
            wall = time.perf_counter() - t0      # latency of the successful attempt
            text = (r.choices[0].message.content or "").strip()
            try:
                value = float(text.split()[0].strip(",;"))
            except (ValueError, IndexError):
                invalid += 1
                if invalid > self.max_retries:
                    raise ValueError(f"{self.model}: no score in the response {text!r}")
                continue
            usage = getattr(r, "usage", None)
            return _round_grid(value), wall, getattr(usage, "prompt_tokens", None)


class ResponseStore:
    """Raw responses of the predicate calls on one pool (a JSON file).

    ``data[predicate/model][key]`` is the list of the scores received so far
    for that call.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self.data = {}
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def get(self, group: str, key) -> list:
        return self.data.get(group, {}).get(_skey(key), [])

    def add(self, group: str, key, scores):
        with self._lock:
            self.data.setdefault(group, {}).setdefault(_skey(key), []).extend(scores)

    def save(self):
        if not self.path:
            return
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f)
            os.replace(tmp, self.path)


def _skey(key) -> str:
    return ",".join(str(int(v)) for v in key[1:])


class RecordedResponder:
    """Predicate calls of scoring function F on a pool.

    ``models`` maps a modality ("text", "image") to the ModelClient that
    evaluates the predicates of that modality.  The x raw responses of a
    call are read from the store; a call that is not in the store yet is
    sent to its model and recorded.
    """

    def __init__(self, F, pool, models: dict, store: ResponseStore, x: int = 3, workers: int = 16):
        self.F, self.pool, self.models, self.store, self.x = F, pool, models, store, x
        self.workers = workers
        self.calls = 0

    def _group(self, key) -> str:
        p = self.F.predicates[key[0]]
        return f"{p.name}/{self.models[p.modality].model}"

    def prompt(self, key) -> dict:
        p = self.F.predicates[key[0]]
        text = "Definition: " + p.definition.format(**self.pool.query) + ".\n\n"
        idx = key[1:]
        if p.modality == "image":
            names = [str(self.pool.items.iloc[i]["name"]) for i in idx]
            text += "\n".join(f"Business {a + 1}: {name} (image {a + 1})" if len(idx) > 1
                              else f"Business: {name} (image attached)" for a, name in enumerate(names))
            return {"text": text, "images": [data_url(self.pool.image(i)) for i in idx]}
        if len(idx) == 1:
            return {"text": text + self.pool.text(idx[0])}
        return {"text": text + "\n\n".join(f"Item {a + 1}\n{self.pool.text(i)}" for a, i in enumerate(idx))}

    def _fill(self, key):
        """Send the call to its model until x responses are recorded."""
        group = self._group(key)
        missing = self.x - len(self.store.get(group, key))
        if missing > 0:
            client = self.models[self.F.predicates[key[0]].modality]
            prompt = self.prompt(key)
            self.store.add(group, key, [client.score(prompt)[0] for _ in range(missing)])
            return True
        return False

    def prefetch(self, keys):
        """Record the given calls, sending the missing ones concurrently."""
        keys = [tuple(k) for k in keys if len(self.store.get(self._group(k), k)) < self.x]
        if keys:
            with ThreadPoolExecutor(self.workers) as ex:
                list(ex.map(self._fill, keys))
            self.store.save()

    def responses(self, key) -> list:
        """The x raw responses of a call (not counted)."""
        key = tuple(key)
        if self._fill(key):
            self.store.save()
        return self.store.get(self._group(key), key)[: self.x]

    def call(self, key) -> list:
        self.calls += self.x
        return self.responses(key)


class VariabilityResponder:
    """Responses with a controlled variability eps.

    Every response of a call is the most frequent recorded response of that
    call; with probability eps it is moved to a neighbouring score (+-0.1).
    eps = 0 gives x identical responses.  The x responses of a call depend
    only on (seed, call), so every algorithm reads the same responses.
    """

    def __init__(self, base: RecordedResponder, eps: float, seed: int = 0):
        self.base, self.eps, self.seed, self.x = base, eps, seed, base.x
        self.calls = 0
        self._memo = {}

    def prefetch(self, keys):
        self.base.prefetch(keys)

    def call(self, key) -> list:
        key = tuple(key)
        self.calls += self.x
        if key not in self._memo:
            raw = self.base.responses(key)
            mode = min(v for v, c in Counter(raw).items() if c == max(Counter(raw).values()))
            rng = np.random.default_rng([self.seed, *map(int, key)])
            out = []
            for _ in range(self.x):
                v = mode
                if rng.random() < self.eps:
                    v = mode + rng.choice([-0.1, 0.1])
                out.append(_round_grid(v))
            self._memo[key] = out
        return list(self._memo[key])


class APIResponder:
    """Calls with explicit prompts: ``prompts[key] = {"text": ..., "image_url": ...}``.
    Used by the model comparison, which records the latency of every call."""

    def __init__(self, prompts: dict, model: str, provider: str = "openai", x: int = 3,
                 temperature: float = 1.0):
        self.prompts, self.x = prompts, x
        self.client = ModelClient(model, provider, temperature)
        self.calls = 0
        self.latency = []          # (prompt tokens, wall-clock seconds) of every call

    @property
    def temperature_rejected(self):
        return self.client.temperature_rejected

    def call(self, key) -> list:
        self.calls += self.x
        p = self.prompts[key]
        prompt = {"text": p["text"], "images": [p["image_url"]] if p.get("image_url") else []}
        out = []
        for _ in range(self.x):
            score, wall, tokens = self.client.score(prompt)
            self.latency.append((tokens, wall))
            out.append(score)
        return out
