"""X7: variability, latency and cost of the same predicate call on four models.

The ambience predicate of F5++ is issued x times on n Yelp businesses to
every model: the LLMs read the review, the VLMs read the image (and, as an
additional input, the image with the review).  The score, the wall-clock
latency and the prompt size of every call are written to results/X7_raw.csv.
``--summary`` prints, per model, the fraction of calls whose x responses
differ, the latency and the cost per 1,000 calls, and across models the
fraction of calls on which the most frequent scores of two models differ.

    python -m experiments.x7_variability            # data/businesses.csv, data/businesses_photos
    python -m experiments.x7_variability --summary
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bestset.data import DATA_DIR                             # noqa: E402
from bestset.responses import APIResponder, data_url          # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
SAMPLE = os.path.join(ROOT, "data", "yelp_sample")
RAW = os.path.join(RESULTS, "X7_raw.csv")

MODELS = [   # (short name, model id, provider, kind)
    ("GPT-6 Luna", "gpt-6-luna", "openai", "llm"),
    ("Llama-4-Scout", "meta-llama/Llama-4-Scout-17B-16E-Instruct", "deepinfra", "llm"),
    ("Qwen3-VL-235B", "Qwen/Qwen3-VL-235B-A22B-Instruct", "deepinfra", "vlm"),
    ("Mistral Large", "mistral-large-latest", "mistral", "vlm"),
]
PRICES = {   # $ per 1M tokens (input, output)
    "gpt-6-luna": (0.10, 0.50),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.11, 0.34),
    "Qwen/Qwen3-VL-235B-A22B-Instruct": (0.20, 0.88),
    "mistral-large-latest": (0.50, 1.50),
}
PREDICATE = (
    "Predicate: the business has a cozy, inviting ambience.\n"
    "Use this scale: 0.0 = clearly not cozy (cold, sterile, loud or hostile); "
    "0.3 = mostly not cozy; 0.5 = neutral, mixed, or no evidence about the ambience; "
    "0.7 = somewhat cozy; 1.0 = clearly cozy and inviting (warm, comfortable, welcoming).\n"
    "Score how well the business below satisfies the predicate; output a single float "
    "in [0,1] with one decimal.\n\n")


def select(yelp_csv: str, photos: str, n: int, seed: int = 7) -> list[dict]:
    rows = list(csv.DictReader(open(yelp_csv, encoding="utf-8")))
    ok = [r for r in rows if r["review"] and 200 <= len(r["review"]) <= 900
          and r["photo_label"] in ("inside", "outside", "food")
          and os.path.exists(os.path.join(photos, r["photo_id"] + ".jpg"))]
    random.seed(seed)
    pick = random.sample(ok, n)
    os.makedirs(SAMPLE, exist_ok=True)
    items = []
    for r in pick:
        img = os.path.join(SAMPLE, r["photo_id"] + ".jpg")
        shutil.copy(os.path.join(photos, r["photo_id"] + ".jpg"), img)
        items.append({"id": r["business_id"], "name": r["name"], "stars": r["stars"],
                      "review": r["review"], "image": os.path.basename(img)})
    json.dump(items, open(os.path.join(SAMPLE, "items.json"), "w"), indent=1)
    return items


def run(items: list[dict], x: int, temperature: float = 1.0, raw: str = RAW):
    PRED = PREDICATE
    os.makedirs(os.path.dirname(os.path.abspath(raw)), exist_ok=True)
    new = not os.path.exists(raw)
    f = open(raw, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "kind", "input", "business", "rep", "score", "wall_s", "prompt_tokens"])
    for short, mid, prov, kind in MODELS:
        inputs = ["text"] if kind == "llm" else ["image", "image+text"]
        for inp in inputs:
            prompts = {}
            for i, it in enumerate(items):
                body = f"Business: {it['name']}\nReview: {it['review']}"
                img = os.path.join(SAMPLE, it["image"])
                if inp == "text":
                    prompts[i] = {"text": PRED + body}
                elif inp == "image":
                    prompts[i] = {"text": PRED + f"Business: {it['name']} (image attached)",
                                  "image_url": data_url(img)}
                else:
                    prompts[i] = {"text": PRED + body + "\n(image attached)",
                                  "image_url": data_url(img)}
            r = APIResponder(prompts, mid, prov, x=1, temperature=temperature)
            for i in prompts:
                for rep in range(x):
                    score = r.call(i)[0]
                    tokens, wall = r.latency[-1]
                    w.writerow([short, kind, inp, i, rep, score, round(wall, 3), tokens or ""])
                    f.flush()
                print(short, inp, i, flush=True)
            if r.temperature_rejected:
                print(f"{short}: the API accepts only its default temperature", flush=True)
    f.close()


def summary(raw: str = RAW):
    rows = list(csv.DictReader(open(raw)))
    for r in rows:
        r["score"] = float(r["score"]); r["wall_s"] = float(r["wall_s"])
        r["prompt_tokens"] = float(r["prompt_tokens"] or 0)
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["input"]), {}).setdefault(r["business"], []).append(r)
    print("Within-model variability (x responses to the same call)")
    print(f"{'model':16}{'input':12}{'calls':>6}{'%disagree':>10}{'range':>7}{'std':>7}{'lat s':>7}{'tok':>6}{'$/1000':>8}")
    means, modes = {}, {}
    for (m, inp), per in groups.items():
        mid = [x for x in MODELS if x[0] == m][0][1]
        pin, pout = PRICES[mid]
        dis, rng_, std, lat, tok = [], [], [], [], []
        for b, rs in per.items():
            s = np.array([r["score"] for r in rs])
            dis.append(len(set(s)) > 1); rng_.append(s.max() - s.min()); std.append(s.std())
            lat += [r["wall_s"] for r in rs]; tok += [r["prompt_tokens"] for r in rs]
            means[(m, inp, b)] = s.mean()
            modes[(m, inp, b)] = Counter(s.tolist()).most_common(1)[0][0]
        cost = (np.mean(tok) * pin + 8 * pout) / 1e6 * 1000
        print(f"{m:16}{inp:12}{len(per):>6}{100*np.mean(dis):>9.0f}%{np.mean(rng_):>7.2f}{np.mean(std):>7.3f}"
              f"{np.mean(lat):>7.2f}{np.mean(tok):>6.0f}{cost:>8.3f}")
    print("\nAcross models (default input: LLM text, VLM image)")
    keys = [(m, "text" if k == "llm" else "image") for m, _, _, k in MODELS]
    businesses = sorted({b for (_, _, b) in means})
    spread = [max(means[(m, i, b)] for m, i in keys) - min(means[(m, i, b)] for m, i in keys) for b in businesses]
    print(f"spread of model means per call: mean {np.mean(spread):.2f}, max {np.max(spread):.2f}")
    for a in range(len(keys)):
        for c in range(a + 1, len(keys)):
            ma, mc = keys[a], keys[c]
            d = [abs(means[ma + (b,)] - means[mc + (b,)]) for b in businesses]
            dm = [modes[ma + (b,)] != modes[mc + (b,)] for b in businesses]
            print(f"{ma[0]:16} vs {mc[0]:16} mean|diff| {np.mean(d):.2f}  modes differ {100*np.mean(dm):.0f}%")
    for m in ("Qwen3-VL-235B", "Mistral Large"):
        d = [abs(means[(m, "image", b)] - means[(m, "image+text", b)]) for b in businesses]
        print(f"{m}: image vs image+text mean|diff| {np.mean(d):.2f}")
    llm = [np.mean([means[(m, 'text', b)] for m in ('GPT-6 Luna', 'Llama-4-Scout')]) for b in businesses]
    vlm = [np.mean([means[(m, 'image', b)] for m in ('Qwen3-VL-235B', 'Mistral Large')]) for b in businesses]
    print(f"LLM(text) vs VLM(image) mean|diff| {np.mean(np.abs(np.array(llm)-np.array(vlm))):.2f}, "
          f"corr {np.corrcoef(llm, vlm)[0,1]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yelp", default=os.path.join(DATA_DIR, "businesses.csv"))
    ap.add_argument("--photos", default=os.path.join(DATA_DIR, "businesses_photos"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--x", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--summary", action="store_true", help="only print the summary of results/X7_raw.csv")
    args = ap.parse_args()
    if not args.summary:
        run(select(args.yelp, args.photos, args.n), args.x, args.temperature)
    summary()


if __name__ == "__main__":
    main()
