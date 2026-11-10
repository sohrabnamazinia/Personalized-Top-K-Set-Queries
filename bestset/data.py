"""Item pools of the three datasets.

Expected files (see data/README.md for the sources):

    data/hotels.csv                      Hotels
    data/wiki_movie_plots_deduped.csv    Movies
    data/businesses.csv                  Yelp businesses (scripts/prepare_yelp.py)
    data/businesses_photos/<photo_id>.jpg

A query is identified by an integer: it fixes the query fields used by the
predicate definitions (budget, plot request, target area, cuisine) and the
pool of n items the best set is retrieved from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

DATA_DIR = os.environ.get("BESTSET_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
MAX_CHARS = 1500     # longest free-text field kept in a prompt

BUDGETS = ["under $80 per night", "$80 to $150 per night", "$150 to $250 per night", "above $250 per night"]
PLOTS = [
    "a scary movie set in an isolated house",
    "a detective investigates a murder in a small town",
    "two strangers fall in love while travelling",
    "a group of friends searches for a hidden treasure",
    "a soldier returns home after a war",
    "a family comedy about a chaotic holiday",
    "a scientist's experiment goes out of control",
    "an underdog team wins a championship",
    "a heist planned by a crew of specialists",
    "a young musician struggles to become famous",
]
CUISINES = ["Italian", "Mexican", "Japanese", "Chinese", "American", "Indian", "Thai", "French",
            "Mediterranean", "Vietnamese"]


@dataclass
class Pool:
    dataset: str
    items: pd.DataFrame     # one row per item
    query: dict             # fields referred to by the predicate definitions

    @property
    def n(self):
        return len(self.items)

    def text(self, i: int) -> str:
        """Textual data of item i, as it is shown to the LLM."""
        r = self.items.iloc[i]
        return "\n".join(f"{label}: {_clip(r[col])}" for label, col in TEXT_FIELDS[self.dataset]
                         if col in r.index and pd.notna(r[col]) and str(r[col]).strip())

    def image(self, i: int) -> str:
        """Path of the image of item i (Yelp only)."""
        return os.path.join(DATA_DIR, "businesses_photos", f"{self.items.iloc[i]['photo_id']}.jpg")


TEXT_FIELDS = {
    "hotels": [("Hotel", "HotelName"), ("City", "cityName"), ("Country", "countyName"),
               ("Star rating", "HotelRating"), ("Address", "Address"), ("Attractions", "Attractions"),
               ("Description", "Description"), ("Facilities", "HotelFacilities")],
    "movies": [("Title", "Title"), ("Release year", "Release Year"), ("Origin", "Origin/Ethnicity"),
               ("Director", "Director"), ("Cast", "Cast"), ("Genre", "Genre"), ("Plot", "Plot")],
    "yelp": [("Business", "name"), ("Address", "address"), ("City", "city"), ("State", "state"),
             ("Stars", "stars"), ("Hours", "hours"), ("Review", "review")],
}


def _clip(v) -> str:
    s = " ".join(str(v).split())
    return s if len(s) <= MAX_CHARS else s[:MAX_CHARS].rsplit(" ", 1)[0] + " ..."


@lru_cache(maxsize=None)
def _hotels() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"), encoding="ISO-8859-1", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["Description"].notna()].drop_duplicates("HotelName")
    return df.reset_index(drop=True)


@lru_cache(maxsize=None)
def _movies() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "wiki_movie_plots_deduped.csv"))
    df = df[df["Plot"].notna()].drop_duplicates("Title")
    return df.reset_index(drop=True)


@lru_cache(maxsize=None)
def _yelp() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "businesses.csv"), engine="python")
    df = df[df["review"].notna() & df["photo_id"].notna()]
    photos = os.path.join(DATA_DIR, "businesses_photos")
    have = {f[:-4] for f in os.listdir(photos) if f.endswith(".jpg")}
    df = df[df["photo_id"].isin(have)].drop_duplicates("business_id")
    return df.reset_index(drop=True)


def load_pool(dataset: str, n: int = 200, query: int = 0) -> Pool:
    """Pool of n items for query number ``query``."""
    if dataset == "hotels":
        df = _hotels()
        # the hotels of one country, so that distances between hotels are comparable
        countries = df["countyName"].value_counts()
        countries = countries[countries >= n].index.tolist()
        country = countries[query % len(countries)]
        items = df[df["countyName"] == country].sample(n=n, random_state=query)
        fields = {"country": country, "budget": BUDGETS[query % len(BUDGETS)]}
    elif dataset == "movies":
        items = _movies().sample(n=n, random_state=query)
        fields = {"plot": PLOTS[query % len(PLOTS)]}
    elif dataset == "yelp":
        df = _yelp()
        areas = df["city"].value_counts().index.tolist()[:10]
        items = df.sample(n=n, random_state=query)
        fields = {"area": areas[query % len(areas)], "cuisine": CUISINES[query % len(CUISINES)]}
    else:
        raise ValueError(dataset)
    return Pool(dataset, items.reset_index(drop=True), fields)
