"""Scoring functions.

A scoring function is a list of predicates, each unary (one item) or binary
(a pair of items):

    F(s, u) = sum_{p unary} sum_{e in s} p(e, u) + sum_{p binary} sum_{e_i, e_j in s} p(e_i, e_j)

A predicate is given by its definition in natural language and by the
modality of the data it reads: text predicates are evaluated by an LLM,
image predicates by a VLM.  A definition may refer to fields of the query
(e.g. ``{area}``), which are filled in when the prompt is built.

F1..F6 have one relevance and one diversity predicate; F1+, F1++, F5+ and
F5++ add a third and a fourth predicate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Predicate:
    name: str
    arity: int            # 1: unary, 2: binary
    definition: str
    modality: str         # "text" (LLM) or "image" (VLM)


@dataclass(frozen=True)
class ScoringFunction:
    name: str
    predicates: tuple

    @property
    def ell(self):
        return len(self.predicates)


P = {
    # Hotels
    "h_rating": Predicate("hotel rating", 1, "the hotel is highly rated", "text"),
    "h_geo_div": Predicate("geo distance", 2, "the hotels are far apart", "text"),
    "h_center": Predicate("proximity to city center", 1, "the hotel is close to the city center", "text"),
    "h_star_div": Predicate("star rating difference", 2, "the hotels differ in star rating", "text"),
    "h_afford": Predicate("affordability", 1, "the hotel is affordable for a student traveler", "text"),
    # Movies
    "m_plot": Predicate("brief-plot similarity", 1, "the plot of the movie matches the request: {plot}", "text"),
    "m_year_div": Predicate("production year", 2, "the movies were produced in different years", "text"),
    "m_popularity": Predicate("popularity", 1, "the movie is popular", "text"),
    "m_genre_era_div": Predicate("genre and era coverage", 2,
                                 "the movies cover different genres and different periods", "text"),
    # Yelp businesses
    "y_location": Predicate("location near the target area", 1,
                            "the business is located in or near {area}", "text"),
    "y_cost_div": Predicate("cost variation", 2,
                            "the businesses differ in price level, as far as the images show", "image"),
    "y_cuisine": Predicate("cuisine match", 1, "the business serves {cuisine} food", "text"),
    "y_hours_div": Predicate("operating hours", 2, "the businesses have different operating hours", "text"),
    "y_ambience_div": Predicate("ambience diversity", 2,
                                "the businesses differ in ambience, as far as the images show", "image"),
}

SCORING = {
    "F1": ScoringFunction("F1", (P["h_rating"], P["h_geo_div"])),
    "F1+": ScoringFunction("F1+", (P["h_rating"], P["h_geo_div"], P["h_afford"])),
    "F1++": ScoringFunction("F1++", (P["h_rating"], P["h_geo_div"], P["h_afford"], P["h_star_div"])),
    "F2": ScoringFunction("F2", (P["h_center"], P["h_star_div"])),
    "F3": ScoringFunction("F3", (P["m_plot"], P["m_year_div"])),
    "F4": ScoringFunction("F4", (P["m_popularity"], P["m_genre_era_div"])),
    "F5": ScoringFunction("F5", (P["y_location"], P["y_cost_div"])),
    "F5+": ScoringFunction("F5+", (P["y_location"], P["y_cost_div"], P["y_cuisine"])),
    "F5++": ScoringFunction("F5++", (P["y_location"], P["y_cost_div"], P["y_cuisine"], P["y_ambience_div"])),
    "F6": ScoringFunction("F6", (P["y_cuisine"], P["y_hours_div"])),
}

DATASET_OF = {"F1": "hotels", "F1+": "hotels", "F1++": "hotels", "F2": "hotels",
              "F3": "movies", "F4": "movies",
              "F5": "yelp", "F5+": "yelp", "F5++": "yelp", "F6": "yelp"}
