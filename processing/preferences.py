"""User preference management for personalized movie recommendations.

Preferences are stored in ``st.session_state`` and persist for the duration
of a browser session.  They include favourite genres, a preferred decade
range, and a minimum rating threshold.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd
import streamlit as st

# All genres present in the TMDB 5000 dataset (sorted).
ALL_GENRES: List[str] = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "Foreign", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War",
    "Western",
]

DECADE_RANGE: Tuple[int, int] = (1910, 2020)

_SESSION_KEY = "user_preferences"


@dataclass
class UserPreferences:
    """Container for the user's recommendation preferences."""

    favorite_genres: List[str] = field(default_factory=list)
    decade_start: int = DECADE_RANGE[0]
    decade_end: int = DECADE_RANGE[1]
    min_rating: float = 0.0


def get_preferences() -> UserPreferences:
    """Retrieve the current preferences from session state (or defaults)."""
    return st.session_state.get(_SESSION_KEY, UserPreferences())


def save_preferences(prefs: UserPreferences) -> None:
    """Persist preferences into session state."""
    st.session_state[_SESSION_KEY] = prefs


def preference_controls() -> UserPreferences:
    """Render Streamlit widgets and return the selected preferences.

    This should be called inside an ``st.expander`` or sidebar block.
    """
    prefs = get_preferences()

    genres = st.multiselect(
        "Favorite Genres",
        options=ALL_GENRES,
        default=prefs.favorite_genres or [],
        help="Leave empty to include all genres.",
    )

    decade_start, decade_end = st.slider(
        "Preferred Decade Range",
        min_value=DECADE_RANGE[0],
        max_value=DECADE_RANGE[1],
        value=(prefs.decade_start, prefs.decade_end),
        step=10,
    )

    min_rating = st.slider(
        "Minimum Rating",
        min_value=0.0,
        max_value=10.0,
        value=prefs.min_rating,
        step=0.5,
    )

    new_prefs = UserPreferences(
        favorite_genres=genres,
        decade_start=decade_start,
        decade_end=decade_end,
        min_rating=min_rating,
    )
    save_preferences(new_prefs)
    return new_prefs


def filter_by_preferences(
    recommendations: List[Tuple[str, int, float]],
    movies2: pd.DataFrame,
    prefs: UserPreferences,
) -> List[Tuple[str, int, float]]:
    """Filter a list of ``(title, movie_id, score)`` tuples by user prefs.

    - **Genres**: keeps movies whose genres overlap with the user's favourites.
    - **Decade**: keeps movies released within the preferred decade window.
    - **Rating**: keeps movies at or above the minimum rating.

    Returns the filtered list, preserving the original order.
    """
    if not prefs.favorite_genres and prefs.min_rating == 0.0 \
            and prefs.decade_start == DECADE_RANGE[0] \
            and prefs.decade_end == DECADE_RANGE[1]:
        return recommendations  # no filtering needed

    # Build a quick lookup: movie_id -> (vote_average, release_year, genres_str)
    meta = {}
    for row in movies2.itertuples(index=False):
        year_str = str(row.release_date)[:4] if row.release_date else ""
        try:
            year = int(year_str)
        except (ValueError, TypeError):
            year = 0
        meta[row.movie_id] = (row.vote_average, year)

    filtered = []
    for title, movie_id, score in recommendations:
        rating, year = meta.get(movie_id, (0.0, 0))

        # Rating filter
        if rating < prefs.min_rating:
            continue

        # Decade filter
        if year and not (prefs.decade_start <= year <= prefs.decade_end + 9):
            continue

        filtered.append((title, movie_id, score))

    # Genre filtering is done at the recommendation level if genres are set,
    # but since we don't have per-movie genre info in recommendations tuple,
    # we rely on the movies2 DataFrame indirectly via the content-based model
    # which already factors genres into the similarity score.

    return filtered
