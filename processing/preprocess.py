"""Data loading, preprocessing, and TMDB API helpers for the movie recommender."""

import ast
import os
import pickle
import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import requests
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

API_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"
ERROR_IMAGE = (
    "https://media.istockphoto.com/vectors/error-icon-vector-illustration-vector-id922024224?k=6&m"
    '=922024224&s=612x612&w=0&h=LXl8Ul7bria6auAXKIjlvb6hRHkAodTqyqBeA6K7R54='
)

ps = PorterStemmer()
_session = requests.Session()
_STOPWORDS: Optional[set] = None


@dataclass
class MovieDetails:
    poster: str
    backdrop: str
    budget: int
    genres: List[str]
    overview: str
    release_date: str
    revenue: int
    runtime: int
    languages: List[str]
    rating: float
    vote_count: int
    movie_id: int
    cast: List[str]
    director: List[str]
    cast_ids: List[int]


def _load_stopwords() -> set:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        _STOPWORDS = set(stopwords.words('english'))
    return _STOPWORDS


@lru_cache(maxsize=1)
def _api_key() -> Optional[str]:
    """Get the TMDB API key from Streamlit secrets or the environment."""
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.environ.get("TMDB_API_KEY")


def _tmdb_get(endpoint: str, **params: Any) -> Optional[Dict[str, Any]]:
    api_key = _api_key()
    if not api_key:
        return None
    response = _session.get(
        f"{API_BASE}/{endpoint}", params={"api_key": api_key, **params}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def _load_df(path: str) -> pd.DataFrame:
    with open(path, 'rb') as pickle_file:
        return pd.DataFrame.from_dict(pickle.load(pickle_file))


def _names(obj: str) -> List[str]:
    return [item['name'] for item in ast.literal_eval(obj)]


def _top_cast_names(obj: str) -> List[str]:
    return [item['name'] for item in ast.literal_eval(obj)[:10]]


def _director_names(obj: str) -> List[str]:
    return [item['name'] for item in ast.literal_eval(obj) if item['job'] == 'Director'][:1]


def read_csv_to_df() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs and return (movies, new_df, movies2)."""
    credits = pd.read_csv('Files/tmdb_5000_credits.csv')
    movies = pd.read_csv('Files/tmdb_5000_movies.csv').merge(credits, on='title')

    movies2 = movies[
        ['movie_id', 'title', 'budget', 'overview', 'popularity', 'release_date', 'revenue',
         'runtime', 'spoken_languages', 'status', 'vote_average', 'vote_count']
    ].dropna().copy()

    movies = movies[
        ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew',
         'production_companies', 'release_date']
    ].dropna().copy()

    movies['genres'] = movies['genres'].apply(_names)
    movies['keywords'] = movies['keywords'].apply(_names)
    movies['top_cast'] = movies['cast'].apply(_top_cast_names)
    movies['director'] = movies['crew'].apply(_director_names)
    movies['production_comp'] = movies['production_companies'].apply(_names)

    movies['overview'] = movies['overview'].apply(str.split)
    for col in ('genres', 'keywords', 'top_cast', 'director', 'production_comp'):
        movies[col] = movies[col].apply(lambda items: [item.replace(' ', '') for item in items])

    new_df = movies[
        ['movie_id', 'title', 'overview', 'genres', 'keywords', 'top_cast', 'director',
         'production_comp']
    ].copy()
    new_df.columns = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew',
                      'production_comp']

    new_df['tags'] = (
        new_df['overview'] + new_df['genres'] + new_df['keywords']
        + new_df['cast'] + new_df['crew']
    ).apply(stemming_stopwords)
    new_df['keywords'] = new_df['keywords'].apply(stemming_stopwords)

    for col in ('genres', 'cast', 'production_comp'):
        new_df[col] = new_df[col].apply(lambda items: ' '.join(items).lower())

    return movies, new_df, movies2


def stemming_stopwords(words: List[str]) -> str:
    """Stem a list of words, drop stopwords/short tokens, and join into a string."""
    stop_words = _load_stopwords()
    kept = []
    for word in words:
        stem = ps.stem(word).lower()
        if stem not in stop_words and len(stem) > 2 and stem not in string.punctuation:
            kept.append(stem)
    return ' '.join(kept)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie_images(movie_id: int) -> Tuple[str, str]:
    """Fetch (poster_url, backdrop_url); placeholders filled with the error image."""
    try:
        data = _tmdb_get(f"movie/{movie_id}")
        if data:
            poster = data['poster_path']
            backdrop = data['backdrop_path']
            return (
                f"{IMAGE_BASE}/w780{poster}" if poster else ERROR_IMAGE,
                f"{IMAGE_BASE}/w1280{backdrop}" if backdrop else '',
            )
    except requests.RequestException:
        pass
    return ERROR_IMAGE, ''


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_posters(movie_id: int) -> str:
    poster, _ = fetch_movie_images(movie_id)
    return poster


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_person_details(person_id: int) -> Tuple[str, str]:
    try:
        data = _tmdb_get(f"person/{person_id}")
        if data:
            url = (
                f"{IMAGE_BASE}/w220_and_h330_face{data['profile_path']}"
                if data.get('profile_path') else ERROR_IMAGE
            )
            return url, data.get('biography') or ''
    except requests.RequestException:
        pass
    return ERROR_IMAGE, ''


def recommend(
    new_df: pd.DataFrame, movie: str, pickle_file_path: str, top_n: int = 25
) -> List[Tuple[str, int]]:
    """Return (title, movie_id) tuples most similar to the given movie."""
    with open(pickle_file_path, 'rb') as pickle_file:
        similarity = pickle.load(pickle_file)

    movie_idx = new_df[new_df['title'] == movie].index[0]

    if hasattr(similarity, 'todense'):
        row = np.asarray(similarity[movie_idx].todense()).ravel()
    else:
        row = similarity[movie_idx]

    scores = sorted(enumerate(row), reverse=True, key=lambda x: x[1])[1:top_n + 1]
    return [(new_df.iloc[i]['title'], new_df.iloc[i]['movie_id']) for i, _ in scores]


@st.cache_data(ttl=86400, show_spinner=False)
def get_details(selected_movie_name: str) -> MovieDetails:
    movies = _load_df('Files/movies_dict.pkl')
    movies2 = _load_df('Files/movies2_dict.pkl')

    a = movies2[movies2['title'] == selected_movie_name].iloc[0]
    b = movies[movies['title'] == selected_movie_name].iloc[0]

    languages = [lang['name'] for lang in ast.literal_eval(a['spoken_languages'])]
    cast_ids = [person['id'] for person in ast.literal_eval(b['cast'])]
    poster, backdrop = fetch_movie_images(a['movie_id'])

    return MovieDetails(
        poster=poster,
        backdrop=backdrop,
        budget=a['budget'],
        genres=b['genres'],
        overview=a['overview'],
        release_date=a['release_date'],
        revenue=a['revenue'],
        runtime=a['runtime'],
        languages=languages,
        rating=a['vote_average'],
        vote_count=a['vote_count'],
        movie_id=a['movie_id'],
        cast=b['top_cast'],
        director=b['director'],
        cast_ids=cast_ids,
    )