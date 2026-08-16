"""Semantic embeddings using TF-IDF vectorization for enhanced movie similarity."""

import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TFIDF_PICKLE = "Files/similarity_tfidf.pkl"


def compute_tfidf_similarity(new_df: pd.DataFrame) -> None:
    """Build a TF-IDF cosine-similarity matrix from the 'tags' column and cache it.

    TF-IDF down-weights common words (e.g. "movie", "story") and up-weights
    distinctive terms, producing a stronger semantic signal than raw
    CountVectorizer.

    The matrix is saved as a sparse pickle in ``Files/similarity_tfidf.pkl``
    and is only regenerated when the file is missing.
    """
    if os.path.exists(TFIDF_PICKLE):
        return

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(new_df["tags"])
    sim = cosine_similarity(tfidf_matrix, dense_output=False)

    with open(TFIDF_PICKLE, "wb") as f:
        pickle.dump(sim, f)


def semantic_recommend(
    new_df: pd.DataFrame, movie: str, top_n: int = 25
) -> List[Tuple[str, int, float]]:
    """Return the *top_n* most semantically similar movies using TF-IDF.

    Returns a list of ``(title, movie_id, similarity_score)`` tuples sorted
    by descending similarity.
    """
    with open(TFIDF_PICKLE, "rb") as f:
        similarity = pickle.load(f)

    movie_idx = new_df[new_df["title"] == movie].index[0]

    if hasattr(similarity, "todense"):
        row = np.asarray(similarity[movie_idx].todense()).ravel()
    else:
        row = similarity[movie_idx]

    scores = sorted(enumerate(row), reverse=True, key=lambda x: x[1])[1: top_n + 1]
    return [
        (new_df.iloc[i]["title"], new_df.iloc[i]["movie_id"], float(score))
        for i, score in scores
    ]
