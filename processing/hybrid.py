"""Hybrid recommendation engine blending content-based and semantic scores.

Combines the existing Bag-of-Words cosine similarity (content-based) with
the TF-IDF semantic similarity into a single weighted score:

    hybrid_score = w × content_score + (1 − w) × semantic_score

where *w* is the ``content_weight`` parameter (default 0.5).
"""

import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd

from processing.embeddings import TFIDF_PICKLE

CONTENT_PICKLE = "Files/similarity_tags_tags.pkl"


def _load_similarity(path: str):
    """Load a similarity matrix from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def hybrid_recommend(
    new_df: pd.DataFrame,
    movie: str,
    content_weight: float = 0.5,
    top_n: int = 25,
) -> List[Tuple[str, int, float]]:
    """Return top-N movies using a weighted blend of BoW + TF-IDF similarity.

    Parameters
    ----------
    new_df : pd.DataFrame
        Processed movie DataFrame with a ``title`` column.
    movie : str
        Title of the seed movie.
    content_weight : float
        Weight for content-based (BoW) similarity.  The remaining
        ``1 - content_weight`` goes to semantic (TF-IDF) similarity.
    top_n : int
        Number of recommendations to return.

    Returns
    -------
    list of (title, movie_id, hybrid_score)
        Sorted by descending hybrid score.
    """
    content_sim = _load_similarity(CONTENT_PICKLE)
    semantic_sim = _load_similarity(TFIDF_PICKLE)

    movie_idx = new_df[new_df["title"] == movie].index[0]

    # Extract rows as dense 1-D arrays
    def _row(mat, idx):
        if hasattr(mat, "todense"):
            return np.asarray(mat[idx].todense()).ravel()
        return mat[idx]

    content_row = _row(content_sim, movie_idx)
    semantic_row = _row(semantic_sim, movie_idx)

    # Normalise both rows to [0, 1] so the weight is meaningful
    def _norm(arr):
        mx = arr.max()
        return arr / mx if mx > 0 else arr

    content_norm = _norm(content_row.astype(float))
    semantic_norm = _norm(semantic_row.astype(float))

    hybrid = content_weight * content_norm + (1 - content_weight) * semantic_norm

    scores = sorted(enumerate(hybrid), reverse=True, key=lambda x: x[1])[1: top_n + 1]
    return [
        (new_df.iloc[i]["title"], int(new_df.iloc[i]["movie_id"]), float(s))
        for i, s in scores
    ]
