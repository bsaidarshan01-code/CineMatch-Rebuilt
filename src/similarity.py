"""
similarity.py
=============
Pure-maths similarity helpers used by the recommendation engine.

Contains:
  * jaccard_similarity      – set-based Jaccard formula
  * jaccard_recommend       – genre-based recommendation ranking
  * cosine_similarity_pair  – manual cosine formula (educational)

The TF-IDF / cosine matrix computation is done inside recommender.py using
scikit-learn for efficiency over the full dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    Formula:
        J(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 when both sets are empty (avoids division by zero).

    Parameters
    ----------
    set_a, set_b : set
        Any two Python sets (e.g. genre sets).

    Returns
    -------
    float
        Similarity score in [0.0, 1.0].

    Examples
    --------
    >>> jaccard_similarity({"action", "thriller"}, {"action", "drama"})
    0.3333333333333333
    >>> jaccard_similarity(set(), set())
    0.0
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def jaccard_recommend(
    movie_id: int,
    movies_df: pd.DataFrame,
    top_n: int = 10,
) -> list:
    """Recommend movies using Jaccard similarity on genre sets.

    Parameters
    ----------
    movie_id : int
        The selected movie's ID.
    movies_df : pd.DataFrame
        DataFrame with at least movieId and genres_list columns.
    top_n : int
        How many recommendations to return.

    Returns
    -------
    list[dict]
        Sorted list of {"movieId": ..., "score": ...} dicts,
        excluding the input movie itself.
    """
    target_row = movies_df[movies_df["movieId"] == movie_id]
    if target_row.empty:
        return []

    target_genres = set(target_row.iloc[0]["genres_list"])

    scores = []
    for _, row in movies_df.iterrows():
        if row["movieId"] == movie_id:
            continue
        candidate_genres = set(row["genres_list"])
        score = jaccard_similarity(target_genres, candidate_genres)
        scores.append({"movieId": int(row["movieId"]), "score": round(score, 4)})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]


# ---------------------------------------------------------------------------
# Manual cosine similarity (for demonstration / testing)
# ---------------------------------------------------------------------------

def cosine_similarity_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D NumPy arrays.

    This is a manual implementation provided for educational clarity.
    The recommender uses scikit-learn's optimised version for the full matrix.

    Formula:
        cos(theta) = (A · B) / (||A|| × ||B||)

    Returns 0.0 when either vector is a zero vector.
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
