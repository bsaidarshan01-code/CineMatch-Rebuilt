"""
recommender.py
==============
Core recommendation engine for CineMatch.

Implements:
  * TF-IDF vectorisation of movie profiles.
  * Cosine similarity matrix computation.
  * Content-based recommendations (single movie or multiple favourites).
  * Jaccard similarity recommendations (delegated to similarity.py).
  * Basic collaborative filtering using the ratings matrix.

Designed as a class so the expensive TF-IDF matrix is built once and reused
across multiple recommendation calls within a Streamlit session.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.similarity import jaccard_recommend

logger = logging.getLogger(__name__)


class MovieRecommender:
    """Content-based movie recommendation engine.

    Attributes
    ----------
    movies_df : pd.DataFrame
        Processed movie DataFrame (output of preprocessing.build_movie_profiles).
    tfidf_matrix : scipy.sparse matrix
        TF-IDF feature matrix; rows = movies, columns = vocabulary terms.
    vectorizer : TfidfVectorizer
        Fitted scikit-learn vectorizer.
    cosine_sim : np.ndarray
        Full pairwise cosine similarity matrix (movies x movies).
    """

    def __init__(self, movies_df: pd.DataFrame) -> None:
        """Initialise and immediately fit the TF-IDF model.

        Parameters
        ----------
        movies_df : pd.DataFrame
            Must contain at least movieId and profile columns.
        """
        self.movies_df = movies_df.reset_index(drop=True)
        # Build a fast movieId -> DataFrame-index lookup
        self._id_to_idx: dict = {
            int(mid): idx
            for idx, mid in enumerate(self.movies_df["movieId"])
        }

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.cosine_sim: Optional[np.ndarray] = None

        self._build_tfidf_model()

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _build_tfidf_model(self) -> None:
        """Fit the TF-IDF vectorizer on all movie profiles and compute
        the full cosine similarity matrix.

        Why TF-IDF?
        -----------
        * TF (Term Frequency) - how often a word appears in a movie's profile.
        * IDF (Inverse Document Frequency) - how rare that word is across ALL
          movie profiles.  Common words like "Drama" appear everywhere and get
          lower IDF weight; distinctive tags like "heist" appear rarely and
          get higher weight.
        * Multiplying TF x IDF gives a score that rewards words that are
          frequent in this movie but rare across the dataset.
        """
        profiles = self.movies_df["profile"].fillna("").tolist()

        # min_df=2: ignore words appearing in only 1 movie (too unique / noisy).
        # max_df=0.8: ignore words appearing in >80% of movies (too generic).
        # ngram_range=(1,2): consider single words AND two-word phrases.
        self.vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.8,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(profiles)
        logger.info(
            "TF-IDF matrix: %d movies x %d features.",
            self.tfidf_matrix.shape[0],
            self.tfidf_matrix.shape[1],
        )

        # Pre-compute the full (n_movies x n_movies) similarity matrix once.
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        logger.info("Cosine similarity matrix computed.")

    # ------------------------------------------------------------------
    # Content-based: single movie
    # ------------------------------------------------------------------

    def recommend_by_movie(self, movie_id: int, top_n: int = 10) -> list:
        """Return the top-N most similar movies to movie_id.

        Parameters
        ----------
        movie_id : int
            The selected movie's ID.
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        list[dict]
            Recommendation dicts sorted by descending similarity score.
        """
        idx = self._id_to_idx.get(int(movie_id))
        if idx is None:
            logger.warning("movie_id %d not found in index.", movie_id)
            return []

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        # Exclude the query movie itself from results
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = sim_scores[:top_n]

        return self._format_results(top_indices)

    # ------------------------------------------------------------------
    # Content-based: multiple favourites
    # ------------------------------------------------------------------

    def recommend_by_favourites(self, movie_ids: list, top_n: int = 10) -> list:
        """Generate recommendations based on multiple favourite movies.

        How it works:
            1. Retrieve the TF-IDF vector for each favourite movie.
            2. Average those vectors -> a single combined "preference profile".
            3. Compute cosine similarity of that profile against all movies.
            4. Exclude the favourite movies themselves.
            5. Return the top-N highest-scoring movies.

        Parameters
        ----------
        movie_ids : list[int]
            List of movie IDs the user likes.
        top_n : int
            Number of recommendations to return.
        """
        valid_indices = []
        for mid in movie_ids:
            idx = self._id_to_idx.get(int(mid))
            if idx is not None:
                valid_indices.append(idx)
            else:
                logger.warning("Favourite movie_id %d not found; skipping.", mid)

        if not valid_indices:
            return []

        # Average TF-IDF vectors -> combined preference profile
        fav_vectors = self.tfidf_matrix[valid_indices]   # sparse slice
        # NOTE: sparse .mean(axis=0) returns np.matrix in older scipy versions.
        # np.asarray() converts it to a plain ndarray that cosine_similarity accepts.
        combined_vector = np.asarray(fav_vectors.mean(axis=0))  # shape (1, n_features)

        # Cosine similarity of combined vector vs every movie
        sim_scores_matrix = cosine_similarity(combined_vector, self.tfidf_matrix)
        # sim_scores_matrix may be np.matrix; flatten to a plain 1-D array first.
        sim_array = np.asarray(sim_scores_matrix).flatten()
        sim_scores = list(enumerate(sim_array))

        # Exclude all selected favourites
        excluded = set(valid_indices)
        sim_scores = [(i, s) for i, s in sim_scores if i not in excluded]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = sim_scores[:top_n]

        return self._format_results(top_indices)

    # ------------------------------------------------------------------
    # Jaccard-based recommendations
    # ------------------------------------------------------------------

    def recommend_by_jaccard(self, movie_id: int, top_n: int = 10) -> list:
        """Wrapper around jaccard_recommend that enriches output with movie info."""
        raw = jaccard_recommend(movie_id, self.movies_df, top_n=top_n)
        results = []
        for item in raw:
            mid = item["movieId"]
            row = self.movies_df[self.movies_df["movieId"] == mid]
            if row.empty:
                continue
            row = row.iloc[0]
            results.append({
                "movieId": mid,
                "title": row.get("title", ""),
                "year": row.get("year"),
                "genres_list": row.get("genres_list", []),
                "score": item["score"],
                "overlap_features": "Genre-based (Jaccard similarity)",
            })
        return results

    # ------------------------------------------------------------------
    # Vocabulary / model introspection (educational)
    # ------------------------------------------------------------------

    def get_vocabulary_sample(self, n: int = 20) -> list:
        """Return a sample of TF-IDF vocabulary terms (for the About page)."""
        if self.vectorizer is None:
            return []
        vocab = list(self.vectorizer.vocabulary_.keys())
        return vocab[:n]

    def get_top_features_for_movie(self, movie_id: int, top_n: int = 10) -> list:
        """Return the top TF-IDF feature words for a given movie."""
        idx = self._id_to_idx.get(int(movie_id))
        if idx is None or self.vectorizer is None:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        row = self.tfidf_matrix[idx]
        scores = np.asarray(row.todense()).flatten()
        top_indices = scores.argsort()[::-1][:top_n]
        return [feature_names[i] for i in top_indices if scores[i] > 0]

    def get_vocab_size(self) -> int:
        """Return the number of unique terms in the TF-IDF vocabulary."""
        if self.vectorizer is None:
            return 0
        return len(self.vectorizer.vocabulary_)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_results(self, indexed_scores: list) -> list:
        """Convert (DataFrame-index, score) pairs into rich recommendation dicts."""
        results = []
        for idx, score in indexed_scores:
            row = self.movies_df.iloc[idx]
            features = self.get_top_features_for_movie(int(row["movieId"]))
            results.append({
                "movieId": int(row["movieId"]),
                "title": row.get("title", ""),
                "year": row.get("year"),
                "genres_list": row.get("genres_list", []),
                "score": round(float(score), 4),
                "overlap_features": ", ".join(features) if features else "No shared features",
            })
        return results


# ---------------------------------------------------------------------------
# Collaborative Filtering (bonus feature)
# ---------------------------------------------------------------------------

class CollaborativeRecommender:
    """Item-based collaborative filtering using the user-rating matrix.

    NOTE: MovieLens user IDs are dataset identifiers only - they do NOT
    correspond to real authenticated users of this application.
    """

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        self.movies_df = movies_df
        self._item_sim: Optional[np.ndarray] = None
        self._movie_ids: list = []
        self._build_model(ratings_df)

    def _build_model(self, ratings_df: pd.DataFrame) -> None:
        """Build the user-item matrix and compute item-item cosine similarity."""
        # Pivot: rows = userId, columns = movieId, values = rating
        # NaN means "not rated" (not 0 - important distinction!)
        pivot = ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self._movie_ids = list(pivot.columns)

        # Fill NaN with 0 only for the cosine computation.
        # We acknowledge that 0 != "disliked" - it means "unrated".
        item_matrix = pivot.fillna(0).values.T  # shape: (n_movies, n_users)

        self._item_sim = cosine_similarity(item_matrix)
        logger.info(
            "Collaborative model: %d x %d item-similarity matrix built.",
            len(self._movie_ids), len(self._movie_ids),
        )

    def recommend(self, movie_id: int, top_n: int = 10) -> list:
        """Recommend movies similar to movie_id based on shared rating patterns."""
        if movie_id not in self._movie_ids:
            return []
        idx = self._movie_ids.index(movie_id)
        sim_row = list(enumerate(self._item_sim[idx]))
        sim_row = [(i, s) for i, s in sim_row if i != idx]
        sim_row.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in sim_row[:top_n]:
            mid = self._movie_ids[i]
            row = self.movies_df[self.movies_df["movieId"] == mid]
            if row.empty:
                continue
            row = row.iloc[0]
            results.append({
                "movieId": mid,
                "title": row.get("title", ""),
                "year": row.get("year"),
                "genres_list": row.get("genres_list", []),
                "score": round(float(score), 4),
                "overlap_features": "Rating-pattern based (collaborative filtering)",
            })
        return results
