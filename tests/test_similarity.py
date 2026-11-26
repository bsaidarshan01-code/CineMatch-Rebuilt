"""
tests/test_similarity.py
========================
Unit tests for similarity.py – Jaccard and cosine similarity functions.
"""

import numpy as np
import pytest

from src.similarity import cosine_similarity_pair, jaccard_similarity, jaccard_recommend
import pandas as pd


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:

    def test_identical_sets(self):
        """Identical genre sets should give score of 1.0."""
        a = {"action", "thriller"}
        assert jaccard_similarity(a, a) == 1.0

    def test_disjoint_sets(self):
        """Completely different genres should give 0.0."""
        a = {"action", "thriller"}
        b = {"romance", "comedy"}
        assert jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        """Two sets sharing one out of three elements -> 1/3."""
        a = {"action", "thriller"}
        b = {"action", "drama"}
        score = jaccard_similarity(a, b)
        assert pytest.approx(score, abs=1e-6) == 1 / 3

    def test_both_empty(self):
        """Two empty sets should return 0.0 (not division by zero)."""
        assert jaccard_similarity(set(), set()) == 0.0

    def test_one_empty(self):
        """One empty set against a non-empty set should return 0.0."""
        assert jaccard_similarity(set(), {"action"}) == 0.0

    def test_symmetry(self):
        """Jaccard similarity must be symmetric: J(A,B) == J(B,A)."""
        a = {"action", "sci-fi", "thriller"}
        b = {"sci-fi", "drama"}
        assert jaccard_similarity(a, b) == jaccard_similarity(b, a)

    def test_score_range(self):
        """Score must always be in [0.0, 1.0]."""
        a = {"action", "comedy", "drama"}
        b = {"drama", "romance"}
        score = jaccard_similarity(a, b)
        assert 0.0 <= score <= 1.0

    def test_full_subset(self):
        """When set_b is a full subset of set_a."""
        a = {"action", "comedy", "drama"}
        b = {"action", "comedy"}
        score = jaccard_similarity(a, b)
        # |intersection|=2, |union|=3
        assert pytest.approx(score) == 2 / 3


# ---------------------------------------------------------------------------
# Jaccard recommend
# ---------------------------------------------------------------------------

class TestJaccardRecommend:

    @pytest.fixture
    def sample_movies(self):
        return pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
            "genres_list": [
                ["Action", "Thriller"],
                ["Action", "Drama"],
                ["Romance", "Comedy"],
                ["Thriller", "Horror"],
            ],
        })

    def test_excludes_query_movie(self, sample_movies):
        """The query movie should NOT appear in its own recommendations."""
        recs = jaccard_recommend(1, sample_movies, top_n=10)
        ids = [r["movieId"] for r in recs]
        assert 1 not in ids

    def test_returns_correct_count(self, sample_movies):
        """Should return at most top_n results."""
        recs = jaccard_recommend(1, sample_movies, top_n=2)
        assert len(recs) <= 2

    def test_sorted_descending(self, sample_movies):
        """Results must be sorted by score descending."""
        recs = jaccard_recommend(1, sample_movies, top_n=10)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_movie_id(self, sample_movies):
        """Should return an empty list for an unknown movie ID."""
        recs = jaccard_recommend(9999, sample_movies, top_n=5)
        assert recs == []

    def test_score_values(self, sample_movies):
        """All scores should be in [0, 1]."""
        recs = jaccard_recommend(1, sample_movies, top_n=10)
        for r in recs:
            assert 0.0 <= r["score"] <= 1.0


# ---------------------------------------------------------------------------
# Cosine similarity (manual)
# ---------------------------------------------------------------------------

class TestCosineSimilarityPair:

    def test_identical_vectors(self):
        """Cosine similarity of a vector with itself should be 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert pytest.approx(cosine_similarity_pair(v, v), abs=1e-6) == 1.0

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have cosine similarity 0.0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert pytest.approx(cosine_similarity_pair(a, b), abs=1e-6) == 0.0

    def test_zero_vector(self):
        """Zero vector should return 0.0 (no division by zero error)."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity_pair(a, b) == 0.0

    def test_symmetry(self):
        """cosine_similarity_pair(a, b) == cosine_similarity_pair(b, a)."""
        a = np.array([1.0, 0.5, 2.0])
        b = np.array([0.0, 1.0, 1.0])
        assert pytest.approx(cosine_similarity_pair(a, b)) == pytest.approx(cosine_similarity_pair(b, a))

    def test_range(self):
        """Score should be in [0, 1] for non-negative vectors."""
        a = np.array([3.0, 1.0, 0.0])
        b = np.array([2.0, 2.0, 1.0])
        score = cosine_similarity_pair(a, b)
        assert 0.0 <= score <= 1.0

    def test_known_value(self):
        """Test a known cosine similarity calculation."""
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        # cos(45 degrees) = 1/sqrt(2) ≈ 0.7071
        expected = 1.0 / np.sqrt(2)
        assert pytest.approx(cosine_similarity_pair(a, b), abs=1e-6) == expected
