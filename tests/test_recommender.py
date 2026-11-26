"""
tests/test_recommender.py
=========================
Unit tests for recommender.py and preprocessing.py.
"""

import pandas as pd
import pytest

from src.preprocessing import (
    aggregate_tags,
    build_movie_profiles,
    clean_title,
    extract_genres,
    genres_to_set,
)
from src.recommender import MovieRecommender


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic movie data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_movies_raw():
    """Small raw movies DataFrame (mimics movies.csv structure)."""
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "title": [
            "Interstellar (2014)", "Gravity (2013)", "The Martian (2015)",
            "Arrival (2016)", "Toy Story (1995)", "Shrek (2001)",
            "The Dark Knight (2008)", "Inception (2010)",
            "The Notebook (2004)", "Titanic (1997)",
        ],
        "genres": [
            "Sci-Fi|Adventure|Drama", "Sci-Fi|Drama|Thriller",
            "Sci-Fi|Drama|Adventure", "Sci-Fi|Drama|Mystery",
            "Animation|Comedy|Family", "Animation|Comedy|Fantasy",
            "Action|Crime|Drama|Thriller", "Action|Adventure|Sci-Fi|Thriller",
            "Drama|Romance", "Drama|Romance",
        ],
    })


@pytest.fixture
def sample_tags():
    """Small synthetic tags DataFrame."""
    return pd.DataFrame({
        "userId": [1, 1, 2, 2, 3],
        "movieId": [1, 1, 2, 3, 7],
        "tag": ["space", "science", "gravity", "mars", "batman"],
        "timestamp": [100, 101, 102, 103, 104],
    })


@pytest.fixture
def sample_profiles(sample_movies_raw, sample_tags):
    """Pre-built movie profiles DataFrame."""
    return build_movie_profiles(sample_movies_raw, sample_tags)


@pytest.fixture
def recommender(sample_profiles):
    """MovieRecommender fitted on sample data."""
    return MovieRecommender(sample_profiles)


# ---------------------------------------------------------------------------
# preprocessing.py tests
# ---------------------------------------------------------------------------

class TestExtractGenres:

    def test_pipe_separated(self):
        assert extract_genres("Action|Adventure|Sci-Fi") == ["Action", "Adventure", "Sci-Fi"]

    def test_single_genre(self):
        assert extract_genres("Drama") == ["Drama"]

    def test_empty_string(self):
        assert extract_genres("") == []

    def test_no_genres_listed(self):
        assert extract_genres("(no genres listed)") == []

    def test_non_string_input(self):
        assert extract_genres(None) == []
        assert extract_genres(123) == []


class TestGenresToSet:

    def test_returns_lowercase_set(self):
        result = genres_to_set("Action|Drama")
        assert result == {"action", "drama"}

    def test_empty_genres(self):
        assert genres_to_set("") == set()


class TestAggregateTags:

    def test_aggregates_per_movie(self, sample_tags):
        result = aggregate_tags(sample_tags)
        assert "movieId" in result.columns
        assert "tags_text" in result.columns
        # Movie 1 has tags "space" and "science"
        movie1_row = result[result["movieId"] == 1]
        assert not movie1_row.empty
        tags_text = movie1_row.iloc[0]["tags_text"]
        assert "space" in tags_text
        assert "science" in tags_text

    def test_empty_tags_df(self):
        empty = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
        result = aggregate_tags(empty)
        assert result.empty

    def test_no_duplicate_tags(self):
        """Duplicate tags for the same movie should appear only once."""
        tags = pd.DataFrame({
            "userId": [1, 2],
            "movieId": [1, 1],
            "tag": ["action", "action"],
            "timestamp": [100, 101],
        })
        result = aggregate_tags(tags)
        movie1_text = result[result["movieId"] == 1].iloc[0]["tags_text"]
        # "action" should appear only once
        assert movie1_text.count("action") == 1


class TestBuildMovieProfiles:

    def test_has_required_columns(self, sample_profiles):
        for col in ["movieId", "title", "genres_list", "tags_text", "profile", "year"]:
            assert col in sample_profiles.columns, f"Missing column: {col}"

    def test_year_extraction(self, sample_profiles):
        interstellar = sample_profiles[sample_profiles["movieId"] == 1].iloc[0]
        assert interstellar["year"] == 2014

    def test_no_tags_movie_has_empty_tags_text(self, sample_profiles):
        """Movies without tags should have empty string tags_text, not NaN."""
        toy_story = sample_profiles[sample_profiles["movieId"] == 5].iloc[0]
        assert isinstance(toy_story["tags_text"], str)
        assert toy_story["tags_text"] == ""

    def test_profile_not_empty_for_genres(self, sample_profiles):
        """Every movie with genres should have a non-empty profile."""
        for _, row in sample_profiles.iterrows():
            if row["genres_list"]:
                assert row["profile"].strip() != ""


class TestCleanTitle:

    def test_removes_year(self):
        assert clean_title("Toy Story (1995)") == "Toy Story"

    def test_no_year(self):
        assert clean_title("Batman") == "Batman"

    def test_empty_string(self):
        assert clean_title("") == ""

    def test_non_string(self):
        assert clean_title(None) == ""


# ---------------------------------------------------------------------------
# recommender.py tests
# ---------------------------------------------------------------------------

class TestMovieRecommender:

    def test_builds_without_error(self, recommender):
        """MovieRecommender should initialise and build the model."""
        assert recommender.cosine_sim is not None
        assert recommender.tfidf_matrix is not None
        assert recommender.vectorizer is not None

    def test_recommend_by_movie_excludes_self(self, recommender):
        """The query movie must NOT appear in its own recommendations."""
        recs = recommender.recommend_by_movie(1, top_n=5)
        ids = [r["movieId"] for r in recs]
        assert 1 not in ids

    def test_recommend_by_movie_sorted(self, recommender):
        """Recommendations must be sorted by score descending."""
        recs = recommender.recommend_by_movie(1, top_n=5)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_by_movie_count(self, recommender):
        """Should return at most top_n results."""
        recs = recommender.recommend_by_movie(1, top_n=3)
        assert len(recs) <= 3

    def test_recommend_unknown_id_returns_empty(self, recommender):
        """Unknown movie ID should return an empty list (not raise)."""
        recs = recommender.recommend_by_movie(9999, top_n=5)
        assert recs == []

    def test_recommend_result_has_required_fields(self, recommender):
        """Each recommendation dict must have the required keys."""
        recs = recommender.recommend_by_movie(1, top_n=5)
        required_keys = {"movieId", "title", "year", "genres_list", "score", "overlap_features"}
        for rec in recs:
            assert required_keys.issubset(rec.keys()), f"Missing keys in {rec}"

    def test_scores_in_valid_range(self, recommender):
        """Cosine similarity scores must be in [0.0, 1.0]."""
        recs = recommender.recommend_by_movie(1, top_n=10)
        for rec in recs:
            assert 0.0 <= rec["score"] <= 1.0, f"Score out of range: {rec['score']}"

    def test_recommend_by_favourites_excludes_favourites(self, recommender):
        """Favourite movies must not appear in the recommendations."""
        fav_ids = [1, 2]
        recs = recommender.recommend_by_favourites(fav_ids, top_n=5)
        rec_ids = [r["movieId"] for r in recs]
        for fid in fav_ids:
            assert fid not in rec_ids

    def test_recommend_by_favourites_multiple(self, recommender):
        """Multi-favourite recommendations should return a non-empty list."""
        recs = recommender.recommend_by_favourites([1, 2, 3], top_n=5)
        assert len(recs) > 0

    def test_recommend_by_favourites_empty_list(self, recommender):
        """Empty favourites list should return an empty list."""
        recs = recommender.recommend_by_favourites([], top_n=5)
        assert recs == []

    def test_jaccard_recommend_excludes_self(self, recommender):
        """Jaccard recommendations must not include the query movie."""
        recs = recommender.recommend_by_jaccard(1, top_n=5)
        ids = [r["movieId"] for r in recs]
        assert 1 not in ids

    def test_sci_fi_recommends_sci_fi(self, recommender):
        """Sci-Fi movies should recommend other Sci-Fi movies."""
        recs = recommender.recommend_by_movie(1, top_n=5)  # Interstellar (Sci-Fi)
        # At least some results should have Sci-Fi
        sci_fi_count = sum(
            1 for r in recs if "Sci-Fi" in r.get("genres_list", [])
        )
        assert sci_fi_count >= 1


# ---------------------------------------------------------------------------
# user_preferences.py tests
# ---------------------------------------------------------------------------

class TestUserPreferences:

    @pytest.fixture(autouse=True)
    def patch_prefs_path(self, tmp_path, monkeypatch):
        """Redirect preferences file to a temp path for isolation."""
        import src.user_preferences as up
        prefs_path = tmp_path / "preferences.json"
        monkeypatch.setattr(up, "PREFS_PATH", prefs_path)

    def test_load_returns_defaults_when_file_missing(self):
        from src.user_preferences import load_preferences
        prefs = load_preferences()
        assert prefs == {"favourites": [], "watchlist": []}

    def test_add_to_favourites(self):
        from src.user_preferences import add_to_favourites, load_preferences
        add_to_favourites(42)
        prefs = load_preferences()
        assert 42 in prefs["favourites"]

    def test_add_to_favourites_idempotent(self):
        from src.user_preferences import add_to_favourites, load_preferences
        add_to_favourites(42)
        add_to_favourites(42)  # second call should not duplicate
        prefs = load_preferences()
        assert prefs["favourites"].count(42) == 1

    def test_remove_from_favourites(self):
        from src.user_preferences import add_to_favourites, load_preferences, remove_from_favourites
        add_to_favourites(42)
        remove_from_favourites(42)
        prefs = load_preferences()
        assert 42 not in prefs["favourites"]

    def test_add_to_watchlist(self):
        from src.user_preferences import add_to_watchlist, load_preferences
        add_to_watchlist(99)
        prefs = load_preferences()
        assert 99 in prefs["watchlist"]

    def test_remove_from_watchlist(self):
        from src.user_preferences import add_to_watchlist, load_preferences, remove_from_watchlist
        add_to_watchlist(99)
        remove_from_watchlist(99)
        prefs = load_preferences()
        assert 99 not in prefs["watchlist"]

    def test_is_favourite(self):
        from src.user_preferences import add_to_favourites, is_favourite
        add_to_favourites(10)
        assert is_favourite(10) is True
        assert is_favourite(999) is False

    def test_is_in_watchlist(self):
        from src.user_preferences import add_to_watchlist, is_in_watchlist
        add_to_watchlist(20)
        assert is_in_watchlist(20) is True
        assert is_in_watchlist(888) is False

    def test_invalid_json_returns_defaults(self, tmp_path, monkeypatch):
        """A corrupted JSON file should not crash – return defaults instead."""
        import src.user_preferences as up
        prefs_path = tmp_path / "preferences.json"
        prefs_path.write_text("{ INVALID JSON !! }")
        monkeypatch.setattr(up, "PREFS_PATH", prefs_path)
        from src.user_preferences import load_preferences
        prefs = load_preferences()
        assert prefs == {"favourites": [], "watchlist": []}
