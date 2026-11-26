"""
tests/test_data_loader.py
=========================
Unit tests for data_loader.py – CSV loading and validation.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    _validate_columns,
    all_files_present,
    dataset_status,
    download_instructions,
)


# ---------------------------------------------------------------------------
# dataset_status / all_files_present
# ---------------------------------------------------------------------------

class TestDatasetStatus:

    def test_returns_dict(self):
        """dataset_status() should return a dict."""
        result = dataset_status()
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """dataset_status() must contain all four file keys."""
        result = dataset_status()
        assert set(result.keys()) == {"movies", "ratings", "tags", "links"}

    def test_all_files_present_returns_bool(self):
        """all_files_present() must return a boolean."""
        result = all_files_present()
        assert isinstance(result, bool)

    def test_download_instructions_is_string(self):
        """download_instructions() must return a non-empty string."""
        result = download_instructions()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_download_instructions_contains_url(self):
        """download_instructions() should include the dataset URL."""
        result = download_instructions()
        assert "grouplens.org" in result


# ---------------------------------------------------------------------------
# _validate_columns (internal helper)
# ---------------------------------------------------------------------------

class TestValidateColumns:

    def test_valid_columns_no_error(self):
        """Should not raise when all expected columns are present."""
        df = pd.DataFrame({"movieId": [1], "title": ["A"], "genres": ["Action"]})
        _validate_columns(df, ["movieId", "title", "genres"], "movies.csv")  # no exception

    def test_missing_column_raises(self):
        """Should raise ValueError when a required column is missing."""
        df = pd.DataFrame({"movieId": [1], "title": ["A"]})
        with pytest.raises(ValueError, match="genres"):
            _validate_columns(df, ["movieId", "title", "genres"], "movies.csv")

    def test_error_message_includes_filename(self):
        """ValueError message should mention the filename."""
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="test_file.csv"):
            _validate_columns(df, ["movieId"], "test_file.csv")

    def test_all_columns_missing(self):
        """Should list all missing columns in the error."""
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError):
            _validate_columns(df, ["movieId", "title", "genres"], "movies.csv")


# ---------------------------------------------------------------------------
# CSV loading with temporary files (integration-style unit tests)
# ---------------------------------------------------------------------------

class TestLoadMoviesFromTempFile:
    """Tests that create minimal CSV files in a temp directory and load them."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path, monkeypatch):
        """Patch DATA_DIR and REQUIRED_FILES to point to a temp directory."""
        import src.data_loader as dl
        monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
        monkeypatch.setattr(dl, "REQUIRED_FILES", {
            "movies":  tmp_path / "movies.csv",
            "ratings": tmp_path / "ratings.csv",
            "tags":    tmp_path / "tags.csv",
            "links":   tmp_path / "links.csv",
        })
        return tmp_path

    def test_load_movies_success(self, temp_data_dir, monkeypatch):
        """load_movies() should return a clean DataFrame from a valid CSV."""
        from src.data_loader import load_movies
        csv_content = "movieId,title,genres\n1,Toy Story (1995),Animation|Comedy\n2,Jumanji (1995),Adventure\n"
        (temp_data_dir / "movies.csv").write_text(csv_content)
        df = load_movies()
        assert len(df) == 2
        assert "movieId" in df.columns
        assert df["movieId"].dtype in [int, "int64", "int32"]

    def test_load_movies_missing_file_raises(self, temp_data_dir):
        """load_movies() must raise FileNotFoundError when CSV is absent."""
        from src.data_loader import load_movies
        with pytest.raises(FileNotFoundError):
            load_movies()

    def test_load_movies_drops_null_movieid(self, temp_data_dir):
        """Rows with missing movieId should be dropped."""
        from src.data_loader import load_movies
        csv_content = "movieId,title,genres\n1,Valid Movie,Action\n,Bad Row,Drama\n"
        (temp_data_dir / "movies.csv").write_text(csv_content)
        df = load_movies()
        assert len(df) == 1

    def test_load_ratings_success(self, temp_data_dir):
        """load_ratings() should return a valid DataFrame."""
        from src.data_loader import load_ratings
        csv_content = "userId,movieId,rating,timestamp\n1,1,4.0,964982703\n1,2,3.0,964981247\n"
        (temp_data_dir / "ratings.csv").write_text(csv_content)
        df = load_ratings()
        assert len(df) == 2
        assert df["rating"].dtype == float

    def test_load_tags_success(self, temp_data_dir):
        """load_tags() should lowercase all tags."""
        from src.data_loader import load_tags
        csv_content = "userId,movieId,tag,timestamp\n1,1,Action,964982703\n1,1,FUNNY,964981247\n"
        (temp_data_dir / "tags.csv").write_text(csv_content)
        df = load_tags()
        assert all(df["tag"].str.islower() | (df["tag"] == df["tag"].str.lower()))
