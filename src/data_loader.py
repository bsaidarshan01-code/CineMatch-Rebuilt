"""
data_loader.py
==============
Handles loading and basic validation of the MovieLens CSV dataset files.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REQUIRED_FILES = {
    "movies": DATA_DIR / "movies.csv",
    "ratings": DATA_DIR / "ratings.csv",
    "tags": DATA_DIR / "tags.csv",
    "links": DATA_DIR / "links.csv",
}

DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def dataset_status() -> dict:
    """Return a dict mapping each required CSV name to whether it exists."""
    return {name: path.exists() for name, path in REQUIRED_FILES.items()}


def all_files_present() -> bool:
    """Return True when every required CSV file is present."""
    return all(dataset_status().values())


def download_instructions() -> str:
    """Return a user-friendly string with dataset download instructions."""
    return f"""
Dataset not found in the `data/` folder.

Please download the MovieLens Latest Small dataset:

1. Run the helper script from the project root:
       python download_data.py

   OR download manually from:
       {DOWNLOAD_URL}

2. Extract the ZIP file.
3. Copy the following files into the `data/` folder:
   - movies.csv
   - ratings.csv
   - tags.csv
   - links.csv
"""


def load_movies() -> pd.DataFrame:
    """Load movies.csv and return a cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        Movies with valid movieId (int), title (str), genres (str).
    """
    path = REQUIRED_FILES["movies"]
    if not path.exists():
        raise FileNotFoundError(
            f"movies.csv not found at {path}\n{download_instructions()}"
        )

    df = pd.read_csv(path)
    _validate_columns(df, ["movieId", "title", "genres"], "movies.csv")

    df = df.dropna(subset=["movieId", "title"])
    df["movieId"] = df["movieId"].astype(int)
    df["title"] = df["title"].str.strip()
    df["genres"] = (
        df["genres"].fillna("").str.replace("(no genres listed)", "", regex=False)
    )

    logger.info("Loaded %d movies.", len(df))
    return df.drop_duplicates(subset=["movieId"]).reset_index(drop=True)


def load_ratings() -> pd.DataFrame:
    """Load ratings.csv and return a cleaned DataFrame."""
    path = REQUIRED_FILES["ratings"]
    if not path.exists():
        raise FileNotFoundError(
            f"ratings.csv not found at {path}\n{download_instructions()}"
        )

    df = pd.read_csv(path)
    _validate_columns(df, ["userId", "movieId", "rating", "timestamp"], "ratings.csv")

    df = df.dropna(subset=["userId", "movieId", "rating"])
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)

    logger.info("Loaded %d ratings.", len(df))
    return df.reset_index(drop=True)


def load_tags() -> pd.DataFrame:
    """Load tags.csv and return a cleaned DataFrame."""
    path = REQUIRED_FILES["tags"]
    if not path.exists():
        raise FileNotFoundError(
            f"tags.csv not found at {path}\n{download_instructions()}"
        )

    df = pd.read_csv(path)
    _validate_columns(df, ["userId", "movieId", "tag", "timestamp"], "tags.csv")

    df = df.dropna(subset=["movieId", "tag"])
    df["movieId"] = df["movieId"].astype(int)
    df["tag"] = df["tag"].astype(str).str.strip().str.lower()

    logger.info("Loaded %d tags.", len(df))
    return df.reset_index(drop=True)


def load_links() -> pd.DataFrame:
    """Load links.csv and return a cleaned DataFrame."""
    path = REQUIRED_FILES["links"]
    if not path.exists():
        raise FileNotFoundError(
            f"links.csv not found at {path}\n{download_instructions()}"
        )

    df = pd.read_csv(path, dtype={"imdbId": str, "tmdbId": str})
    _validate_columns(df, ["movieId", "imdbId", "tmdbId"], "links.csv")

    df = df.dropna(subset=["movieId"])
    df["movieId"] = df["movieId"].astype(int)

    logger.info("Loaded %d link records.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, expected: list, filename: str) -> None:
    """Raise ValueError when required columns are missing."""
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
