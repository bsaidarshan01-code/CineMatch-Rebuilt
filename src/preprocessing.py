"""
preprocessing.py
================
Cleans and merges movie data to create a combined textual profile per movie.
The profile is later used to build TF-IDF feature vectors.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Genre helpers
# ---------------------------------------------------------------------------

def extract_genres(genres_str: str) -> list:
    """Split a pipe-separated genre string into a list of genre tokens.

    Parameters
    ----------
    genres_str : str
        Raw genres string from movies.csv, e.g. "Action|Adventure|Sci-Fi".

    Returns
    -------
    list
        List of individual genres. Returns an empty list for missing/empty input.
    """
    if not isinstance(genres_str, str) or not genres_str.strip():
        return []
    genres_str = genres_str.replace("(no genres listed)", "").strip()
    return [g.strip() for g in genres_str.split("|") if g.strip()]


def genres_to_set(genres_str: str) -> set:
    """Return genres as a set of lowercase strings (for Jaccard similarity)."""
    return {g.lower() for g in extract_genres(genres_str)}


# ---------------------------------------------------------------------------
# Tag aggregation
# ---------------------------------------------------------------------------

def aggregate_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tags per movie into a single whitespace-separated string.

    Parameters
    ----------
    tags_df : pd.DataFrame
        The raw tags DataFrame.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame: movieId (int) and tags_text (str).
    """
    if tags_df.empty:
        return pd.DataFrame(columns=["movieId", "tags_text"])

    aggregated = (
        tags_df.groupby("movieId")["tag"]
        .apply(lambda tags: " ".join(tags.unique()))
        .reset_index()
        .rename(columns={"tag": "tags_text"})
    )
    return aggregated


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_movie_profiles(movies_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined textual profile for every movie.

    The profile concatenates:
      * Genre tokens (e.g. "Action Adventure SciFi").
      * User-supplied tags (e.g. "space exploration future").

    Movies without any tags still get a profile from their genres alone.

    Parameters
    ----------
    movies_df : pd.DataFrame
        Cleaned movies DataFrame.
    tags_df : pd.DataFrame
        Cleaned tags DataFrame.

    Returns
    -------
    pd.DataFrame
        Original movies columns plus:
          * genres_list  – list of genre strings.
          * tags_text    – aggregated tags (empty string when none).
          * profile      – combined textual profile used for TF-IDF.
          * year         – release year extracted from title (int or None).
    """
    movies_df = movies_df.copy()

    # ---- Extract release year from title ----
    movies_df["year"] = movies_df["title"].apply(_extract_year)

    # ---- Build genre tokens ----
    # Replace hyphens so "Sci-Fi" becomes "SciFi" (single token for TF-IDF)
    movies_df["genres_list"] = movies_df["genres"].apply(extract_genres)
    movies_df["genres_tokens"] = movies_df["genres_list"].apply(
        lambda gl: " ".join(g.replace("-", "").replace(" ", "") for g in gl)
    )

    # ---- Aggregate tags ----
    agg_tags = aggregate_tags(tags_df)

    # ---- Merge movies with their aggregated tags ----
    merged = movies_df.merge(agg_tags, on="movieId", how="left")
    merged["tags_text"] = merged["tags_text"].fillna("")

    # ---- Build combined profile ----
    merged["profile"] = (merged["genres_tokens"] + " " + merged["tags_text"]).str.strip()

    logger.info("Built profiles for %d movies.", len(merged))
    return merged.drop(columns=["genres_tokens"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_year(title: str):
    """Extract the 4-digit year from a MovieLens title like 'Toy Story (1995)'."""
    if not isinstance(title, str):
        return None
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return int(match.group(1))
    return None


def clean_title(title: str) -> str:
    """Remove the year suffix from a movie title for cleaner display."""
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
