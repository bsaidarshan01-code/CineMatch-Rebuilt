"""
movie_details.py
================
Functions for retrieving detailed metadata about individual movies.
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing import clean_title


def get_movie_details(
    movie_id: int,
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    links_df: pd.DataFrame,
) -> dict | None:
    """Return a comprehensive metadata dict for one movie.

    Parameters
    ----------
    movie_id : int
        The MovieLens movie ID.
    movies_df, ratings_df, tags_df, links_df : pd.DataFrame
        The four loaded DataFrames.

    Returns
    -------
    dict or None
        None when the movie_id is not found.
    """
    movie_row = movies_df[movies_df["movieId"] == movie_id]
    if movie_row.empty:
        return None
    movie_row = movie_row.iloc[0]

    # ---- Rating statistics (from ratings.csv) ----
    movie_ratings = ratings_df[ratings_df["movieId"] == movie_id]["rating"]
    avg_rating = round(float(movie_ratings.mean()), 2) if not movie_ratings.empty else None
    num_ratings = len(movie_ratings)

    # ---- Tags (from tags.csv) ----
    movie_tags = tags_df[tags_df["movieId"] == movie_id]["tag"].unique().tolist()

    # ---- External links (from links.csv) ----
    link_row = links_df[links_df["movieId"] == movie_id]
    imdb_id = link_row.iloc[0]["imdbId"] if not link_row.empty else None
    tmdb_id = link_row.iloc[0]["tmdbId"] if not link_row.empty else None

    return {
        "movieId": int(movie_id),
        "title": movie_row.get("title", ""),
        "clean_title": clean_title(str(movie_row.get("title", ""))),
        "year": movie_row.get("year"),
        "genres_list": movie_row.get("genres_list", []),
        "genres_str": " | ".join(movie_row.get("genres_list", [])),
        "tags": movie_tags,
        "avg_rating": avg_rating,
        "num_ratings": num_ratings,
        "imdb_id": imdb_id,
        "tmdb_id": tmdb_id,
    }


def search_movies(query: str, movies_df: pd.DataFrame, max_results: int = 20) -> pd.DataFrame:
    """Search movies by partial title (case-insensitive).

    Parameters
    ----------
    query : str
        Partial or full title to search for.
    movies_df : pd.DataFrame
        The processed movies DataFrame.
    max_results : int
        Upper limit on results returned.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with matching rows, limited to max_results.
        Returns an empty DataFrame for empty queries.
    """
    if not query or not str(query).strip():
        return pd.DataFrame()

    q = str(query).strip().lower()
    mask = movies_df["title"].str.lower().str.contains(q, regex=False, na=False)
    return movies_df[mask].head(max_results).reset_index(drop=True)
