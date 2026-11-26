"""Movie poster lookup helpers (TMDB and OMDb).

MovieLens provides TMDB IDs and IMDb IDs, but not poster image URLs. This
module supports both TMDB (via tmdbId) and OMDb (via imdbId) and caches poster
lookups for 24 hours. Credentials are read from environment variables or
Streamlit secrets; never store API credentials in source control.
"""

from __future__ import annotations

import os
from typing import Optional

import requests
import streamlit as st

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
OMDB_API_BASE = "https://www.omdbapi.com/"


def get_tmdb_credentials() -> tuple[str, str]:
    """Return (read_access_token, v3_api_key), preferring Streamlit secrets."""
    read_token = os.getenv("TMDB_READ_ACCESS_TOKEN", "").strip()
    api_key = os.getenv("TMDB_API_KEY", "").strip()

    try:
        read_token = str(st.secrets.get("TMDB_READ_ACCESS_TOKEN", read_token)).strip()
        api_key = str(st.secrets.get("TMDB_API_KEY", api_key)).strip()
    except Exception:
        # Streamlit secrets are not available during local development unless
        # configured. Environment variables remain a supported alternative.
        pass

    return read_token, api_key


def get_omdb_api_key() -> str:
    """Return the OMDb API key from Streamlit secrets or environment variables."""
    api_key = os.getenv("OMDB_API_KEY", "").strip()

    try:
        api_key = str(st.secrets.get("OMDB_API_KEY", api_key)).strip()
    except Exception:
        pass

    return api_key


def tmdb_is_configured() -> bool:
    """Whether either supported TMDB application credential is configured."""
    token, key = get_tmdb_credentials()
    return bool(token or key)


def omdb_is_configured() -> bool:
    """Whether an OMDb API key is configured."""
    return bool(get_omdb_api_key())


def posters_configured() -> bool:
    """Whether at least one poster provider (TMDB or OMDb) is configured."""
    return tmdb_is_configured() or omdb_is_configured()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_poster_path(
    tmdb_id: int,
    read_access_token: str,
    api_key: str,
) -> Optional[str]:
    """Fetch a movie poster path from TMDB; cached for one day."""
    if not tmdb_id or (not read_access_token and not api_key):
        return None

    headers = {"accept": "application/json"}
    params = {"language": "en-US"}
    if read_access_token:
        headers["Authorization"] = f"Bearer {read_access_token}"
    else:
        params["api_key"] = api_key

    try:
        response = requests.get(
            f"{TMDB_API_BASE}/movie/{int(tmdb_id)}",
            headers=headers,
            params=params,
            timeout=5,
        )
        if response.status_code in (401, 403, 404, 429):
            return None
        response.raise_for_status()
        payload = response.json()
        poster_path = payload.get("poster_path")
        if not isinstance(poster_path, str) or not poster_path.startswith("/"):
            return None
        return poster_path
    except (requests.RequestException, ValueError, TypeError):
        return None


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_omdb_poster_url(imdb_tt_id: str, api_key: str) -> Optional[str]:
    """Fetch a movie poster URL from OMDb using an IMDb tt-ID; cached for one day."""
    if not imdb_tt_id or not api_key:
        return None

    try:
        response = requests.get(
            OMDB_API_BASE,
            params={"i": imdb_tt_id, "apikey": api_key},
            timeout=5,
        )
        if response.status_code in (401, 403, 404, 429):
            return None
        response.raise_for_status()
        payload = response.json()
        if payload.get("Response") != "True":
            return None
        poster_url = payload.get("Poster")
        if not isinstance(poster_url, str) or not poster_url.startswith("http"):
            return None
        return poster_url
    except (requests.RequestException, ValueError, TypeError):
        return None


def get_tmdb_poster_url(tmdb_id: object) -> Optional[str]:
    """Return a full w500 poster URL for a MovieLens-linked TMDB ID."""
    if tmdb_id is None:
        return None
    try:
        numeric_id = int(float(tmdb_id))
    except (TypeError, ValueError, OverflowError):
        return None
    if numeric_id <= 0:
        return None

    token, key = get_tmdb_credentials()
    if not token and not key:
        return None

    poster_path = _fetch_poster_path(numeric_id, token, key)
    return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None


def get_omdb_poster_url(imdb_id: object) -> Optional[str]:
    """Return a poster URL from OMDb for a MovieLens-linked IMDb ID."""
    if imdb_id is None:
        return None
    raw = str(imdb_id).strip()
    if not raw or raw.lower() == "nan":
        return None

    # Handle numeric values like "114709" or "114709.0" -> "tt0114709"
    if raw.startswith("tt"):
        tt_id = raw
    else:
        try:
            digits = str(int(float(raw)))
        except (TypeError, ValueError, OverflowError):
            return None
        tt_id = f"tt{digits.zfill(7)}"

    api_key = get_omdb_api_key()
    if not api_key:
        return None

    return _fetch_omdb_poster_url(tt_id, api_key)


def get_movie_poster_url(
    tmdb_id: object = None,
    imdb_id: object = None,
) -> Optional[str]:
    """Return a poster URL using TMDB first, or OMDb as an alternative."""
    url = get_tmdb_poster_url(tmdb_id)
    if url:
        return url
    return get_omdb_poster_url(imdb_id)
