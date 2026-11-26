"""TMDB poster lookup helpers.

MovieLens provides TMDB IDs but not poster paths. This module uses those IDs
with TMDB's movie-details endpoint and builds the image CDN URL from the
returned poster_path. Credentials are read from environment variables or
Streamlit secrets; never store API credentials in source control.
"""

from __future__ import annotations

import os
from typing import Optional

import requests
import streamlit as st

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


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


def tmdb_is_configured() -> bool:
    """Whether either supported TMDB application credential is configured."""
    token, key = get_tmdb_credentials()
    return bool(token or key)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_poster_path(
    tmdb_id: int,
    read_access_token: str,
    api_key: str,
) -> Optional[str]:
    """Fetch a movie poster path; cached for one day per movie and credential."""
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
