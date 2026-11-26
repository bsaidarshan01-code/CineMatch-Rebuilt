"""
user_preferences.py
===================
Saves and loads user favourites and watchlist to / from a local JSON file.

The JSON structure is:
    {
        "favourites": [<movieId>, ...],
        "watchlist":  [<movieId>, ...]
    }
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PREFS_PATH = Path(__file__).resolve().parent.parent / "user_data" / "preferences.json"
PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)

_DEFAULT_PREFS: dict = {"favourites": [], "watchlist": []}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_preferences() -> dict:
    """Load preferences from JSON.  Returns defaults on any error."""
    if not PREFS_PATH.exists():
        return copy.deepcopy(_DEFAULT_PREFS)
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON object expected.")
        data.setdefault("favourites", [])
        data.setdefault("watchlist", [])
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not load preferences (%s). Using defaults.", exc)
        return copy.deepcopy(_DEFAULT_PREFS)


def save_preferences(prefs: dict) -> None:
    """Write the preferences dict to JSON (overwrites the file)."""
    try:
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
    except OSError as exc:
        logger.error("Could not save preferences: %s", exc)


def add_to_favourites(movie_id: int) -> dict:
    """Add a movie to favourites and persist.  Idempotent."""
    prefs = load_preferences()
    if movie_id not in prefs["favourites"]:
        prefs["favourites"].append(movie_id)
        save_preferences(prefs)
    return prefs


def remove_from_favourites(movie_id: int) -> dict:
    """Remove a movie from favourites and persist."""
    prefs = load_preferences()
    prefs["favourites"] = [m for m in prefs["favourites"] if m != movie_id]
    save_preferences(prefs)
    return prefs


def add_to_watchlist(movie_id: int) -> dict:
    """Add a movie to the watchlist and persist.  Idempotent."""
    prefs = load_preferences()
    if movie_id not in prefs["watchlist"]:
        prefs["watchlist"].append(movie_id)
        save_preferences(prefs)
    return prefs


def remove_from_watchlist(movie_id: int) -> dict:
    """Remove a movie from the watchlist and persist."""
    prefs = load_preferences()
    prefs["watchlist"] = [m for m in prefs["watchlist"] if m != movie_id]
    save_preferences(prefs)
    return prefs


def is_favourite(movie_id: int) -> bool:
    """Return True when movie_id is in favourites."""
    return movie_id in load_preferences()["favourites"]


def is_in_watchlist(movie_id: int) -> bool:
    """Return True when movie_id is in watchlist."""
    return movie_id in load_preferences()["watchlist"]
