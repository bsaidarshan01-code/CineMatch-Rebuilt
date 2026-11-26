"""
app.py
======
CineMatch – Intelligent Movie Recommendation System
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import html
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project root on sys.path (so `src` package imports work)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    all_files_present,
    dataset_status,
    download_instructions,
    load_links,
    load_movies,
    load_ratings,
    load_tags,
)
from src.movie_details import get_movie_details, search_movies
from src.preprocessing import build_movie_profiles
from src.recommender import CollaborativeRecommender, MovieRecommender
from src.user_preferences import (
    add_to_favourites,
    add_to_watchlist,
    is_favourite,
    is_in_watchlist,
    load_preferences,
    remove_from_favourites,
    remove_from_watchlist,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.poster_service import get_movie_poster_url, posters_configured

# ---------------------------------------------------------------------------
# Custom CSS – cinematic, responsive dark interface
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --cm-bg: #080b12;
        --cm-panel: #111722;
        --cm-panel-2: #151e2c;
        --cm-border: #253246;
        --cm-text: #f3f5f8;
        --cm-muted: #9aa8bb;
        --cm-accent: #f2b766;
        --cm-accent-2: #e56b55;
    }
    .stApp {
        background:
            radial-gradient(circle at 88% 0%, rgba(111, 57, 76, .18), transparent 26rem),
            radial-gradient(circle at 0% 22%, rgba(41, 83, 116, .17), transparent 28rem),
            var(--cm-bg);
        color: var(--cm-text);
    }
    [data-testid="stHeader"] { background: rgba(8, 11, 18, .88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101827 0%, #0b1019 100%);
        border-right: 1px solid var(--cm-border);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--cm-muted); }
    [data-testid="stSidebar"] .stRadio label { color: #d9e0e9 !important; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: -.035em; color: var(--cm-text); }
    p, li { color: #d3dbe6; }
    a { color: var(--cm-accent) !important; }
    .brand-mark {
        color: var(--cm-text); font-size: 1.55rem; font-weight: 900;
        letter-spacing: -.06em; margin: 0;
    }
    .brand-mark span { color: var(--cm-accent); }
    .brand-caption { color: #7f90a6; font-size: .72rem; letter-spacing: .16em;
        text-transform: uppercase; margin-top: -.2rem; }
    .hero-banner {
        position: relative; overflow: hidden;
        background: linear-gradient(115deg, rgba(21, 34, 52, .98), rgba(32, 22, 34, .96) 58%, rgba(80, 38, 43, .90));
        border: 1px solid rgba(242, 183, 102, .32); border-radius: 24px;
        padding: clamp(2rem, 5vw, 4.2rem); margin: .4rem 0 1.7rem;
        box-shadow: 0 22px 70px rgba(0,0,0,.25);
    }
    .hero-banner:after {
        content: ""; position: absolute; right: -5rem; top: -9rem; width: 27rem; height: 27rem;
        border: 1px solid rgba(242,183,102,.12); border-radius: 50%;
        box-shadow: 0 0 0 2rem rgba(242,183,102,.025), 0 0 0 4rem rgba(242,183,102,.02);
    }
    .hero-kicker { color: var(--cm-accent); font-size: .75rem; letter-spacing: .2em;
        text-transform: uppercase; font-weight: 800; margin: 0 0 .8rem; }
    .hero-title { font-size: clamp(2.3rem, 6vw, 4.8rem); line-height: .98;
        font-weight: 950; color: #fff; letter-spacing: -.075em; margin: 0; }
    .hero-title span { color: var(--cm-accent); }
    .hero-subtitle { color: #c0c9d5; max-width: 42rem; font-size: 1.05rem;
        line-height: 1.65; margin: 1.1rem 0 0; }
    .section-header { color: #f4f6fa; font-size: 1.45rem; font-weight: 850;
        letter-spacing: -.04em; padding: .15rem 0 .65rem; margin: 1.25rem 0 1rem;
        border-bottom: 1px solid var(--cm-border); }
    .eyebrow { color: var(--cm-accent); text-transform: uppercase; letter-spacing: .16em;
        font-size: .7rem; font-weight: 800; margin-bottom: .45rem; }
    .stat-card { background: linear-gradient(145deg, #141e2d, #101722);
        border: 1px solid var(--cm-border); border-radius: 17px; padding: 1.1rem 1.15rem; }
    .stat-number { color: var(--cm-accent); font-size: clamp(1.5rem, 3vw, 2.15rem);
        line-height: 1.1; font-weight: 900; letter-spacing: -.05em; }
    .stat-label { color: var(--cm-muted); font-size: .78rem; margin-top: .35rem; }
    .movie-info-card { background: linear-gradient(145deg, #151f2e, #101722);
        border: 1px solid var(--cm-border); border-radius: 16px; padding: 1rem 1.1rem;
        min-height: 100%; }
    .movie-info-card:hover { border-color: rgba(242,183,102,.55); }
    .card-title { color: #f7f8fa; font-size: 1.04rem; line-height: 1.25;
        font-weight: 800; margin: 0 0 .4rem; }
    .card-meta { color: var(--cm-muted); font-size: .8rem; line-height: 1.5; margin: .18rem 0; }
    .genre-tag, .feature-pill { display: inline-block; border-radius: 999px;
        padding: 3px 9px; margin: 2px 3px 2px 0; font-size: .68rem; font-weight: 650; }
    .genre-tag { color: #f5d8a8; background: rgba(242,183,102,.1);
        border: 1px solid rgba(242,183,102,.2); }
    .feature-pill { color: #b7d7f0; background: rgba(82,145,190,.12);
        border: 1px solid rgba(82,145,190,.22); }
    .card-score { display: inline-block; color: #17120b; background: var(--cm-accent);
        border-radius: 999px; padding: 4px 10px; font-size: .72rem; font-weight: 850; margin-top: .55rem; }
    .poster-placeholder { width: 100%; aspect-ratio: 2 / 3; display: flex; flex-direction: column;
        align-items: center; justify-content: center; gap: .5rem; border-radius: 12px;
        color: #8e9aab; text-align: center; border: 1px solid #2a3749;
        background: linear-gradient(145deg, #192333, #10151f); padding: .5rem; }
    .poster-placeholder .poster-icon { font-size: 2rem; }
    .poster-placeholder small { color: #8290a3; font-size: .65rem; }
    [data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--cm-border); }
    .info-box { background: rgba(25, 42, 59, .72); border: 1px solid rgba(94,153,196,.22);
        border-left: 3px solid #70b6df; color: #c0d0df; padding: 1rem 1.15rem;
        border-radius: 12px; margin: .8rem 0; font-size: .9rem; }
    .feature-card { background: #111a27; border: 1px solid var(--cm-border);
        border-radius: 16px; padding: 1.15rem; min-height: 145px; }
    .feature-icon { font-size: 1.45rem; }
    .feature-title { color: #f2f5f8; font-weight: 800; margin: .45rem 0 .25rem; }
    .feature-copy { color: var(--cm-muted); font-size: .82rem; line-height: 1.45; }
    .detail-panel { background: linear-gradient(145deg, #151f2e, #101722);
        border: 1px solid var(--cm-border); border-radius: 20px; padding: 1.3rem; }
    .detail-title { font-size: clamp(1.45rem, 3vw, 2rem); font-weight: 900;
        color: #fff; letter-spacing: -.05em; margin: 0; }
    .detail-stat { color: var(--cm-accent); font-size: 1.2rem; font-weight: 850; }
    .muted-note { color: var(--cm-muted); font-size: .78rem; }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(100deg, #efb363, #e18a62); color: #17120c;
        border: 0; border-radius: 10px; font-weight: 800; padding: .5rem 1rem;
        transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: #17120c; filter: brightness(1.07); transform: translateY(-1px);
    }
    .stTextInput input, .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] { background: #111a27 !important; }
    div[data-baseweb="input"] { border-radius: 10px; }
    [data-testid="stMetric"] { background: #111a27; padding: .8rem; border-radius: 12px; }
    hr { border-color: var(--cm-border) !important; }
    @media (max-width: 700px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero-banner { border-radius: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached so it runs only once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset …")
def load_all_data():
    """Load, clean, and preprocess the full MovieLens dataset."""
    movies_raw = load_movies()
    ratings_df = load_ratings()
    tags_df = load_tags()
    links_df = load_links()
    movies_df = build_movie_profiles(movies_raw, tags_df)
    return movies_df, ratings_df, tags_df, links_df


@st.cache_resource(show_spinner="Building recommendation engine …")
def build_recommender(movies_df: pd.DataFrame) -> MovieRecommender:
    """Build and cache the TF-IDF recommendation model."""
    return MovieRecommender(movies_df)


@st.cache_resource(show_spinner="Building collaborative filtering model …")
def build_collaborative(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> CollaborativeRecommender:
    return CollaborativeRecommender(ratings_df, movies_df)


# ---------------------------------------------------------------------------
# Helper UI functions
# ---------------------------------------------------------------------------

def render_page_header(title: str, subtitle: str, icon: str = "🎬") -> None:
    """Consistent page heading used across the app."""
    st.markdown(
        f'<div class="eyebrow">{icon} &nbsp; CINEMATCH</div>'
        f'<h1 style="margin:0 0 .35rem 0">{html.escape(title)}</h1>'
        f'<p style="color:#9aa8bb;margin:0 0 1.2rem 0">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def get_poster_url(movie_id: int, links_df: pd.DataFrame) -> str | None:
    """Resolve MovieLens movieId -> TMDB / IMDb ID -> poster URL."""
    if links_df is None or links_df.empty:
        return None
    rows = links_df.loc[links_df["movieId"] == int(movie_id), ["tmdbId", "imdbId"]]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return get_movie_poster_url(tmdb_id=row.get("tmdbId"), imdb_id=row.get("imdbId"))


def render_poster(movie_id: int, title: str, links_df: pd.DataFrame, width: int = 138) -> None:
    """Show a TMDB poster or a styled placeholder if one is not available."""
    poster_url = get_poster_url(movie_id, links_df)
    if poster_url:
        st.image(poster_url, width=width)
    else:
        safe_title = html.escape(str(title)[:42])
        st.markdown(
            '<div class="poster-placeholder">'
            '<span class="poster-icon">🎞️</span>'
            f'<small>{safe_title}</small>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_movie_card(movie: dict, rank: int = None, links_df: pd.DataFrame | None = None) -> None:
    """Render a polished horizontal recommendation card with its poster."""
    links_df = links_df if links_df is not None else pd.DataFrame(columns=["movieId", "tmdbId"])
    movie_id = int(movie.get("movieId", 0))
    title = str(movie.get("title", "Untitled"))
    year = movie.get("year")
    year_str = f" · {int(year)}" if year is not None and pd.notna(year) else ""
    genres = movie.get("genres_list", []) or []
    genres_html = "".join(f'<span class="genre-tag">{html.escape(str(g))}</span>' for g in genres[:5])
    score = movie.get("score")
    score_html = f'<span class="card-score">Match strength · {int(float(score) * 100)}%</span>' if score is not None else ""
    features = str(movie.get("overlap_features", ""))
    if features and features != "No shared features":
        features_html = "".join(
            f'<span class="feature-pill">{html.escape(feature.strip())}</span>'
            for feature in features.split(",")[:5]
        )
    else:
        features_html = ""
    rank_text = f'<span style="color:#f2b766">#{rank} &nbsp;</span>' if rank else ""

    image_col, info_col = st.columns([1.05, 5], gap="medium", vertical_alignment="center")
    with image_col:
        render_poster(movie_id, title, links_df, width=124)
    with info_col:
        st.markdown(
            f'<div class="movie-info-card">'
            f'<p class="card-title">{rank_text}{html.escape(title)}{year_str}</p>'
            f'<div class="card-meta">{genres_html}</div>'
            f'<div class="card-meta" style="margin-top:.35rem">{features_html}</div>'
            f'{score_html}'
            f'<div class="muted-note" style="margin-top:.45rem">Similarity is an algorithmic content match, not a predicted user rating.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_catalog_tile(row, links_df: pd.DataFrame) -> None:
    """Compact poster tile for search and genre browsing grids."""
    movie_id = int(row["movieId"])
    title = str(row["title"])
    year = row.get("year")
    year_str = str(int(year)) if year is not None and pd.notna(year) else "Year unavailable"
    genres = row.get("genres_list", []) or []
    genre_text = " · ".join(str(g) for g in genres[:2]) or "Uncategorised"
    with st.container():
        render_poster(movie_id, title, links_df, width=170)
        st.markdown(
            f'<div style="padding:.55rem .1rem .9rem">'
            f'<div class="card-title">{html.escape(title)}</div>'
            f'<div class="card-meta">{html.escape(year_str)} · {html.escape(genre_text)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_detail_card(details: dict, ratings_df: pd.DataFrame, links_df: pd.DataFrame) -> None:
    """Render selected movie metadata next to its poster."""
    year_str = f"({int(details['year'])})" if details.get("year") is not None and pd.notna(details.get("year")) else ""
    genres_html = "".join(f'<span class="genre-tag">{html.escape(str(g))}</span>' for g in details.get("genres_list", []))
    tags_html = "".join(
        f'<span class="feature-pill">{html.escape(str(tag))}</span>'
        for tag in (details.get("tags") or [])[:12]
    )
    avg_rating = details.get("avg_rating")
    rating_value = f"{avg_rating:.2f}/5" if avg_rating is not None else "N/A"
    imdb_link = ""
    imdb_id = str(details.get("imdb_id", ""))
    if imdb_id and imdb_id.lower() != "nan":
        imdb_link = f'<a href="https://www.imdb.com/title/tt{imdb_id.zfill(7)}/" target="_blank" rel="noopener">Open on IMDb ↗</a>'

    poster_col, info_col = st.columns([1.15, 4], gap="large", vertical_alignment="top")
    with poster_col:
        render_poster(details["movieId"], details.get("title", ""), links_df, width=205)
    with info_col:
        st.markdown(
            f'<div class="detail-panel">'
            f'<h2 class="detail-title">{html.escape(str(details.get("clean_title", details.get("title", ""))))}</h2>'
            f'<div class="card-meta">{year_str} &nbsp; · &nbsp; MovieLens ID {int(details["movieId"])}</div>'
            f'<div style="margin:.8rem 0">{genres_html}</div>'
            f'<div style="display:flex;gap:2.3rem;flex-wrap:wrap;margin:.7rem 0 1rem">'
            f'<div><div class="muted-note">COMMUNITY RATING</div><div class="detail-stat">⭐ {rating_value}</div></div>'
            f'<div><div class="muted-note">TOTAL RATINGS</div><div class="detail-stat">{int(details.get("num_ratings", 0)):,}</div></div>'
            f'</div>{imdb_link}'
            f'{"<div style='margin-top:1rem'><div class='muted-note'>USER TAGS</div>" + tags_html + "</div>" if tags_html else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def page_home(movies_df, ratings_df, tags_df, links_df, recommender):
    """Landing page with product-style hero, data stats and visual search."""
    st.markdown(
        """
        <div class="hero-banner">
            <p class="hero-kicker">YOUR NEXT GREAT WATCH STARTS HERE</p>
            <p class="hero-title">Find your next<br><span>favourite film.</span></p>
            <p class="hero-subtitle">CineMatch learns what makes a movie feel like your kind of movie.
            Explore recommendations built from genres, community tags and similarity algorithms.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_movies = len(movies_df)
    total_ratings = len(ratings_df)
    tagged_movies = movies_df[movies_df["tags_text"].str.strip() != ""].shape[0]
    vocab_size = recommender.get_vocab_size()

    stat_cols = st.columns(4)
    for col, value, label in [
        (stat_cols[0], f"{total_movies:,}", "MOVIES IN THE CATALOG"),
        (stat_cols[1], f"{total_ratings:,}", "COMMUNITY RATINGS"),
        (stat_cols[2], f"{tagged_movies:,}", "MOVIES WITH TAGS"),
        (stat_cols[3], f"{vocab_size:,}", "TF-IDF FEATURES"),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="stat-number">{value}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-header">Search the collection</p>', unsafe_allow_html=True)
    query = st.text_input(
        "Search by movie title",
        placeholder="Try Interstellar, Toy Story, The Dark Knight…",
        label_visibility="collapsed",
    )
    if query:
        results = search_movies(query, movies_df, max_results=10)
        if results.empty:
            st.warning(f'No movies found matching "{query}". Try another title.')
        else:
            st.caption(f"Showing {len(results)} matching titles")
            for start_idx in range(0, len(results), 5):
                cols = st.columns(5, gap="medium")
                chunk = results.iloc[start_idx:start_idx + 5]
                for col, (_, row) in zip(cols, chunk.iterrows()):
                    with col:
                        render_catalog_tile(row, links_df)

    st.markdown('<p class="section-header">How CineMatch works</p>', unsafe_allow_html=True)
    feature_cols = st.columns(3, gap="medium")
    features = [
        ("🧠", "Understands content", "Combines movie genres and community tags into a searchable profile."),
        ("✨", "Measures similarity", "Uses TF-IDF vectors and cosine similarity to compare movie profiles."),
        ("🎯", "Learns your taste", "Blend multiple favourites to discover films that match your overall taste."),
    ]
    for col, (icon, title, copy) in zip(feature_cols, features):
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div><div class="feature-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )

    if not posters_configured():
        st.caption("Poster images are optional. Add a TMDB or OMDb API key in Streamlit Cloud → Settings → Secrets to enable them.")


def page_discover(movies_df, ratings_df, tags_df, links_df, recommender):
    """Search for a movie, inspect it, then compare similar titles."""
    render_page_header("Discover movies", "Choose a film you enjoy and let CineMatch find similar titles.", "🎯")

    with st.container(border=True):
        algo_col, count_col = st.columns([3, 1])
        with algo_col:
            algo = st.radio(
                "Recommendation method",
                ["TF-IDF + Cosine Similarity", "Jaccard Similarity (Genre-based)"],
                horizontal=True,
                help="TF-IDF uses genres and tags. Jaccard compares genre sets only.",
            )
        with count_col:
            top_n = st.slider("Results", 5, 20, 10)
        query = st.text_input("Find a movie", placeholder="Start typing a movie title…")

    if not query:
        st.markdown(
            '<div class="info-box">Start with a movie you already like. CineMatch will use its content profile to find related films.</div>',
            unsafe_allow_html=True,
        )
        return

    results = search_movies(query, movies_df, max_results=20)
    if results.empty:
        st.warning(f'No movies found for "{query}". Try a shorter or different term.')
        return

    options = {}
    for _, row in results.iterrows():
        year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
        options[f"{row['title']}{year}"] = int(row["movieId"])

    selected_label = st.selectbox("Choose the exact title", list(options.keys()))
    selected_id = options[selected_label]
    details = get_movie_details(selected_id, movies_df, ratings_df, tags_df, links_df)

    if details:
        st.markdown('<p class="section-header">Your selected movie</p>', unsafe_allow_html=True)
        render_detail_card(details, ratings_df, links_df)

        action_cols = st.columns(2)
        with action_cols[0]:
            if is_favourite(selected_id):
                if st.button("♥ Remove from favourites", key="selected_remove_fav", use_container_width=True):
                    remove_from_favourites(selected_id)
                    st.rerun()
            else:
                if st.button("♥ Add to favourites", key="selected_add_fav", use_container_width=True):
                    add_to_favourites(selected_id)
                    st.success("Saved to favourites.")
        with action_cols[1]:
            if is_in_watchlist(selected_id):
                if st.button("✓ Remove from watchlist", key="selected_remove_watchlist", use_container_width=True):
                    remove_from_watchlist(selected_id)
                    st.rerun()
            else:
                if st.button("＋ Add to watchlist", key="selected_add_watchlist", use_container_width=True):
                    add_to_watchlist(selected_id)
                    st.success("Added to your watchlist.")

    st.markdown('<p class="section-header">Recommended for this movie</p>', unsafe_allow_html=True)
    with st.spinner("Finding movies with a similar profile…"):
        if algo == "TF-IDF + Cosine Similarity":
            recs = recommender.recommend_by_movie(selected_id, top_n=top_n)
            algo_note = "TF-IDF + cosine similarity · genres and tags"
        else:
            recs = recommender.recommend_by_jaccard(selected_id, top_n=top_n)
            algo_note = "Jaccard similarity · genre overlap"

    if not recs:
        st.warning("Could not generate recommendations for this movie.")
        return

    st.caption(algo_note)
    for rank, movie in enumerate(recs, start=1):
        render_movie_card(movie, rank=rank, links_df=links_df)
        action1, action2, _ = st.columns([1.2, 1.2, 5])
        mid = int(movie["movieId"])
        with action1:
            if st.button("＋ Watchlist", key=f"wl_{mid}_{rank}", use_container_width=True):
                add_to_watchlist(mid)
                st.toast("Added to watchlist")
        with action2:
            if st.button("♥ Favourite", key=f"fav_{mid}_{rank}", use_container_width=True):
                add_to_favourites(mid)
                st.toast("Added to favourites")


def page_recommendations(movies_df, ratings_df, tags_df, links_df, recommender, collab_recommender):
    """Multiple favourites recommendation page."""
    render_page_header("Made for your taste", "Combine a few favourite movies to build a taste profile.", "🌟")

    st.markdown(
        """
        <div class="info-box">
            Select <strong>2–10 movies</strong> you enjoy. CineMatch will average their TF-IDF profiles
            to build a combined taste vector and find movies that match all your preferences together.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Build a full title list for the multiselect
    title_map = {}
    for _, row in movies_df.iterrows():
        year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
        label = f"{row['title']}{year} [ID:{row['movieId']}]"
        title_map[label] = int(row["movieId"])

    selected_labels = st.multiselect(
        "Choose your favourite movies:",
        list(title_map.keys()),
        max_selections=10,
        placeholder="Search and select movies …",
    )

    if len(selected_labels) < 2:
        st.info("Select at least 2 movies to generate combined recommendations.")
        return

    selected_ids = [title_map[lbl] for lbl in selected_labels]

    algo = st.radio(
        "Algorithm:",
        ["TF-IDF + Cosine (Combined Profile)", "Collaborative Filtering"],
        horizontal=True,
    )
    top_n = st.slider("Recommendations to show", 5, 20, 10, key="multi_topn")

    if st.button("🚀 Generate Recommendations"):
        with st.spinner("Building combined taste profile …"):
            if algo == "TF-IDF + Cosine (Combined Profile)":
                recs = recommender.recommend_by_favourites(selected_ids, top_n=top_n)
                note = "Combined TF-IDF average profile → cosine similarity"
            else:
                if collab_recommender is None:
                    st.warning("Collaborative filtering is unavailable for this session. Try the TF-IDF option.")
                    return
                # Collaborative mode currently uses the first selected movie as its seed.
                recs = collab_recommender.recommend(selected_ids[0], top_n=top_n)
                note = "Collaborative filtering (item-item cosine on ratings matrix)"

        if not recs:
            st.warning("Could not generate recommendations. Try different movies.")
            return

        st.success(f"Top {len(recs)} recommendations based on your combined taste:")
        st.caption(f"Algorithm: {note}")
        for rank, movie in enumerate(recs, start=1):
            render_movie_card(movie, rank=rank, links_df=links_df)


def page_genre_explorer(movies_df, ratings_df, tags_df, links_df):
    """Browse the catalog by one or more genres."""
    render_page_header("Explore by genre", "Filter the catalog and browse films by the kind of story you want.", "🎭")

    all_genres = sorted({genre for values in movies_df["genres_list"] for genre in values})
    filter_col, sort_col = st.columns([3, 1])
    with filter_col:
        selected_genres = st.multiselect(
            "Genres", all_genres, default=["Action"] if "Action" in all_genres else [],
            placeholder="Choose one or more genres…",
        )
    with sort_col:
        sort_by = st.selectbox("Sort", ["Title (A-Z)", "Year (Newest)", "Year (Oldest)"])

    if not selected_genres:
        st.info("Choose at least one genre to see matching movies.")
        return

    match_all = st.checkbox("Require every selected genre", value=False)
    if match_all:
        mask = movies_df["genres_list"].apply(lambda values: all(g in values for g in selected_genres))
    else:
        mask = movies_df["genres_list"].apply(lambda values: any(g in values for g in selected_genres))

    filtered = movies_df[mask].copy()
    if sort_by == "Title (A-Z)":
        filtered = filtered.sort_values("title")
    elif sort_by == "Year (Newest)":
        filtered = filtered.sort_values("year", ascending=False, na_position="last")
    else:
        filtered = filtered.sort_values("year", ascending=True, na_position="last")

    st.markdown(f"**{len(filtered):,} films found** · showing up to 50")
    st.divider()
    # Keep poster API requests modest; TMDB has request-rate limits.
    display_rows = filtered.head(25)
    for start_idx in range(0, len(display_rows), 5):
        cols = st.columns(5, gap="medium")
        chunk = display_rows.iloc[start_idx:start_idx + 5]
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                render_catalog_tile(row, links_df)

    if len(filtered) > 25:
        st.caption(f"Showing the first 25 of {len(filtered):,} matches. Refine your genre selection to narrow the list.")


def page_watchlist(movies_df, ratings_df, tags_df, links_df, recommender):
    """Manage saved movies with poster-led rows."""
    render_page_header("My collection", "Keep track of films you want to watch and the ones you love.", "📋")
    prefs = load_preferences()
    tab_wl, tab_fav = st.tabs([f"📌 Watchlist · {len(prefs['watchlist'])}", f"♥ Favourites · {len(prefs['favourites'])}"])

    def render_saved_list(movie_ids: list, list_key: str, remove_fn):
        if not movie_ids:
            st.markdown(
                '<div class="info-box">Nothing saved here yet. Head to Discover and add a movie to build your collection.</div>',
                unsafe_allow_html=True,
            )
            return
        for mid in movie_ids:
            details = get_movie_details(mid, movies_df, ratings_df, tags_df, links_df)
            if not details:
                continue
            image_col, info_col, action_col = st.columns([1, 4.8, 1.1], gap="medium", vertical_alignment="center")
            with image_col:
                render_poster(mid, details["title"], links_df, width=115)
            with info_col:
                year = f" · {int(details['year'])}" if details.get("year") is not None and pd.notna(details.get("year")) else ""
                genres = " · ".join(details.get("genres_list", [])[:4])
                rating_str = f"⭐ {details['avg_rating']}/5 · {details['num_ratings']:,} ratings" if details.get("avg_rating") else "No community rating available"
                st.markdown(
                    f'<div class="movie-info-card"><div class="card-title">{html.escape(details["title"])}{year}</div>'
                    f'<div class="card-meta">{html.escape(genres)}</div>'
                    f'<div class="card-meta">{html.escape(rating_str)}</div></div>',
                    unsafe_allow_html=True,
                )
            with action_col:
                if st.button("Remove", key=f"remove_{list_key}_{mid}", use_container_width=True):
                    remove_fn(mid)
                    st.rerun()

    with tab_wl:
        st.caption(f"{len(prefs['watchlist'])} saved movie(s)")
        render_saved_list(prefs["watchlist"], "wl", remove_from_watchlist)

    with tab_fav:
        st.caption(f"{len(prefs['favourites'])} saved movie(s)")
        render_saved_list(prefs["favourites"], "fav", remove_from_favourites)
        if len(prefs["favourites"]) >= 2:
            st.markdown('<p class="section-header">Recommendations based on your favourites</p>', unsafe_allow_html=True)
            if st.button("✨ Generate recommendations", use_container_width=False):
                recs = recommender.recommend_by_favourites(prefs["favourites"], top_n=10)
                for rank, movie in enumerate(recs, start=1):
                    render_movie_card(movie, rank=rank, links_df=links_df)


def page_about():
    """About page explaining the project, algorithms and data attribution."""
    render_page_header("About CineMatch", "A transparent look at the data and algorithms behind your recommendations.", "ℹ️")

    st.markdown(
        """
        ## CineMatch – Intelligent Movie Recommendation System

        CineMatch is a Python mini-project that demonstrates **content-based filtering**
        using real movie data from the [MovieLens dataset](https://grouplens.org/datasets/movielens/).

        ---

        ### 📐 Recommendation Algorithms

        #### 1. TF-IDF (Term Frequency–Inverse Document Frequency)
        Each movie gets a *profile* by combining its genres and user tags.
        TF-IDF converts these profiles into numerical vectors:

        | Term | Meaning |
        |------|---------|
        | **TF** | How often a word appears in *this* movie's profile |
        | **IDF** | How rare that word is across *all* movies |
        | **TF × IDF** | High weight for distinctive, movie-specific words |

        > Words like *"Drama"* appear in thousands of movies → low IDF → low weight.
        > Words like *"heist"* appear rarely → high IDF → high weight.

        #### 2. Cosine Similarity
        After TF-IDF vectorisation, each movie is a point in high-dimensional space.
        Cosine similarity measures the **angle** between two movie vectors:

        ```
        similarity = (A · B) / (||A|| × ||B||)
        ```

        Score = 1.0 → identical profiles. Score = 0.0 → no shared features.

        #### 3. Jaccard Similarity (Genre-based)
        A simpler baseline using **genre sets**:

        ```
        J(A, B) = |A ∩ B| / |A ∪ B|
        ```

        Example:
        - Movie A genres: {Action, Thriller}
        - Movie B genres: {Action, Drama}
        - Intersection: {Action} → size 1
        - Union: {Action, Thriller, Drama} → size 3
        - Jaccard score: 1/3 ≈ 0.33

        ---

        ### 🏗️ Project Architecture

        ```
        Movie-Recommendation-System/
        ├── app.py               ← Streamlit interface
        ├── download_data.py     ← Dataset download helper
        ├── src/
        │   ├── data_loader.py   ← CSV loading & validation
        │   ├── preprocessing.py ← Data cleaning & profile building
        │   ├── similarity.py    ← Jaccard & cosine formulas
        │   ├── recommender.py   ← TF-IDF model & recommendation logic
        │   ├── movie_details.py ← Movie metadata & search
        │   ├── poster_service.py ← TMDB poster lookup & caching
        │   └── user_preferences.py ← Watchlist / favourites (JSON)
        └── tests/               ← Pytest test suite
        ```

        ---

        ### 📦 Dataset
        [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/)
        by GroupLens Research, University of Minnesota.

        Contains ~9,700 movies and ~100,000 ratings from ~600 users.

        ---

        ### ⚠️ Limitations
        - Cold-start: new movies with no tags rely on genres only.
        - Genre labels are coarse; two "Action" movies can be very different.
        - Similarity scores are content matches, **not** rating predictions.
        - No real user authentication; preferences are stored locally.
        """
    )


    st.markdown('<p class="section-header">Movie poster images</p>', unsafe_allow_html=True)
    st.markdown(
        """
        CineMatch uses the TMDB API to retrieve poster paths for the TMDB IDs already included in the MovieLens `links.csv` file.
        To enable poster images, add one of the following values in **Streamlit Cloud → App → Settings → Secrets**:

        ```toml
        TMDB_READ_ACCESS_TOKEN = "your_tmdb_api_read_access_token"
        ```

        You can also use `TMDB_API_KEY` with your TMDB v3 API key. Keep credentials in Secrets or environment variables; do not commit them to GitHub.

        **This product uses the TMDB API but is not endorsed or certified by TMDB.**
        """
    )
    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;margin:.5rem 0 1.5rem">'
        '<img src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_2-d537fb228cf3ded904ef09b136fe3fec72548ebc1fea3fbbd1ad9e36364db38b.svg" '
        'alt="TMDB logo" width="42" style="height:auto">'
        '<span class="muted-note">Movie poster imagery provided by <a href="https://www.themoviedb.org" target="_blank" rel="noopener">TMDB</a>.</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Load data/models once, then route to the selected Streamlit page."""
    with st.sidebar:
        st.markdown(
            '<p class="brand-mark">cine<span>match</span> 🎬</p>'
            '<p class="brand-caption">A smarter way to choose</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🎯 Discover", "🌟 For You", "🎭 Genres", "📋 My List", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Powered by MovieLens data and classical recommendation algorithms.")
        if not posters_configured():
            st.caption("Poster images: connect a TMDB or OMDb key in app Secrets.")

    if not all_files_present():
        with st.spinner("Downloading the MovieLens dataset for the first launch…"):
            try:
                from download_data import download_and_extract
                download_and_extract()
                st.rerun()
            except Exception as exc:
                status = dataset_status()
                st.error(f"Dataset missing and automatic download failed: {exc}")
                st.markdown(download_instructions())
                for name, present in status.items():
                    st.markdown(f"{'✅' if present else '❌'} `{name}.csv`")
                st.stop()

    try:
        movies_df, ratings_df, tags_df, links_df = load_all_data()
    except Exception as exc:
        logger.exception("Dataset loading failed")
        st.error(f"Error loading dataset: {exc}")
        st.stop()

    try:
        recommender = build_recommender(movies_df)
    except Exception as exc:
        logger.exception("Recommendation model initialization failed")
        st.error(f"Error building recommendation model: {exc}")
        st.stop()

    try:
        collab_recommender = build_collaborative(ratings_df, movies_df)
    except Exception as exc:
        logger.exception("Collaborative filtering initialization failed")
        st.warning(f"Collaborative filtering unavailable: {exc}")
        collab_recommender = None

    if page == "🏠 Home":
        page_home(movies_df, ratings_df, tags_df, links_df, recommender)
    elif page == "🎯 Discover":
        page_discover(movies_df, ratings_df, tags_df, links_df, recommender)
    elif page == "🌟 For You":
        page_recommendations(movies_df, ratings_df, tags_df, links_df, recommender, collab_recommender)
    elif page == "🎭 Genres":
        page_genre_explorer(movies_df, ratings_df, tags_df, links_df)
    elif page == "📋 My List":
        page_watchlist(movies_df, ratings_df, tags_df, links_df, recommender)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
