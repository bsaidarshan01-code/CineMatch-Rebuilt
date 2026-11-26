# CineMatch – Intelligent Movie Recommendation System

> A Python mini-project demonstrating content-based movie recommendations using
> TF-IDF, Cosine Similarity, and Jaccard Similarity on the MovieLens dataset.

---

## 📌 Project Introduction

**CineMatch** is a complete, locally-runnable movie recommendation application
built with Python and Streamlit. It recommends movies based on what you like
using real machine-learning techniques — no manually hardcoded results.

---

## 🎯 Project Objectives

| Objective | Description |
|-----------|-------------|
| Content-based filtering | Recommend movies based on genre and tag similarity |
| TF-IDF vectorisation | Convert movie descriptions into numerical vectors |
| Cosine similarity | Measure similarity between movie vectors |
| Jaccard similarity | Simple genre-set baseline comparison |
| Multi-favourite support | Combine profiles of multiple liked movies |
| Watchlist & favourites | Save personal preferences to JSON |
| Streamlit UI | Interactive dark-themed web interface |

---

## ✨ Features

- 🔍 **Movie Search** – Partial title search, handles typos and multiple results
- 🤖 **TF-IDF Recommendations** – Genre + tag content vectors, cosine similarity
- 📐 **Jaccard Recommendations** – Simple genre set comparison
- 🌟 **Multi-Favourite Mode** – Average TF-IDF profiles of 2–10 favourite movies
- 🎭 **Genre Explorer** – Filter and browse movies by genre
- ❤️ **Watchlist & Favourites** – Saved to `user_data/preferences.json`
- 🎞️ **Movie Details** – Rating statistics, tags, IMDb link, movie ID
- 🔗 **Collaborative Filtering** – Bonus item-based mode using the ratings matrix

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.10+ | Core programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations and array handling |
| Scikit-learn | TF-IDF vectorisation and cosine similarity |
| Streamlit | Interactive web application interface |
| Pathlib | Cross-platform file path handling |
| JSON | Storing user preferences locally |
| Logging | Error tracking and event logging |
| Pytest | Automated unit testing |
| Matplotlib | Optional visualisation |

---

## 📦 Dataset Information

**MovieLens Latest Small** by [GroupLens Research](https://grouplens.org/datasets/movielens/latest/)

| File | Contents |
|------|---------|
| `movies.csv` | 9,742 movies: movieId, title, genres |
| `ratings.csv` | 100,836 ratings: userId, movieId, rating, timestamp |
| `tags.csv` | 3,683 tags: userId, movieId, tag, timestamp |
| `links.csv` | 9,742 links: movieId, imdbId, tmdbId |

**Download URL:**
```
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
```

---

## 📂 Project Folder Structure

```
Movie-Recommendation-System/
│
├── app.py                  ← Streamlit main application
├── download_data.py        ← Dataset download helper script
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── PROJECT_EXPLANATION.md  ← Educational walkthrough
├── VIVA_PREPARATION.md     ← College viva Q&A
│
├── data/                   ← MovieLens CSV files (download separately)
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
│
├── src/                    ← Core Python modules
│   ├── __init__.py
│   ├── data_loader.py      ← CSV loading and validation
│   ├── preprocessing.py    ← Data cleaning, genre/tag processing, profiles
│   ├── similarity.py       ← Jaccard and cosine similarity formulas
│   ├── recommender.py      ← TF-IDF model, MovieRecommender class
│   ├── movie_details.py    ← Movie metadata and search
│   └── user_preferences.py ← Watchlist/favourites in JSON
│
├── tests/                  ← Pytest test suite
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_similarity.py
│   └── test_recommender.py
│
├── user_data/              ← Auto-created; stores preferences.json
│
└── .streamlit/
    └── config.toml         ← Dark theme configuration
```

---

## ⚙️ Installation Requirements

- Python 3.10 or later
- pip (or pip3)
- Internet connection (for downloading the dataset once)

---

## 🚀 Setup and Run (Windows PowerShell)

### Step 1 – Navigate to the project folder

```powershell
cd "C:\Users\HP\Downloads\AATP\Movie-Recommendation-System"
```

### Step 2 – Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3 – Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

> If you get an execution policy error, run first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 4 – Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 5 – Download the dataset

```powershell
python download_data.py
```

This downloads and extracts the MovieLens dataset into the `data/` folder automatically.

> **Manual alternative:** Download from https://files.grouplens.org/datasets/movielens/ml-latest-small.zip,
> extract, and copy `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv` into the `data/` folder.

### Step 6 – Run the application

```powershell
streamlit run app.py
```

The application opens at **http://localhost:8501** in your browser.

---

## 🐧 Setup and Run (macOS / Linux)

```bash
cd ~/Downloads/Movie-Recommendation-System

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python download_data.py

streamlit run app.py
```

---

## 🧪 Running Tests

```powershell
# Windows (with venv active)
pytest tests/ -v

# Or run a specific test file
pytest tests/test_similarity.py -v
```

---

## 🧠 Recommendation Algorithm Explained

### Step 1 – Feature Extraction
Each movie gets a **textual profile** combining its genres and user tags:
```
Interstellar → "SciFi Adventure Drama space science"
```

### Step 2 – TF-IDF Vectorisation
Each profile is converted into a numerical vector using **Term Frequency–Inverse Document Frequency**:

- **TF**: How often a word appears in this movie's profile
- **IDF**: How rare that word is across all movie profiles
- **TF × IDF**: High weight for distinctive words, low weight for common ones

### Step 3 – Cosine Similarity
Two movie vectors are compared using their angle:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Score = 1.0 → identical. Score = 0.0 → completely different.

### Step 4 – Jaccard Similarity (alternative)
Genre sets are compared:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Example: {Action, Thriller} vs {Action, Drama}
→ Intersection = {Action} → size 1
→ Union = {Action, Thriller, Drama} → size 3
→ Score = 1/3 ≈ 0.33

---

## 📸 Screenshots

> *Run the application and explore the interface. Key pages:*
> - **Home** – Stats and quick search
> - **Discover** – Search → Select → Recommendations
> - **For You** – Multi-favourite combined profile
> - **Genres** – Filter by genre
> - **My List** – Watchlist and favourites

---

## ⚠️ Limitations

1. **Cold-start**: Movies with no tags rely on genres only (less distinctive profile)
2. **Genre labels are coarse**: Two "Action" movies can be very different in style
3. **Similarity ≠ rating prediction**: Scores are content matches, not quality measures
4. **No real authentication**: Preferences are stored in a local JSON file
5. **Dataset size**: MovieLens Latest Small covers ~9,700 movies; newer titles may be missing

---

## 🔮 Future Enhancements

- [ ] Integrate TMDB API for movie posters and plot summaries
- [ ] Add hybrid filtering (content + collaborative combined)
- [ ] Matrix factorisation (SVD) for improved collaborative filtering
- [ ] User login with SQLite database
- [ ] Sentiment analysis of user reviews
- [ ] Deploy to Streamlit Cloud for online access

---

## 👨‍💻 Project Information

| Item | Detail |
|------|--------|
| Project Name | CineMatch – Intelligent Movie Recommendation System |
| Type | Python Mini-Project |
| Level | First-year B.Tech Computer Science |
| Dataset | MovieLens Latest Small (GroupLens Research) |
| Algorithm | TF-IDF + Cosine Similarity, Jaccard Similarity |
| Interface | Streamlit (dark theme) |

---

## 🎞️ Movie Posters (Optional TMDB Setup)

MovieLens provides TMDB IDs in `data/links.csv` but does not include poster file paths. CineMatch uses those IDs with the TMDB API to retrieve poster paths and displays the corresponding TMDB image CDN URLs. Poster lookups are cached for 24 hours.

To enable poster images locally, add a `.streamlit/secrets.toml` file:

```toml
TMDB_READ_ACCESS_TOKEN = "your_tmdb_api_read_access_token"
```

Alternatively, use `TMDB_API_KEY` with a TMDB v3 API key. Do not commit `.streamlit/secrets.toml` or expose API credentials; it is ignored by Git. On Streamlit Community Cloud, add the same setting in the app's **Settings → Secrets** panel.

Without a TMDB credential, the app remains usable and displays designed poster placeholders. TMDB attribution is included in the About page. See [TMDB's official FAQ](https://developer.themoviedb.org/docs/faq) for its API and attribution terms.
