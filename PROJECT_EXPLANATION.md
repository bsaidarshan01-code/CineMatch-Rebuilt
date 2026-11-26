# PROJECT_EXPLANATION.md
# CineMatch – How the Project Works (Beginner-Friendly Guide)

> This document teaches you how the project works from the ground up.
> Read it alongside the source code to understand every part confidently.

---

## PART A – Python Concepts Used in This Project

---

### A1. Imports and Modules

A **module** is simply a `.py` file containing Python code (functions, classes, variables).
An **import** brings code from one module into another.

```python
import pandas as pd       # Import the pandas library, give it the short name "pd"
from pathlib import Path  # Import just the Path class from the pathlib module
```

In this project:
- `src/data_loader.py` is a module that handles loading CSV files.
- `src/recommender.py` is a module for the recommendation engine.
- `app.py` imports from both: `from src.data_loader import load_movies`

**Why modules?** They keep code organised. Instead of one giant 2000-line file,
each file has a single clear responsibility (the "single responsibility" principle).

---

### A2. Functions

A function is a reusable block of code that takes inputs (parameters) and returns an output.

```python
def extract_genres(genres_str: str) -> list:
    """Split 'Action|Drama|Comedy' into ['Action', 'Drama', 'Comedy']."""
    if not isinstance(genres_str, str):
        return []
    return [g.strip() for g in genres_str.split("|") if g.strip()]
```

- `def` – keyword that starts a function definition
- `genres_str: str` – parameter with a **type hint** (str = string)
- `-> list` – the function promises to return a list
- `"""..."""` – a **docstring** that documents what the function does

---

### A3. Classes and Objects

A **class** is a blueprint for creating objects with shared attributes and methods.

```python
class MovieRecommender:
    def __init__(self, movies_df):   # Constructor - runs when you create an object
        self.movies_df = movies_df   # self.xxx stores data on this specific object
        self._build_tfidf_model()    # Immediately build the model on creation

    def recommend_by_movie(self, movie_id, top_n=10):
        # ... recommendation logic ...
        pass
```

**Creating an object (instance) from the class:**
```python
rec = MovieRecommender(movies_df)   # This calls __init__ automatically
recs = rec.recommend_by_movie(1)    # Call a method on the object
```

**Why use a class here?** The TF-IDF model (tfidf_matrix, cosine_sim) is built once
and reused many times. Storing it in an object (self.tfidf_matrix) avoids rebuilding
it on every recommendation call.

---

### A4. File Handling

```python
# Writing to a JSON file
import json

with open("preferences.json", "w", encoding="utf-8") as f:
    json.dump({"favourites": [1, 2, 3]}, f, indent=2)

# Reading from a JSON file
with open("preferences.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

The `with` statement automatically closes the file even if an error occurs.
This is called a **context manager**.

---

### A5. CSV Files and JSON

**CSV (Comma-Separated Values)** – A plain text format for tabular data:
```
movieId,title,genres
1,Toy Story (1995),Animation|Comedy|Family
2,Jumanji (1995),Adventure|Children|Fantasy
```

**JSON (JavaScript Object Notation)** – A format for storing structured data:
```json
{
  "favourites": [1, 42, 100],
  "watchlist":  [5, 23]
}
```

We use CSV for the MovieLens dataset (loaded with Pandas) and JSON for user preferences.

---

### A6. Exception Handling

```python
try:
    df = pd.read_csv("movies.csv")          # Code that might fail
except FileNotFoundError as exc:
    print(f"File not found: {exc}")          # Handle the specific error
except Exception as exc:
    print(f"Unexpected error: {exc}")        # Catch anything else
```

Without exception handling, the whole program crashes. With it, we show a helpful
message and let the application continue gracefully.

---

### A7. List Comprehensions

A compact way to create lists:

```python
# Traditional for loop
genres = []
for g in "Action|Drama|Comedy".split("|"):
    genres.append(g.strip())

# List comprehension (same result, one line)
genres = [g.strip() for g in "Action|Drama|Comedy".split("|")]

# With a condition (filter)
non_empty = [g for g in genres if g != ""]
```

Used heavily in this project for genre processing and similarity calculations.

---

### A8. Dictionary Comprehensions

```python
# Map movieId to its DataFrame row index for fast lookup
id_to_idx = {int(mid): idx for idx, mid in enumerate(movies_df["movieId"])}

# Equivalent for loop:
id_to_idx = {}
for idx, mid in enumerate(movies_df["movieId"]):
    id_to_idx[int(mid)] = idx
```

---

### A9. Lambda Functions

A **lambda** is a one-line anonymous function:

```python
# Regular function
def get_score(x):
    return x["score"]

# Lambda equivalent
get_score = lambda x: x["score"]

# Used in sort:
scores.sort(key=lambda x: x["score"], reverse=True)
```

Used in `recommender.py` to sort similarity scores by value without defining
a named function.

---

## PART B – External Libraries

---

### B1. What is a Library?

A **library** (or package) is a collection of Python files written by others
that you can install and reuse. Instead of writing your own CSV reader or
machine learning algorithms, you install a library that already does it.

Install libraries with pip:
```
pip install pandas numpy scikit-learn streamlit
```

---

### B2. Pandas

Pandas provides the **DataFrame** – a table of data similar to an Excel spreadsheet.

```python
import pandas as pd

# Load a CSV file into a DataFrame
df = pd.read_csv("movies.csv")

# View first 5 rows
df.head()

# Select a column (returns a Series)
titles = df["title"]

# Filter rows where genre contains "Action"
action_movies = df[df["genres"].str.contains("Action")]

# Group by movieId and aggregate tags
agg = df.groupby("movieId")["tag"].apply(" ".join)
```

**In this project:** Pandas loads movies.csv, ratings.csv, tags.csv, links.csv
and merges them into a single clean DataFrame.

---

### B3. NumPy

NumPy provides fast **arrays** and mathematical operations.

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

dot_product = np.dot(a, b)         # a · b = 1*4 + 2*5 + 3*6 = 32
magnitude_a = np.linalg.norm(a)    # ||a|| = sqrt(1²+2²+3²) = 3.74
```

**In this project:** NumPy is used in `similarity.py` to calculate cosine similarity
manually, and by scikit-learn internally for all matrix operations.

---

### B4. Scikit-learn

Scikit-learn provides ready-made machine learning tools.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF: convert text profiles into number matrices
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(["Action Drama", "Sci-Fi Thriller"])
# tfidf_matrix is now a 2×N matrix of floats

# Cosine similarity between all pairs
sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
# sim_matrix[i][j] = similarity between movie i and movie j
```

**In this project:** `MovieRecommender._build_tfidf_model()` uses these two classes
to build the full recommendation system.

---

### B5. Streamlit

Streamlit turns Python scripts into interactive web applications with no HTML/CSS/JavaScript needed.

```python
import streamlit as st

st.title("Hello World")
st.text_input("Search:", placeholder="Type here…")
st.button("Click me!")
```

**In this project:** `app.py` is a pure Streamlit file that creates the entire
dark-themed interface — pages, search boxes, movie cards, navigation sidebar.

---

## PART C – Recommendation Algorithm (Step by Step)

---

### Step 1 – Loading the Dataset

```python
movies_df = pd.read_csv("data/movies.csv")
# movieId | title                | genres
# 1       | Toy Story (1995)     | Animation|Comedy|Family
# 2       | Jumanji (1995)       | Adventure|Children|Fantasy
```

We also load `tags.csv` which contains user-written words describing movies:
```
1 | 1 | fun | 964982703
1 | 1 | pixar | 964982704
```

---

### Step 2 – Combining Movie Information

Each movie gets a **textual profile** by merging its genres and tags.

```python
genres_tokens = "Animation Comedy Family"       # from genres column
tags_text      = "fun pixar family entertaining"   # from tags.csv

profile = genres_tokens + " " + tags_text
# → "Animation Comedy Family fun pixar family entertaining"
```

Movies without tags still get a profile from their genres alone.

---

### Step 3 – TF-IDF Vectorisation

**What is TF-IDF?**

Consider the word "Drama". It appears in thousands of movie profiles.
Now consider the word "heist". Very few movies have that tag.

TF-IDF gives each word a weight:
- **TF (Term Frequency)** = how many times the word appears in THIS movie's profile
- **IDF (Inverse Document Frequency)** = log(total movies / movies containing this word)
- **Final weight = TF × IDF**

| Word    | Appears in | IDF (approx) | Weight |
|---------|-----------|---------------|--------|
| "Drama" | 4,000 movies | low ≈ 0.9  | low    |
| "heist" | 50 movies    | high ≈ 5.3 | high   |

**Result:** A numerical vector for each movie. Each position in the vector
represents one word from the vocabulary.

```
"Animation Comedy" → [0.0, 0.71, 0.0, 0.0, 0.70, 0.0, ...]
"Action Thriller"  → [0.80, 0.0, 0.0, 0.70, 0.0, 0.0, ...]
```

The vocabulary has thousands of unique words. Each movie becomes a point
in that high-dimensional space.

---

### Step 4 – Cosine Similarity

After TF-IDF, every movie is a vector. How do we compare them?

**Cosine Similarity** measures the angle between two vectors:

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` = dot product (sum of element-wise products)
- `||A||` = magnitude of vector A (sqrt of sum of squares)

**Example:**
```
A = [1, 0, 1]   (Action, Comedy features)
B = [1, 1, 0]   (Action, Drama features)

A · B = 1×1 + 0×1 + 1×0 = 1
||A|| = sqrt(1+0+1) = 1.414
||B|| = sqrt(1+1+0) = 1.414

cos(θ) = 1 / (1.414 × 1.414) = 1/2 = 0.5
```

Score = 0.5 means "moderately similar".
Score = 1.0 means "identical profiles".
Score = 0.0 means "no shared features".

The score is **NOT a probability that you will like it**. It is purely
a measure of content overlap.

---

### Step 5 – Building the Similarity Matrix

Scikit-learn computes cosine similarity for ALL pairs at once:

```python
# cosine_sim is an (n_movies × n_movies) matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# cosine_sim[0][1] = similarity between movie 0 and movie 1
# cosine_sim[5][100] = similarity between movie 5 and movie 100
```

For 9,742 movies this is a 9742×9742 matrix — computed once during startup
and cached in memory.

---

### Step 6 – Generating Recommendations

```python
# User selected: Interstellar (movieId=5618, index=200)
idx = 200

# Get similarity scores of Interstellar vs ALL movies
sim_scores = list(enumerate(cosine_sim[idx]))
# → [(0, 0.12), (1, 0.34), (2, 0.67), (3, 0.08), ...]

# Remove the movie itself
sim_scores = [(i, s) for i, s in sim_scores if i != idx]

# Sort by similarity score (highest first)
sim_scores.sort(key=lambda x: x[1], reverse=True)

# Take top 10
top_10 = sim_scores[:10]
```

---

### Step 7 – Multi-Favourite Recommendations

When a user selects multiple favourite movies, we average their TF-IDF vectors:

```python
fav_indices = [200, 350, 1200]           # indices of favourite movies
fav_vectors = tfidf_matrix[fav_indices]  # select their rows
combined = fav_vectors.mean(axis=0)      # average → 1 combined vector

# Find movies most similar to this combined taste profile
sim_scores = cosine_similarity(combined, tfidf_matrix)
```

**Why average?** The average vector lands "between" all favourites in the
feature space, representing the shared taste across all selected movies.

---

### Step 8 – Jaccard Similarity (Genre-based)

A simpler approach using Python sets:

```python
# Movie A: Interstellar  → genres = {"Sci-Fi", "Adventure", "Drama"}
# Movie B: Gravity       → genres = {"Sci-Fi", "Drama", "Thriller"}

intersection = {"Sci-Fi"} ∩ {"Sci-Fi", "Adventure", "Drama"} = {"Sci-Fi", "Drama"}
union        = {"Sci-Fi", "Adventure", "Drama", "Thriller"}

J(A, B) = 2 / 4 = 0.5
```

Jaccard only uses genres. TF-IDF uses genres + tags, making it more precise.

---

### Step 9 – Displaying Results

Streamlit renders each recommended movie as a styled HTML card:
```
#1 Gravity (2013)
Sci-Fi | Drama | Thriller
Key features: sci fi, space, drama
Similarity: 85%  (algorithmic content match — not a rating prediction)
```

---

## PART D – Code Walkthrough

---

### File: src/data_loader.py

**Purpose:** Load and validate the four MovieLens CSV files.

**Key functions:**

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `dataset_status()` | None | dict | Check which CSVs exist |
| `all_files_present()` | None | bool | True if all 4 CSVs present |
| `load_movies()` | None | DataFrame | movies.csv → clean DataFrame |
| `load_ratings()` | None | DataFrame | ratings.csv → clean DataFrame |
| `load_tags()` | None | DataFrame | tags.csv → clean DataFrame |
| `load_links()` | None | DataFrame | links.csv → clean DataFrame |
| `_validate_columns()` | df, expected, filename | None | Raise ValueError if columns missing |

**How movies are loaded:**
```python
df = pd.read_csv("data/movies.csv")          # 1. Read CSV into DataFrame
df = df.dropna(subset=["movieId", "title"])  # 2. Drop rows with missing IDs
df["movieId"] = df["movieId"].astype(int)    # 3. Ensure correct data type
df["genres"] = df["genres"].fillna("")       # 4. Replace missing genres with ""
```

---

### File: src/preprocessing.py

**Purpose:** Clean movie data and build a combined textual profile per movie.

**Key functions:**

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `extract_genres()` | "Action\|Drama" | ["Action","Drama"] | Split genres string |
| `genres_to_set()` | "Action\|Drama" | {"action","drama"} | For Jaccard similarity |
| `aggregate_tags()` | tags_df | DataFrame | Group tags per movie |
| `build_movie_profiles()` | movies_df, tags_df | enriched movies_df | Full pipeline |
| `_extract_year()` | "Toy Story (1995)" | 1995 | Parse year from title |
| `clean_title()` | "Toy Story (1995)" | "Toy Story" | Remove year suffix |

---

### File: src/similarity.py

**Purpose:** Pure mathematical similarity functions.

```python
# Jaccard: 2 sets → 1 float
def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

# Cosine: 2 numpy vectors → 1 float
def cosine_similarity_pair(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

These are **pure functions** – they have no side effects and always return
the same output for the same input.

---

### File: src/recommender.py – MovieRecommender class

**Purpose:** The core recommendation engine.

```python
class MovieRecommender:
    def __init__(self, movies_df):
        # Build TF-IDF model immediately
        self.movies_df = movies_df
        self._build_tfidf_model()

    def _build_tfidf_model(self):
        # 1. Extract all movie profiles
        profiles = self.movies_df["profile"].fillna("").tolist()
        # 2. Fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)
        self.tfidf_matrix = self.vectorizer.fit_transform(profiles)
        # 3. Compute full cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend_by_movie(self, movie_id, top_n=10):
        # Look up the DataFrame index for this movie ID
        idx = self._id_to_idx[movie_id]
        # Get similarity scores vs all movies
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        # Remove self, sort, take top N
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        return self._format_results(sim_scores[:top_n])
```

---

### File: src/user_preferences.py

**Purpose:** Save/load user data as JSON.

```python
# preferences.json looks like:
{
    "favourites": [1, 42, 100],
    "watchlist":  [5, 23]
}

# How to safely load with error handling:
def load_preferences():
    if not PREFS_PATH.exists():
        return {"favourites": [], "watchlist": []}
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"favourites": [], "watchlist": []}  # corrupted file → safe default
```

---

### File: app.py

**Purpose:** Streamlit interface connecting all modules.

```python
# Cache expensive operations so they run once per session
@st.cache_data
def load_all_data():
    movies_df = load_movies()
    # ... load others ...
    return build_movie_profiles(movies_df, tags_df), ...

@st.cache_resource
def build_recommender(movies_df):
    return MovieRecommender(movies_df)
```

The `@st.cache_data` and `@st.cache_resource` decorators are Streamlit's way
of memoising (remembering) results so loading 9,742 movies only happens once.

---

## PART E – Data Flow

```
User Input (movie title typed in search bar)
        │
        ▼
  Movie Search (search_movies function)
  - Case-insensitive partial match
  - Returns matching rows from movies_df
        │
        ▼
  Movie Selection (user picks from dropdown)
  - movieId retrieved from selected label
        │
        ▼
  Movie Data Retrieval (get_movie_details)
  - Genres, tags, avg rating, IMDb ID
        │
        ▼
  Feature Extraction (already done at startup)
  - Each movie has a profile: "SciFi Drama space exploration"
        │
        ▼
  TF-IDF Vectorisation (pre-computed at startup)
  - Profile → numerical vector
  - e.g. [0.0, 0.71, 0.0, 0.52, ...]
        │
        ▼
  Cosine Similarity (pre-computed at startup)
  - cosine_sim[idx] = row of scores vs all movies
        │
        ▼
  Ranking
  - Sort by score descending
  - Remove query movie
  - Take top N
        │
        ▼
  Top Recommendations (list of dicts)
  - movieId, title, year, genres_list, score, overlap_features
        │
        ▼
  Streamlit Display
  - Render movie cards with score badges
  - Show top TF-IDF features as "key feature" pills
  - Provide watchlist / favourite buttons
```

**At each step, what happens:**

| Step | What happens | Python concept |
|------|-------------|----------------|
| User Input | String from st.text_input | Streamlit widget |
| Movie Search | String contains check on DataFrame | Pandas .str.contains |
| Movie Selection | Dict lookup: label → movieId | Dictionary |
| Feature Extraction | Genres split + tags joined | String methods, list comprehension |
| TF-IDF | Text → sparse matrix | Scikit-learn TfidfVectorizer |
| Cosine Similarity | Matrix multiplication | Scikit-learn cosine_similarity |
| Ranking | Sort list of tuples | list.sort(key=lambda) |
| Display | HTML cards | Streamlit st.markdown |

---

## Summary

You have now seen:
- How **modular Python** separates concerns into clean files
- How **Pandas DataFrames** represent and manipulate tabular data
- How **TF-IDF** turns text into numbers
- How **cosine similarity** measures vector closeness
- How **Jaccard similarity** compares sets
- How **Streamlit** creates the web interface
- How **JSON** stores user data persistently
- How **classes** keep the recommendation model in memory between calls
- How **exception handling** prevents crashes on bad input
- How **pytest** automatically tests all these components
