# VIVA_PREPARATION.md
# CineMatch – College Project Viva Preparation Guide

> Prepare your 2-minute intro, study the Q&A, and follow the demo sequence below.

---

## A. PROJECT INTRODUCTION (1–2 Minute Script)

> *Speak this naturally in your own words during the demo.*

---

"Good morning / afternoon, sir/ma'am.

My project is called **CineMatch** – an Intelligent Movie Recommendation System built in Python.

The basic problem I wanted to solve is: *how does a computer decide which movie to suggest to you?* Platforms like Netflix and Amazon Prime do this automatically. I wanted to understand and implement the same technique from scratch.

My system uses a technique called **content-based filtering**. It analyses each movie's genres and user-written tags, then represents each movie as a numerical vector using an algorithm called **TF-IDF** — Term Frequency-Inverse Document Frequency. After that, it compares movies using **cosine similarity** — a mathematical measure of how close two vectors are.

The project uses the official **MovieLens dataset** from GroupLens Research at the University of Minnesota, which contains nearly 10,000 movies and 100,000 ratings from real users.

I built the entire interface using **Streamlit**, which lets you create interactive web applications using pure Python — no HTML or JavaScript needed.

Key features include:
- Partial-title movie search
- TF-IDF based recommendations
- Jaccard similarity as a simpler genre-only comparison
- Multi-favourite mode — combine the profiles of multiple liked movies
- Genre filtering
- Watchlist and favourites saved locally using JSON

I'll now demonstrate the application."

---

## B. IMPORTANT VIVA QUESTIONS AND ANSWERS

---

### Section 1 – Python Fundamentals

**Q1. What is a module in Python?**

A module is a `.py` file containing Python code — functions, classes, and variables — that can be imported into other files. In this project, `src/data_loader.py`, `src/recommender.py`, etc. are all modules.

---

**Q2. What is the difference between `import pandas` and `from pandas import DataFrame`?**

`import pandas` imports the whole library. You then write `pandas.DataFrame`. `from pandas import DataFrame` imports only the `DataFrame` class directly. `from src.data_loader import load_movies` is the second style — importing just the needed function.

---

**Q3. What does `@st.cache_data` do in app.py?**

It is a **decorator** that tells Streamlit to cache (remember) the return value of a function. The first time `load_all_data()` is called, it loads and processes the CSV files. After that, calling it again returns the saved result without re-reading the files. This makes the app much faster.

---

**Q4. What is a list comprehension? Give an example from the project.**

A list comprehension creates a list in one compact line:
```python
genres = [g.strip() for g in "Action|Drama|Comedy".split("|")]
# → ["Action", "Drama", "Comedy"]
```
Used in `preprocessing.py` to split and clean genre strings.

---

**Q5. What is a lambda function? Where is it used in this project?**

A lambda is an anonymous (unnamed) one-line function:
```python
scores.sort(key=lambda x: x["score"], reverse=True)
```
Here the lambda says: "sort by the 'score' key inside each dictionary". Used in `recommender.py` to sort similarity results.

---

**Q6. What is exception handling and why is it important here?**

Exception handling catches errors that would otherwise crash the program. In `user_preferences.py`:
```python
try:
    data = json.load(f)
except json.JSONDecodeError:
    return {"favourites": [], "watchlist": []}  # Safe fallback
```
If the preferences file is corrupted, we return safe defaults instead of crashing.

---

### Section 2 – OOP Concepts

**Q7. What class did you create in this project? What is its purpose?**

`MovieRecommender` in `src/recommender.py`. It is a class that holds the TF-IDF matrix, the vocabulary, and the cosine similarity matrix as attributes, and exposes methods like `recommend_by_movie()` and `recommend_by_favourites()`. Using a class ensures the expensive model is built once and reused.

---

**Q8. What is `self` in a Python class?**

`self` refers to the current instance of the class. When you write `self.movies_df = movies_df` inside `__init__`, you are storing the movies DataFrame on this specific object so other methods can access it.

---

**Q9. What is `__init__` and when is it called?**

`__init__` is the constructor — it runs automatically when you create an object from a class. In this project: `recommender = MovieRecommender(movies_df)` triggers `__init__`, which immediately builds the TF-IDF model.

---

**Q10. What is the difference between a class method and a regular function?**

A class method is defined inside a class and receives `self` as the first parameter. A regular function stands alone. `recommend_by_movie(self, movie_id)` is a method; `extract_genres(genres_str)` in `preprocessing.py` is a regular function.

---

### Section 3 – File Handling, CSV, JSON

**Q11. How do you read a CSV file in Python?**

Using pandas: `df = pd.read_csv("data/movies.csv")`. Pandas reads the first line as column names and creates a DataFrame with one row per line.

---

**Q12. How is the JSON file structured in this project?**

```json
{
  "favourites": [1, 42, 100],
  "watchlist":  [5, 23]
}
```
A JSON object (Python dict) with two lists. The movie IDs are integers.

---

**Q13. What does `with open(...) as f` mean?**

It is a **context manager**. It opens the file, runs the indented block, and automatically closes the file even if an error occurs. Safer than manually calling `f.close()`.

---

### Section 4 – Pandas

**Q14. What is a DataFrame?**

A DataFrame is a 2-dimensional labelled data structure — like a table. Rows represent records (movies), columns represent attributes (movieId, title, genres).

---

**Q15. How did you handle missing values in the dataset?**

```python
df = df.dropna(subset=["movieId", "title"])   # Remove rows with missing ID/title
df["genres"] = df["genres"].fillna("")         # Replace missing genres with empty string
df["tags_text"] = df["tags_text"].fillna("")   # Same for aggregated tags
```

---

**Q16. How did you merge the movies and tags DataFrames?**

```python
merged = movies_df.merge(agg_tags, on="movieId", how="left")
```
A `left` join keeps all movies, even those with no tags (they get an empty string instead of NaN).

---

**Q17. How did you aggregate tags per movie?**

```python
agg_tags = (
    tags_df.groupby("movieId")["tag"]
    .apply(lambda tags: " ".join(tags.unique()))
    .reset_index()
)
```
`groupby` groups rows by movieId. `.apply(...)` joins all unique tags into one string per movie.

---

### Section 5 – NumPy

**Q18. What is a NumPy array and how is it different from a Python list?**

A NumPy array is a fixed-type, C-backed array optimised for mathematical operations. `[1, 2, 3] + [4, 5, 6]` in Python concatenates lists. `np.array([1,2,3]) + np.array([4,5,6])` returns `[5, 7, 9]` (element-wise addition). NumPy is much faster for large numerical computations.

---

**Q19. How is NumPy used in the cosine similarity calculation?**

```python
dot = np.dot(vec_a, vec_b)             # Dot product
norm_a = np.linalg.norm(vec_a)         # Magnitude (Euclidean length)
norm_b = np.linalg.norm(vec_b)
result = dot / (norm_a * norm_b)       # Cosine formula
```

---

### Section 6 – Scikit-learn and TF-IDF

**Q20. What does TF stand for in TF-IDF?**

**Term Frequency** – how many times a particular word appears in a specific movie's profile compared to the total words in that profile.

---

**Q21. What does IDF stand for?**

**Inverse Document Frequency** – a measure of how rare a word is across ALL movie profiles.

```
IDF(word) = log(total_movies / movies_containing_word)
```

A word appearing in every movie has IDF ≈ 0. A rare word has high IDF.

---

**Q22. Why do we use TF-IDF instead of simply counting words?**

Simple word counts treat all words equally. The word "Drama" appears in 4,000 movies — it doesn't distinguish Interstellar from a romance film. TF-IDF automatically down-weights common words and up-weights distinctive ones (like "heist" or "dystopia"), producing more meaningful comparisons.

---

**Q23. What is a TF-IDF matrix?**

A 2D matrix where:
- Each **row** represents one movie
- Each **column** represents one unique word in the vocabulary
- Each **cell** contains the TF-IDF weight of that word for that movie

If the vocabulary has 5,000 words and there are 9,742 movies, the matrix is 9,742 × 5,000.

---

**Q24. What parameters did you use for TfidfVectorizer and why?**

```python
TfidfVectorizer(
    min_df=2,          # Ignore words in fewer than 2 movies (too unique/noisy)
    max_df=0.8,        # Ignore words in more than 80% of movies (too generic)
    stop_words="english",  # Remove words like "the", "is", "a"
    ngram_range=(1, 2),    # Consider single words AND two-word phrases
)
```

---

**Q25. What is `fit_transform()` in scikit-learn?**

`fit()` learns the vocabulary and IDF values from the data. `transform()` converts the data into the TF-IDF matrix using those learned values. `fit_transform()` does both in one step.

---

### Section 7 – Cosine Similarity

**Q26. What is cosine similarity?**

Cosine similarity measures the **angle** between two vectors. If the angle is small (vectors point in the same direction), the similarity is close to 1.0. If they are perpendicular (no shared features), similarity is 0.0.

Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`

---

**Q27. Why do we use cosine similarity instead of Euclidean distance?**

Euclidean distance is sensitive to the **magnitude** (length) of vectors. A movie with many tags has longer vectors. Cosine similarity ignores magnitude — it only cares about **direction** (the pattern of features), making it fairer for comparing movies of different profile lengths.

---

**Q28. What does a cosine similarity score of 0.85 mean?**

The two movie profiles share strong common features — similar genres and/or tags. It does NOT mean there is an 85% probability that you will like the recommended movie. It is purely a content-matching score.

---

### Section 8 – Jaccard Similarity

**Q29. What is Jaccard similarity?**

A simple set-based similarity measure:
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Intersection divided by union. Used in this project to compare genre sets.

---

**Q30. When would Jaccard give 1.0?**

When both movies share exactly the same set of genres (identical sets). The intersection equals the union.

---

**Q31. What is the difference between Jaccard and TF-IDF cosine similarity?**

| Aspect | Jaccard | TF-IDF + Cosine |
|--------|---------|-----------------|
| Data used | Genre sets only | Genres + user tags |
| Weighing | Each genre equal weight | Rare terms get higher weight |
| Complexity | Simple and fast | More computation |
| Precision | Coarser (genres are broad) | More precise (tags are specific) |

---

### Section 9 – Recommendation Systems

**Q32. What is a recommendation system?**

A system that predicts items a user might like, based on data. Used by Netflix, YouTube, Spotify, Amazon.

---

**Q33. What is content-based filtering?**

A recommendation approach that recommends items similar to what a user has liked, based on item *features* (in this case: genres and tags). It doesn't need other users' data.

---

**Q34. What is collaborative filtering?**

It recommends items based on the preferences of *similar users*. "Users who liked Interstellar also liked Arrival." This project includes a basic item-based collaborative filtering mode using the MovieLens ratings matrix.

---

**Q35. What is the difference between content-based and collaborative filtering?**

| | Content-Based | Collaborative |
|-|--------------|--------------|
| Data used | Item features (genres, tags) | User ratings |
| Cold start | Works for new items | Fails for new users/items with no ratings |
| Personalisation | Based on item similarity | Based on user behaviour patterns |

---

**Q36. What is the cold-start problem?**

When a movie has no tags and very generic genres, its TF-IDF profile is weak. The system can only use the genre label, which may produce less precise recommendations. This is called the "cold-start" problem.

---

**Q37. What is the MovieLens dataset?**

A dataset published by GroupLens Research at the University of Minnesota, containing real anonymised movie ratings collected from real users. The "latest small" version has ~9,700 movies and ~100,000 ratings.

---

### Section 10 – Project Architecture

**Q38. Why did you separate the code into multiple files instead of one?**

Separation of concerns. Each file has a single responsibility:
- `data_loader.py` only loads files
- `preprocessing.py` only cleans and transforms data
- `recommender.py` only handles recommendation logic
- `app.py` only handles the interface

This makes the code easier to test, understand, and modify.

---

**Q39. What does `pathlib.Path` do and why use it?**

`Path` creates file paths that work correctly on Windows, macOS, and Linux automatically. `Path(__file__).parent / "data" / "movies.csv"` builds the correct path regardless of the operating system, unlike hardcoded strings like `"C:\\Users\\...\\movies.csv"`.

---

**Q40. What testing framework did you use and what did you test?**

**Pytest**. I tested:
- Jaccard similarity for identical, disjoint, partial sets
- Cosine similarity for orthogonal and identical vectors
- Genre extraction for valid, empty, and invalid inputs
- Data loading with valid and missing files
- Recommendations excluding the query movie
- Multi-favourite mode
- Watchlist/favourites save and load
- Corrupt JSON file handling

---

### Section 11 – Limitations and Improvements

**Q41. What are the limitations of your project?**

1. Cold-start: movies with no tags have weak profiles
2. Similarity score ≠ quality or personal preference
3. No real user authentication
4. MovieLens has ~9,700 movies; newer titles may be missing
5. Genre labels are coarse (many different "Action" movies)

---

**Q42. How would you improve the recommendation accuracy?**

- Integrate TMDB API for plot summaries → richer textual profiles
- Add matrix factorisation (SVD) for better collaborative filtering
- Use BERT or sentence embeddings for semantic similarity
- Implement hybrid filtering (content + collaborative combined)
- Add user feedback loops (like/dislike)

---

**Q43. What happens when two movies have the same title?**

The system uses `movieId` (not title) as the unique identifier. Even if two movies have the same title, they have different IDs. The search results display the year in brackets to help users distinguish them. The user selects by ID, not by title string.

---

**Q44. What happens when a user searches for a movie that doesn't exist?**

`search_movies()` returns an empty DataFrame. The `app.py` checks:
```python
if results.empty:
    st.warning(f'No movies found for "{query}".')
```
The application shows a helpful message and does not crash.

---

## C. PROJECT DEMONSTRATION SEQUENCE

---

### Step 1 – Open the terminal

```powershell
cd "C:\Users\HP\Downloads\AATP\Movie-Recommendation-System"
.venv\Scripts\Activate.ps1
streamlit run app.py
```

Say: *"This command starts the Streamlit server and opens the application in the browser."*

---

### Step 2 – Home Page

Point out:
- The CineMatch banner and dark theme
- The four statistics cards (movies, ratings, tagged movies, TF-IDF features)
- The info box explaining the algorithm

Say: *"The home page shows an overview of the dataset and a quick search."*

---

### Step 3 – Quick Search on Home Page

Type: `inter`

Show: *"Even a partial title 'inter' finds Interstellar."*

---

### Step 4 – Navigate to Discover

Click **Discover** in the sidebar.

Type: `Interstellar`

Select: **Interstellar (2014)** from the dropdown.

Show the movie details card:
- Year, genres, average rating, number of ratings, tags, IMDb link

Say: *"These details come from merging movies.csv, ratings.csv, tags.csv, and links.csv."*

---

### Step 5 – TF-IDF Recommendations

Make sure **TF-IDF + Cosine Similarity** is selected.

Say: *"The system looks up Interstellar's row in the cosine similarity matrix and returns the 10 most similar movies."*

Scroll through the recommendations. Point out:
- Gravity, The Martian, Arrival, etc. (Sci-Fi films)
- The similarity percentage badge
- The "key features" pills showing which TF-IDF terms they share

---

### Step 6 – Jaccard Comparison

Switch to **Jaccard Similarity (Genre-based)**.

Say: *"Jaccard uses only genre sets. The formula is: intersection divided by union. This is simpler — it doesn't use tags at all."*

Compare results with TF-IDF mode. TF-IDF is usually more precise.

---

### Step 7 – Add to Watchlist / Favourites

Click **Add to Watchlist** for Interstellar.

Say: *"The movie ID is saved to user_data/preferences.json locally."*

---

### Step 8 – Genre Explorer

Navigate to **Genres**.

Select genres: `Sci-Fi` and `Thriller`.

Toggle **Match ALL** to show strict filtering.

Say: *"This filters movies by genre using list comprehensions on the genres_list column."*

---

### Step 9 – Multi-Favourite Recommendations

Navigate to **For You**.

Add: Interstellar, The Dark Knight, Inception.

Click **Generate Recommendations**.

Say: *"The system takes the TF-IDF vectors of all three movies and averages them. The combined vector represents the user's overall taste. Movies closest to this combined vector are recommended."*

---

### Step 10 – My List

Navigate to **My List**.

Show the Watchlist tab with Interstellar added earlier.

Say: *"Remove works by filtering the list in the JSON file and writing it back."*

---

### Step 11 – About Page

Navigate to **About**.

Show the algorithm explanation table and architecture diagram.

Say: *"This page summarises how the project works at a high level."*

---

### Step 12 – Show the Code

Open `src/recommender.py` briefly.

Point to:
- `__init__` and `_build_tfidf_model`
- The `recommend_by_movie` method
- The `recommend_by_favourites` method

---

### Step 13 – Run the Tests

```powershell
pytest tests/ -v
```

Say: *"These are automated tests using pytest. They verify that the similarity formulas, data loading, recommendation ranking, and user preferences all work correctly."*

---

### Conclusion (30 seconds)

*"To summarise:*
- *CineMatch loads real movie data from MovieLens.*
- *It builds a TF-IDF vector for each movie from its genres and tags.*
- *It computes cosine similarity between all pairs.*
- *When you search a movie, it looks up the pre-computed similarity row and returns the top results.*
- *It also supports Jaccard similarity for genre-only comparison, multi-favourite combined profiles, and collaborative filtering.*
- *Everything is tested with pytest and the interface is built with Streamlit.*

*Thank you."*

---

## D. DIFFICULT QUESTIONS FROM THE TEACHER

---

**D1. Why did you choose content-based filtering over collaborative filtering?**

Content-based filtering works without needing information about other users. It only needs the movie's own features (genres, tags). This avoids the cold-start problem for new users. Collaborative filtering requires at least some rating history. Since this application doesn't have real user accounts, content-based is more appropriate. I did implement collaborative filtering as an optional bonus mode.

---

**D2. Why did you use TF-IDF instead of simply counting words?**

Simple word counts (Bag of Words) treat all words equally. The word "Drama" appears in thousands of profiles and tells us nothing specific. TF-IDF assigns lower weight to common words and higher weight to rare, distinctive words. This makes the similarity scores more meaningful — movies with rare shared tags are ranked higher than movies that merely share common genre labels.

---

**D3. What happens if two movie titles are identical?**

The system uses `movieId` as the unique key, not the title string. The search function returns both movies, and each is shown with its year (from the title) and unique ID. The user selects the intended one. The TF-IDF and similarity matrices are indexed by DataFrame position, which maps to `movieId`, not to title.

---

**D4. What is the difference between cosine similarity and Jaccard similarity?**

| Aspect | Cosine Similarity | Jaccard Similarity |
|--------|------------------|--------------------|
| Input | Numerical vectors | Sets |
| Data used | TF-IDF of genres + tags | Genre sets only |
| Considers frequency | Yes (via TF weights) | No |
| Considers rarity | Yes (via IDF) | No |
| Complexity | O(n×d) vectorised | O(n×g) per comparison |

Cosine is more precise because it uses richer information. Jaccard is simpler and interpretable.

---

**D5. What are the limitations of your algorithm?**

1. **Cold-start**: Movies with no tags have weak profiles based on genres only.
2. **Genre coarseness**: Two "Action" movies (e.g. Die Hard and Avengers) may be very different.
3. **Similarity ≠ preference**: A score of 0.9 means similar content, not "you will love it."
4. **Static dataset**: MovieLens doesn't include very recent movies.
5. **Tag sparsity**: Only 3,683 tags for 9,742 movies — many have none.
6. **No real personalisation**: Without user accounts, we can't track long-term preferences.

---

**D6. How would you improve recommendation accuracy?**

Short-term improvements:
- Fetch plot summaries from TMDB API → much richer text profiles
- Use hybrid filtering (average content and collaborative scores)

Long-term improvements:
- Use sentence transformers (BERT-based) for semantic understanding
- Implement matrix factorisation (SVD or ALS) for collaborative filtering
- Collect explicit user feedback and retrain the model

---

**D7. What is the difference between `fit()` and `transform()` in scikit-learn?**

`fit()` learns from the data: for TfidfVectorizer, it builds the vocabulary and computes IDF values for each word. `transform()` applies that learned transformation to convert new text into vectors. `fit_transform()` does both together on the same dataset, which is more efficient.

We NEVER use `fit()` on the test/recommendation data — only `transform()`. (In this project, all movies are transformed at once, so we always use `fit_transform`.)

---

**D8. Why did you use `@st.cache_resource` for the recommender but `@st.cache_data` for the data?**

Streamlit recommends `@st.cache_resource` for objects that hold shared state (like ML models, database connections) because they should not be copied between threads. `@st.cache_data` is for serialisable data (DataFrames, lists, dicts) which can be safely copied. The `MovieRecommender` holds a large numpy array and sparse matrix, so it uses `@st.cache_resource`.

---

**D9. If a user selects 5 favourite movies, how exactly does the combined vector work?**

Each favourite movie has a TF-IDF row vector of shape (1, N_features). We stack the 5 vectors into a matrix of shape (5, N_features) and compute the **mean** across rows, giving a single (1, N_features) vector. This vector represents the "average taste" in the TF-IDF feature space. Movies closest (highest cosine similarity) to this combined vector are the recommendations.

---

**D10. Can the similarity score ever be negative?**

With TF-IDF vectors — no, because all TF-IDF values are non-negative (counts and log-based scores are always ≥ 0). The dot product of two non-negative vectors is always ≥ 0, and their magnitudes are positive, so cosine similarity stays in [0.0, 1.0]. Negative cosine similarity is only possible with vectors that have negative components (not the case here).
