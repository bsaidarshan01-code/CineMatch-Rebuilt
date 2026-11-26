import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_movies, load_ratings, load_tags, load_links
from src.preprocessing import build_movie_profiles
from src.recommender import MovieRecommender

print("1. Loading raw dataset files...")
movies = load_movies()
ratings = load_ratings()
tags = load_tags()
links = load_links()
print(f"   Loaded: {len(movies)} movies, {len(ratings)} ratings, {len(tags)} tags, {len(links)} links")

print("\n2. Building movie profiles (genres + tags)...")
profiles = build_movie_profiles(movies, tags)
print(f"   Built profiles for {len(profiles)} movies.")

print("\n3. Initializing TF-IDF & Cosine Similarity Recommender...")
rec = MovieRecommender(profiles)
print(f"   TF-IDF Vocabulary size: {rec.get_vocab_size()} terms")

print("\n4. Testing recommendation for 'Toy Story (1995)' (movieId=1):")
toy_story_recs = rec.recommend_by_movie(1, top_n=5)
for i, r in enumerate(toy_story_recs, 1):
    print(f"   {i}. {r['title']} ({r['year']}) - Similarity: {r['score']*100:.1f}%")

print("\n5. Testing Multi-Favourite recommendation (Toy Story + Star Wars):")
fav_recs = rec.recommend_by_favourites([1, 260], top_n=5)
for i, r in enumerate(fav_recs, 1):
    print(f"   {i}. {r['title']} ({r['year']}) - Similarity: {r['score']*100:.1f}%")

print("\n6. Testing Jaccard Genre-based recommendation for 'Toy Story (1995)':")
jacc_recs = rec.recommend_by_jaccard(1, top_n=5)
for i, r in enumerate(jacc_recs, 1):
    print(f"   {i}. {r['title']} ({r['year']}) - Score: {r['score']:.4f}")

print("\n=======================================================")
print(" ALL RECOMMENDATION ENGINES FUNCTIONING WITH REAL DATA!")
print("=======================================================")
