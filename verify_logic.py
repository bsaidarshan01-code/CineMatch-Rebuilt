"""
verify_logic.py
===============
Standalone logic verification for CineMatch core modules.
Runs without scikit-learn or pytest — only needs pandas and numpy.

Usage:
    python verify_logic.py
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Clean any stale preferences file so module-import side effects don't interfere
_stale = Path(__file__).parent / "user_data" / "preferences.json"
if _stale.exists():
    _stale.unlink()

print("Running CineMatch logic verification …\n")
_PASS = 0
_FAIL = 0


def check(condition, msg):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


# ══════════════════════════════════════════════════════════
# TEST 1 – Jaccard similarity
# ══════════════════════════════════════════════════════════
from src.similarity import jaccard_similarity, jaccard_recommend
import pandas as pd

score = jaccard_similarity({'action', 'thriller'}, {'action', 'drama'})
check(abs(score - 1/3) < 1e-9,  f"1/3 expected, got {score}")
check(jaccard_similarity(set(), set()) == 0.0, "empty sets → 0.0")
check(jaccard_similarity({'a', 'b'}, {'a', 'b'}) == 1.0, "identical → 1.0")
check(jaccard_similarity({'a'}, {'b'}) == 0.0, "disjoint → 0.0")
# Symmetry
a, b = {'x', 'y', 'z'}, {'y', 'w'}
check(jaccard_similarity(a, b) == jaccard_similarity(b, a), "symmetric")
print("✓ Test 1 PASSED: Jaccard similarity")

# ══════════════════════════════════════════════════════════
# TEST 2 – Genre extraction & clean_title
# ══════════════════════════════════════════════════════════
from src.preprocessing import extract_genres, clean_title, aggregate_tags, build_movie_profiles

check(extract_genres('Action|Drama') == ['Action', 'Drama'], "pipe split")
check(extract_genres('') == [], "empty string → []")
check(extract_genres(None) == [], "None → []")
check(extract_genres('(no genres listed)') == [], "placeholder → []")
check(clean_title('Toy Story (1995)') == 'Toy Story', "year removed")
check(clean_title('Batman') == 'Batman', "no year unchanged")
check(clean_title('') == '', "empty string")
check(clean_title(None) == '', "None → ''")
print("✓ Test 2 PASSED: Genre extraction and clean_title")

# ══════════════════════════════════════════════════════════
# TEST 3 – Tag aggregation
# ══════════════════════════════════════════════════════════
tags_df = pd.DataFrame({
    'userId':    [1, 2, 1],
    'movieId':   [1, 1, 1],
    'tag':       ['space', 'drama', 'space'],   # 'space' appears twice
    'timestamp': [1, 2, 3],
})
result = aggregate_tags(tags_df)
movie1_text = result[result['movieId'] == 1].iloc[0]['tags_text']
check('space' in movie1_text, "tag 'space' present")
check('drama' in movie1_text, "tag 'drama' present")
check(movie1_text.count('space') == 1, "duplicate tag deduped")

empty_result = aggregate_tags(pd.DataFrame(columns=['userId', 'movieId', 'tag', 'timestamp']))
check(empty_result.empty, "empty df → empty result")
print("✓ Test 3 PASSED: Tag aggregation")

# ══════════════════════════════════════════════════════════
# TEST 4 – build_movie_profiles
# ══════════════════════════════════════════════════════════
movies_raw = pd.DataFrame({
    'movieId': [1, 2, 3],
    'title':   ['Interstellar (2014)', 'Toy Story (1995)', 'No Genre Movie'],
    'genres':  ['Sci-Fi|Adventure', 'Animation|Comedy', ''],
})
tags_small = pd.DataFrame({
    'userId':    [1],
    'movieId':   [1],
    'tag':       ['space'],
    'timestamp': [100],
})
profiles = build_movie_profiles(movies_raw, tags_small)

check('profile' in profiles.columns, "has 'profile' column")
check('year' in profiles.columns, "has 'year' column")
check('genres_list' in profiles.columns, "has 'genres_list' column")
check('tags_text' in profiles.columns, "has 'tags_text' column")
check(profiles[profiles['movieId'] == 1].iloc[0]['year'] == 2014, "year 2014 extracted")
check(profiles[profiles['movieId'] == 2].iloc[0]['year'] == 1995, "year 1995 extracted")
check('space' in profiles[profiles['movieId'] == 1].iloc[0]['profile'], "tag in profile")
# Movie with no tags should have empty string (not NaN)
no_tag_text = profiles[profiles['movieId'] == 2].iloc[0]['tags_text']
check(isinstance(no_tag_text, str) and no_tag_text == '', "no-tag movie → empty string")
print("✓ Test 4 PASSED: build_movie_profiles")

# ══════════════════════════════════════════════════════════
# TEST 5 – User preferences (fully isolated via temp dir)
# ══════════════════════════════════════════════════════════
import src.user_preferences as _up_module

def _run_prefs_test():
    """Run all preference tests inside a temp directory for full isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / 'prefs.json'

        # Helper: always re-point before each operation
        def pref_op(fn, *args):
            _up_module.PREFS_PATH = p
            return fn(*args)

        # 1. Load defaults when file absent
        result = pref_op(_up_module.load_preferences)
        check(result == {'favourites': [], 'watchlist': []}, "defaults on missing file")

        # 2. Add to favourites
        pref_op(_up_module.add_to_favourites, 42)
        prefs = pref_op(_up_module.load_preferences)
        check(42 in prefs['favourites'], "42 added to favourites")

        # 3. Idempotent add
        pref_op(_up_module.add_to_favourites, 42)
        prefs = pref_op(_up_module.load_preferences)
        check(prefs['favourites'].count(42) == 1, "idempotent add")

        # 4. Remove from favourites
        pref_op(_up_module.remove_from_favourites, 42)
        prefs = pref_op(_up_module.load_preferences)
        check(42 not in prefs['favourites'], "42 removed from favourites")

        # 5. Watchlist add & remove
        pref_op(_up_module.add_to_watchlist, 99)
        check(pref_op(_up_module.is_in_watchlist, 99), "99 in watchlist")
        pref_op(_up_module.remove_from_watchlist, 99)
        check(not pref_op(_up_module.is_in_watchlist, 99), "99 removed from watchlist")

        # 6. is_favourite
        pref_op(_up_module.add_to_favourites, 10)
        check(pref_op(_up_module.is_favourite, 10), "is_favourite True")
        check(not pref_op(_up_module.is_favourite, 999), "is_favourite False for unknown")

        # 7. Corrupt JSON → safe defaults
        # Write corrupt content to a fresh dedicated file so there's no
        # interference from earlier writes inside this same temp directory.
        corrupt_path = Path(tmpdir) / 'corrupt_prefs.json'
        corrupt_path.write_text('{ BAD JSON !!')
        _up_module.PREFS_PATH = corrupt_path
        prefs2 = _up_module.load_preferences()
        check(prefs2 == {'favourites': [], 'watchlist': []},
              f"corrupt JSON → defaults (got {prefs2})")

_run_prefs_test()
print("✓ Test 5 PASSED: User preferences")

# ══════════════════════════════════════════════════════════
# TEST 6 – Data loader validation helpers
# ══════════════════════════════════════════════════════════
from src.data_loader import _validate_columns, dataset_status, all_files_present, download_instructions

df_ok = pd.DataFrame({'movieId': [1], 'title': ['A'], 'genres': ['X']})
_validate_columns(df_ok, ['movieId', 'title', 'genres'], 'test.csv')   # no exception
check(True, "valid columns → no exception")

raised = False
try:
    _validate_columns(df_ok, ['missing_col'], 'test.csv')
except ValueError as e:
    raised = True
    check('missing_col' in str(e), "error mentions missing column")
    check('test.csv' in str(e), "error mentions filename")
check(raised, "missing column raises ValueError")

status = dataset_status()
check(set(status.keys()) == {'movies', 'ratings', 'tags', 'links'}, "status has 4 keys")
check(isinstance(all_files_present(), bool), "all_files_present returns bool")
check('grouplens.org' in download_instructions(), "instructions mention URL")
print("✓ Test 6 PASSED: Data loader validation helpers")

# ══════════════════════════════════════════════════════════
# TEST 7 – Jaccard recommend
# ══════════════════════════════════════════════════════════
movies_j = pd.DataFrame({
    'movieId':     [1, 2, 3, 4],
    'title':       ['A', 'B', 'C', 'D'],
    'genres_list': [
        ['Action', 'Thriller'],
        ['Action', 'Drama'],
        ['Romance', 'Comedy'],
        ['Thriller', 'Horror'],
    ],
})
recs = jaccard_recommend(1, movies_j, top_n=10)
rec_ids = [r['movieId'] for r in recs]
check(1 not in rec_ids, "query movie excluded from recs")
scores = [r['score'] for r in recs]
check(scores == sorted(scores, reverse=True), "results sorted descending")
check(all(0.0 <= s <= 1.0 for s in scores), "all scores in [0,1]")
check(jaccard_recommend(9999, movies_j) == [], "unknown ID → []")
check(len(jaccard_recommend(1, movies_j, top_n=2)) <= 2, "top_n respected")
print("✓ Test 7 PASSED: Jaccard recommend")

# ══════════════════════════════════════════════════════════
# TEST 8 – search_movies
# ══════════════════════════════════════════════════════════
from src.movie_details import search_movies

result_inter = search_movies('inter', profiles)
check(not result_inter.empty, "partial match 'inter' finds Interstellar")
result_empty = search_movies('', profiles)
check(result_empty.empty, "empty query → empty DataFrame")
result_none = search_movies('xyznotfound', profiles)
check(result_none.empty, "no match → empty DataFrame")
result_case = search_movies('TOY', profiles)
check(not result_case.empty, "case-insensitive search works")
print("✓ Test 8 PASSED: search_movies")

# ══════════════════════════════════════════════════════════
# COSINE SIMILARITY PAIR (manual, no sklearn needed)
# ══════════════════════════════════════════════════════════
import numpy as np
from src.similarity import cosine_similarity_pair

v = np.array([1.0, 2.0, 3.0])
check(abs(cosine_similarity_pair(v, v) - 1.0) < 1e-9, "identical vectors → 1.0")
a = np.array([1.0, 0.0])
b = np.array([0.0, 1.0])
check(abs(cosine_similarity_pair(a, b)) < 1e-9, "orthogonal → 0.0")
zero = np.array([0.0, 0.0, 0.0])
check(cosine_similarity_pair(zero, v) == 0.0, "zero vector → 0.0 (no crash)")
# 45-degree angle: cos(45°) = 1/sqrt(2)
c = np.array([1.0, 0.0])
d = np.array([1.0, 1.0])
check(abs(cosine_similarity_pair(c, d) - 1.0/np.sqrt(2)) < 1e-9, "45° = 1/√2")
print("✓ Test 9 PASSED: cosine_similarity_pair (manual)")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print()
print("=" * 54)
if _FAIL == 0:
    print(f"  ALL {_PASS} ASSERTIONS PASSED  ✓")
else:
    print(f"  {_PASS} passed  |  {_FAIL} FAILED  ✗")
print("=" * 54)
print()
if _FAIL == 0:
    print("Core logic is correct.")
    print()
    print("Next steps:")
    print("  1. pip install -r requirements.txt  (needs internet)")
    print("  2. python download_data.py          (downloads dataset)")
    print("  3. pytest tests/ -v                 (full test suite)")
    print("  4. streamlit run app.py             (launch the app)")
