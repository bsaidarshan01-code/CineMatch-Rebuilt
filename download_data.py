"""
download_data.py
================
Downloads and extracts the MovieLens Latest Small dataset into data/.

Usage – try whichever works on your machine:
    py download_data.py
    python3 download_data.py
    C:\\Users\\HP\\AppData\\Local\\Python\\bin\\python.exe download_data.py

Or simply double-click run.bat
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).parent / "data"
REQUIRED_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]


def download_and_extract() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    already = [f for f in REQUIRED_FILES if (DATA_DIR / f).exists()]
    if len(already) == len(REQUIRED_FILES):
        print("Dataset already present in data/ – nothing to download.")
        return

    missing = [f for f in REQUIRED_FILES if f not in already]
    print(f"Missing files: {missing}")
    print(f"\nDownloading MovieLens dataset from:\n  {URL}\n")
    print("This may take 10–30 seconds depending on your connection …")

    try:
        with urllib.request.urlopen(URL, timeout=90) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            chunks = []
            downloaded = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"  {pct}% ({downloaded // 1024} KB / {total // 1024} KB)", end="\r")
            data = b"".join(chunks)
            print(f"\n  Downloaded {len(data) // 1024} KB total.")
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("\nPlease download manually:")
        print(f"  1. Visit: {URL}")
        print("  2. Extract the ZIP file")
        print("  3. Copy movies.csv, ratings.csv, tags.csv, links.csv into the data/ folder")
        sys.exit(1)

    print("\nExtracting …")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if filename in REQUIRED_FILES:
                dest = DATA_DIR / filename
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                size_kb = dest.stat().st_size // 1024
                print(f"  Extracted -> data/{filename}  ({size_kb} KB)")

    still_missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if still_missing:
        print(f"\nWarning: could not find in ZIP: {still_missing}")
        print("The ZIP structure may have changed. Try downloading manually.")
    else:
        print("\n✓ Dataset ready!")
        print("\nNext step: streamlit run app.py")
        print("     (or:  python -m streamlit run app.py)")


if __name__ == "__main__":
    download_and_extract()
