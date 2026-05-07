#!/usr/bin/env python3
"""
Download English fiction from Project Gutenberg for Phase 1 pre-training.
Saves plain-text books to corpus/raw/gutenberg/.

Gutenberg's robots policy requires polite downloading — this script enforces
a 3-second delay between requests. At the default 500 books that's ~25 minutes.
Already-downloaded books are skipped, so the script is safe to re-run.

Usage:
    uv run python download_gutenberg.py              # download up to 500 books
    uv run python download_gutenberg.py --max 1000   # download up to 1000 books
    uv run python download_gutenberg.py --list       # show filtered catalog, no download
"""
import csv
import gzip
import io
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
OUT_DIR     = Path("corpus/raw/gutenberg")
DELAY       = 3.0    # seconds between requests — Gutenberg asks for this
DEFAULT_MAX = 500

# Library of Congress Classification codes for prose fiction in English
FICTION_LOCC = {"PR", "PS", "PZ"}  # English lit, American lit, fiction/juvenile

HEADERS = {
    "User-Agent": "Mica LM research project (educational, non-commercial)",
}


def fetch_catalog() -> list[dict]:
    print("Fetching Gutenberg catalog...")
    req = urllib.request.Request(CATALOG_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = gzip.decompress(resp.read())
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    return list(reader)


def is_english_fiction(row: dict) -> bool:
    if row.get("Language", "").strip() != "en":
        return False
    if row.get("Type", "").strip() != "Text":
        return False
    # Exclude poetry and drama — we want prose
    subjects = row.get("Subjects", "").lower()
    if any(x in subjects for x in ("poetry", "drama", "plays", "verse")):
        return False
    locc = row.get("LoCC", "")
    # LoCC field can contain multiple codes separated by ";"
    codes = {c.strip()[:2] for c in locc.split(";")}
    return bool(codes & FICTION_LOCC)


def try_download(book_id: str) -> str | None:
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue          # try next URL
            raise
        except Exception:
            continue
    return None


def main():
    args = sys.argv[1:]
    list_only = "--list" in args
    max_books = DEFAULT_MAX
    if "--max" in args:
        max_books = int(args[args.index("--max") + 1])

    rows = fetch_catalog()
    fiction = [r for r in rows if is_english_fiction(r)]
    print(f"Catalog: {len(rows):,} total entries → {len(fiction):,} English fiction texts")

    if list_only:
        for r in fiction[:50]:
            print(f"  {r['Text#']:>6}  {r['Title'][:60]}")
        if len(fiction) > 50:
            print(f"  ... and {len(fiction)-50} more")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failed = 0

    for row in fiction:
        if downloaded >= max_books:
            break

        book_id = row["Text#"].strip()
        title   = row["Title"][:60]
        dest    = OUT_DIR / f"gut_{book_id}.txt"

        if dest.exists():
            skipped += 1
            continue

        print(f"  [{downloaded+1}/{max_books}] {book_id}: {title} ...", end=" ", flush=True)
        text = try_download(book_id)

        if text:
            dest.write_text(text, encoding="utf-8")
            size_kb = dest.stat().st_size / 1024
            print(f"{size_kb:.0f} KB")
            downloaded += 1
        else:
            print("not found")
            failed += 1

        time.sleep(DELAY)

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} not found")
    print(f"Books saved to {OUT_DIR}/")
    print("\nNext steps:")
    print("  uv run python prepare_wiki.py   # encode books → wiki_train.bin / wiki_val.bin")


if __name__ == "__main__":
    main()
