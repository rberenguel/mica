#!/usr/bin/env python3
"""
Download plain-text OCR from archive.org magazine issues.
Uses the archive.org Items API to find and fetch the best text file per item.

Usage:
    python download_corpus.py                      # download all items in ITEMS list
    python download_corpus.py bm_1920_08           # download a single item by ID
    python download_corpus.py --list               # print item IDs and exit
    python download_corpus.py --dry-run            # show what would be downloaded

Downloaded files land in corpus/raw/ and are NOT automatically cleaned or
added to corpus.txt. Run clean_corpus.py on them first.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

RAW_DIR = Path(__file__).parent / "corpus" / "raw"

# ── Items to download ─────────────────────────────────────────────────────────
# Add/remove archive.org item identifiers here.
# Find them in the URL: archive.org/details/<IDENTIFIER>

ITEMS = [
    # Black Mask — where Hammett published; pre-1928 = fully public domain
    "bm_1920_08",
    "bm_1920_09",
    "bm_1920_11",
    "bm_1922_08",
    # Note: BlackMask193X identifiers on archive.org are image-only scans, no OCR text available

    # Detective Story Magazine — broad detective pulp, pre-1928
    "sim_street-smiths-detective-story-magazine_1920-01-27_29_4",
    "sim_street-smiths-detective-story-magazine_1920-04-20_31_2",
    "sim_street-smiths-detective-story-magazine_1921-05-07_40_2",
    "detective-story-magazine-v-091-n-06-1927-04-16",
    "detective-story-magazine-v-098-n-02-1928-01-07",

    # Dime Detective — hardboiled era, most issues public domain
    "dime-detective-v-03-n-03-1932-09-ibcbc-115-6-dmgd",
    "dime-detective-v-09-n-03-1933-12-15",
    "dimedetectivev10n0419340301",

    # True Detective Mysteries — 1929-1930, fully public domain
    "TrueDetectiveJan1930",
    "TrueDetectiveFeb1930",
    "TrueDetective0330",
    "TrueDetective0430",
    "TrueDetectiveMay1930",
    "TrueDetective0730",
    "TrueDetective0830",
    "TrueDetective0930",
    "TrueDetective1030",
]

# Preference order for text file formats (first match wins)
FORMAT_PREFERENCE = [
    "_djvu.txt",       # DjVu OCR text — usually best quality
    "_full.txt",       # Full text when available
    ".txt",            # Generic plain text
    "_abbyy.gz",       # ABBYY OCR (gzipped XML — skip unless nothing else)
]

API_BASE = "https://archive.org"
SLEEP_BETWEEN = 1.5   # seconds — be polite to archive.org


def get_item_files(identifier: str) -> list[dict]:
    url = f"{API_BASE}/metadata/{identifier}/files"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("result", [])
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} fetching metadata for {identifier}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  Error fetching metadata for {identifier}: {e}", file=sys.stderr)
        return []


def pick_text_file(files: list[dict]) -> str | None:
    names = [f["name"] for f in files]
    for suffix in FORMAT_PREFERENCE:
        if suffix == "_abbyy.gz":
            continue  # skip compressed XML
        for name in names:
            if name.endswith(suffix):
                return name
    return None


def download_file(identifier: str, filename: str, dest: Path) -> bool:
    url = f"{API_BASE}/download/{urllib.parse.quote(identifier)}/{urllib.parse.quote(filename)}"
    print(f"  Downloading {filename} ...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            dest.write_bytes(resp.read())
        size_kb = dest.stat().st_size / 1024
        print(f"{size_kb:.0f} KB")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def process_item(identifier: str, dry_run: bool = False) -> bool:
    dest = RAW_DIR / f"{identifier}.txt"
    if dest.exists():
        print(f"[skip] {identifier} — already downloaded")
        return True

    print(f"[fetch] {identifier}")
    files = get_item_files(identifier)
    if not files:
        print(f"  No files found for {identifier}")
        return False

    filename = pick_text_file(files)
    if not filename:
        available = [f["name"] for f in files[:8]]
        print(f"  No suitable text file found. Available: {available}")
        return False

    if dry_run:
        print(f"  Would download: {filename}")
        return True

    return download_file(identifier, filename, dest)


def main():
    args = sys.argv[1:]
    dry_run  = "--dry-run" in args
    list_only = "--list" in args
    args = [a for a in args if not a.startswith("--")]

    if list_only:
        for item in ITEMS:
            print(item)
        return

    items = args if args else ITEMS

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RAW_DIR}")
    print(f"Items to process: {len(items)}\n")

    ok = failed = skipped = 0
    for i, identifier in enumerate(items):
        result = process_item(identifier, dry_run=dry_run)
        if result:
            ok += 1
        else:
            failed += 1

        if not dry_run and i < len(items) - 1:
            time.sleep(SLEEP_BETWEEN)

    print(f"\nDone: {ok} downloaded, {failed} failed")
    if not dry_run and ok > 0:
        print(f"\nNext step: clean and add to corpus:")
        print(f"  uv run python clean_corpus.py corpus/raw/*.txt --append")


if __name__ == "__main__":
    main()
