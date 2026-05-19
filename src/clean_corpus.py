#!/usr/bin/env python3
"""
Clean OCR'd pulp magazine text for corpus ingestion.
Handles archive.org scan artifacts, Project Gutenberg boilerplate,
two-column layout drift, and unicode noise.

Usage:
    python clean_corpus.py file1.txt file2.txt ...        # print cleaned text
    python clean_corpus.py file1.txt --append             # append to corpus/corpus.txt
    python clean_corpus.py --rebuild                      # rebuild corpus.txt from all corpus/raw/*.txt
    python clean_corpus.py file1.txt --stats              # show what was removed, don't output
"""
import re
import sys
import unicodedata
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
RAW_DIR = CORPUS_DIR / "raw"
CORPUS_FILE = CORPUS_DIR / "corpus.txt"

# ── Gutenberg boilerplate delimiters ─────────────────────────────────────────

PG_START = re.compile(
    r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG",
    re.IGNORECASE,
)
PG_END = re.compile(
    r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG",
    re.IGNORECASE,
)

# ── Line-level patterns to drop entirely ─────────────────────────────────────

DROP_PATTERNS = [
    # Standalone page numbers (possibly with whitespace)
    re.compile(r"^\s*\d{1,4}\s*$"),
    # Horizontal rules: ---, ***, ===, ~~~, ___ (3+ chars)
    re.compile(r"^\s*[-*=~_]{3,}\s*$"),
    # Chapter headings: CHAPTER I, CHAPTER 12, etc.
    re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\s*$"),
    # Magazine/story section titles: all-caps, no sentence-ending period,
    # 2–7 words. Excludes telegrams/notes which end with periods.
    re.compile(r"^\s*(?:[A-Z]+(?:\s+[A-Z']+){1,6})\s*$"),
    # "Continued on page N" / "Continued from page N"
    re.compile(r"^\s*continued (on|from) page \d+", re.IGNORECASE),
    # "Page N" or "p. N" standalone
    re.compile(r"^\s*p+age\.?\s*\d+\s*$", re.IGNORECASE),
    # Table of contents entries: "Story Title .... 42"
    re.compile(r"^.{5,60}\.{3,}\s*\d+\s*$"),
    # Price / subscription boilerplate
    re.compile(r"^\s*(subscription|price|cents|per copy|published (monthly|bi-monthly))", re.IGNORECASE),
    # Copyright lines
    re.compile(r"^\s*copyright\b", re.IGNORECASE),
    # Produced-by / scanned-by credits (PG)
    re.compile(r"^\s*(produced|scanned|prepared|transcribed|digitized)\s+by\b", re.IGNORECASE),
    # URLs
    re.compile(r"https?://\S+"),
    # Underscored section headers from Gutenberg markup: _CHAPTER IV_
    # (kept if they contain lowercase — likely emphasis, not a header)
    re.compile(r"^\s*_[A-Z][A-Z\s\d\.\-]+_\s*$"),
]

# ── Unicode ligature map ──────────────────────────────────────────────────────

LIGATURES = str.maketrans({
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
})

# ── Per-line cleaning ─────────────────────────────────────────────────────────

def clean_line(line: str) -> str:
    # Translate ligatures
    line = line.translate(LIGATURES)

    # Normalize unicode (NFC): fixes decomposed accents, curly quotes, etc.
    line = unicodedata.normalize("NFC", line)

    # Smart quotes → straight quotes
    line = line.replace("‘", "'").replace("’", "'")
    line = line.replace("“", '"').replace("”", '"')

    # Em/en dashes → double hyphen (preserves pause cadence)
    line = line.replace("—", "--").replace("–", "--")

    # Non-breaking and other funky spaces → regular space
    line = re.sub(r"[     ​]", " ", line)

    # Collapse multiple spaces (but not leading indent — preserve paragraph indent if any)
    line = re.sub(r"  +", " ", line)

    return line.rstrip()


def dehyphenate(lines: list[str]) -> list[str]:
    """Join lines where a word is broken across the line end with a hyphen."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith("-") and i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()
            # Only join if next line starts with a lowercase letter (avoids
            # joining intentional em-dash constructs or proper nouns split by
            # layout, though proper nouns starting lowercase are rare enough)
            if next_line and next_line[0].islower():
                out.append(line[:-1] + next_line)
                i += 2
                continue
        out.append(line)
        i += 1
    return out


def strip_pg_boilerplate(lines: list[str]) -> list[str]:
    """Remove everything before PG start marker and after PG end marker."""
    start = 0
    end = len(lines)
    for i, line in enumerate(lines):
        if PG_START.match(line):
            start = i + 1
            break
    for i, line in enumerate(lines):
        if PG_END.match(line):
            end = i
            break
    return lines[start:end]


def should_drop(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False  # blank lines handled separately
    return any(p.search(stripped) for p in DROP_PATTERNS)


def collapse_blank_lines(lines: list[str], max_consecutive: int = 2) -> list[str]:
    out = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_consecutive:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    return out


# ── Main cleaning pipeline ────────────────────────────────────────────────────

def clean_text(raw: str) -> tuple[str, dict]:
    lines = raw.splitlines()
    original_count = len(lines)

    lines = strip_pg_boilerplate(lines)
    after_pg = len(lines)

    lines = [clean_line(l) for l in lines]
    lines = dehyphenate(lines)

    kept = []
    dropped = 0
    for line in lines:
        if should_drop(line):
            dropped += 1
        else:
            kept.append(line)

    kept = collapse_blank_lines(kept)

    stats = {
        "original_lines": original_count,
        "after_pg_strip": after_pg,
        "lines_dropped": dropped,
        "final_lines": len(kept),
    }
    return "\n".join(kept).strip() + "\n", stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    mode_append  = "--append"  in args
    mode_rebuild = "--rebuild" in args
    mode_stats   = "--stats"   in args
    files = [a for a in args if not a.startswith("--")]

    if not files and not mode_rebuild:
        print("No input files specified.", file=sys.stderr)
        sys.exit(1)

    cleaned_parts = []
    for path_str in files:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping.", file=sys.stderr)
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned, stats = clean_text(raw)
        cleaned_parts.append(cleaned)

        if mode_stats or True:  # always show stats to stderr
            orig_kb = len(raw) / 1024
            clean_kb = len(cleaned) / 1024
            print(
                f"{path.name}: {stats['original_lines']} lines → {stats['final_lines']} lines "
                f"({orig_kb:.0f}KB → {clean_kb:.0f}KB, dropped {stats['lines_dropped']} pattern-matched lines)",
                file=sys.stderr,
            )

    combined = "\n\n".join(cleaned_parts)

    if mode_stats:
        return  # stats already printed, done

    if mode_rebuild:
        sources = sorted(RAW_DIR.glob("*.txt"))
        parts = []
        for src in sources:
            raw = src.read_text(encoding="utf-8", errors="replace")
            cleaned, stats = clean_text(raw)
            parts.append(cleaned)
            print(
                f"  {src.name}: {stats['original_lines']} → {stats['final_lines']} lines",
                file=sys.stderr,
            )
        full = "\n\n".join(parts)
        CORPUS_FILE.write_text(full, encoding="utf-8")
        print(f"Rebuilt {CORPUS_FILE} ({len(full)/1024:.0f} KB)", file=sys.stderr)
        return

    if mode_append:
        with CORPUS_FILE.open("a", encoding="utf-8") as f:
            f.write("\n\n" + combined)
        print(
            f"Appended {len(combined)/1024:.0f} KB to {CORPUS_FILE}",
            file=sys.stderr,
        )
        return

    # Default: print to stdout
    print(combined)


if __name__ == "__main__":
    main()
