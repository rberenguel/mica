#!/usr/bin/env python3
"""
Encode downloaded Gutenberg books into binary token files for Phase 1 training.
Run AFTER download_gutenberg.py and prepare.py.

Reads:  corpus/raw/gutenberg/gut_*.txt
Cleans: via clean_corpus.clean_text() (strips Gutenberg boilerplate, OCR noise)
Writes: wiki_train.bin, wiki_val.bin  (uint16, same format as train.bin/val.bin)
"""
import re
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from clean_corpus import clean_text

TOKENIZER_PATH = "mica_tokenizer.json"
GUT_DIR        = Path("corpus/raw/gutenberg")
VAL_FRACTION   = 0.02

print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
eot_id = tokenizer.token_to_id("<|endoftext|>")
eos_id = tokenizer.token_to_id("<|eos|>")
if eos_id is None:
    raise RuntimeError("Tokenizer is missing <|eos|>. Re-run prepare.py first to rebuild mica_tokenizer.json.")

books = sorted(GUT_DIR.glob("gut_*.txt"))
if not books:
    raise FileNotFoundError(f"No books found in {GUT_DIR}. Run download_gutenberg.py first.")
print(f"Found {len(books):,} books in {GUT_DIR}")

all_ids = []
for i, path in enumerate(books):
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned, _ = clean_text(raw)
    for para in cleaned.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        for sent in re.split(r'(?<=[.?!])\s+', para):
            sent = sent.strip()
            if sent:
                all_ids.extend(tokenizer.encode(sent).ids)
                all_ids.append(eos_id)
        all_ids.append(eot_id)
    all_ids.append(eot_id)   # extra boundary between books
    if (i + 1) % 100 == 0:
        print(f"  {i+1:,} / {len(books):,} books  ({len(all_ids):,} tokens so far)")

print(f"\nTotal tokens: {len(all_ids):,}")

n     = len(all_ids)
split = int(n * (1 - VAL_FRACTION))
train_ids = np.array(all_ids[:split], dtype=np.uint16)
val_ids   = np.array(all_ids[split:], dtype=np.uint16)

print(f"Train: {len(train_ids):,} tokens")
print(f"Val:   {len(val_ids):,} tokens")

train_ids.tofile("wiki_train.bin")
val_ids.tofile("wiki_val.bin")
print("Saved wiki_train.bin and wiki_val.bin. Ready for train_phase1.py.")
