import os
import re
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# --- 1. Train the Custom Tokenizer ---
print("Training custom BPE Tokenizer...")
vocab_size = 5000

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# <|eos|>  = sentence boundary (finer grain)
# <|endoftext|> = paragraph / document boundary (coarser)
trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|eos|>"])

# Train directly on your raw text file
corpus_path = "corpus/corpus.txt"
if not os.path.exists(corpus_path):
    raise FileNotFoundError(f"Please place your text data in {corpus_path}")

tokenizer.train([corpus_path], trainer=trainer)
tokenizer.save("mica_tokenizer.json")
print("Tokenizer saved as mica_tokenizer.json")

# --- 2. Encode the Corpus ---
print("Reading and encoding corpus...")
with open(corpus_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Special tokens must be injected at the ID level — BPE will never produce
# them from raw text even if the literal string appears in the input.
eot_id = tokenizer.token_to_id("<|endoftext|>")
eos_id = tokenizer.token_to_id("<|eos|>")

# Structure:  sentence<EOS>sentence<EOS>...<EOT>  paragraph<EOT>  ...
# EOS marks sentence endings; EOT marks paragraph / story boundaries.
ids = []
n_sent = 0
for para in text.split('\n\n'):
    para = para.strip()
    if not para:
        continue
    for sent in re.split(r'(?<=[.?!])\s+', para):
        sent = sent.strip()
        if sent:
            ids.extend(tokenizer.encode(sent).ids)
            ids.append(eos_id)
            n_sent += 1
    ids.append(eot_id)
print(f"Corpus encoded into {len(ids):,} tokens ({n_sent:,} sentences, {text.count(chr(10)+chr(10)):,} paragraph breaks).")

# --- 3. Create Splits and Save Binary Files ---
# 90% for training, 10% for validation to check for overfitting
n = len(ids)
train_ids = ids[:int(n * 0.9)]
val_ids = ids[int(n * 0.9):]

print(f"Train set has {len(train_ids):,} tokens")
print(f"Val set has {len(val_ids):,} tokens")

# Export to bin files
# Since our vocab_size is 10,000, every token fits safely in a 16-bit integer (max 65,535).
# This cuts our RAM and SSD usage in half compared to standard 32-bit ints!
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile('train.bin')
val_ids.tofile('val.bin')
print("Saved data/train.bin and data/val.bin. You are ready to train!")