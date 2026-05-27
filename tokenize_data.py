import json
import numpy as np
from eecs148b_hw1.bpe import Tokenizer

# Load vocab and merges
with open("data/vocab.json") as f:
    raw_vocab = json.load(f)
vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}

with open("data/merges.json") as f:
    raw_merges = json.load(f)
merges = [(bytes(a), bytes(b)) for a, b in raw_merges]

tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# Tokenize train
print("Tokenizing train...")
with open("data/train.txt") as f:
    train_ids = list(tok.encode_iterable(f))
np.array(train_ids, dtype=np.uint16).tofile("data/train.bin")
print(f"  {len(train_ids):,} tokens")

# Tokenize valid
print("Tokenizing valid...")
with open("data/valid.txt") as f:
    val_ids = list(tok.encode_iterable(f))
np.array(val_ids, dtype=np.uint16).tofile("data/valid.bin")
print(f"  {len(val_ids):,} tokens")