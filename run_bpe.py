import json
from eecs148b_hw1.bpe import train_bpe

vocab, merges = train_bpe("data/train.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])

vocab_serial = {k: list(v) for k, v in vocab.items()}
with open("data/vocab.json", "w") as f:
    json.dump(vocab_serial, f)

merges_serial = [[list(a), list(b)] for a, b in merges]
with open("data/merges.json", "w") as f:
    json.dump(merges_serial, f)

longest = max(vocab.values(), key=len)
print(f"Vocab size: {len(vocab)}")
print(f"Longest token: {longest} ({len(longest)} bytes)")
