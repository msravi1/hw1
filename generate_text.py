import json, torch
from eecs148b_hw1.bpe import Tokenizer
from eecs148b_hw1.model import TransformerLM
from eecs148b_hw1.decode import generate

# Load tokenizer
with open("data/vocab.json") as f:
    vocab = {int(k): bytes(v) for k, v in json.load(f).items()}
with open("data/merges.json") as f:
    merges = [(bytes(a), bytes(b)) for a, b in json.load(f)]
tok = Tokenizer(vocab, merges, ["<|endoftext|>"])

# Load model
model = TransformerLM(10000, 256, 512, 4, 8, 2048)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()

# Generate
text = generate(model, tok, prompt="Once upon a time",
                max_tokens=256, temperature=0.8, top_p=0.95)
print(text)