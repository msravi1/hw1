"""Autoregressive decoding with temperature and nucleus (top-p) sampling."""

from __future__ import annotations

import torch
from .model import softmax


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str = "",
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    eos_token: str = "<|endoftext|>",
) -> str:
    """Generate text autoregressively from *model*.

    Parameters
    ----------
    model       : TransformerLM in eval mode.
    tokenizer   : Tokenizer with encode / decode.
    prompt      : Optional prompt string.
    max_tokens  : Max new tokens to generate.
    temperature : Softmax temperature (>0). Lower → more greedy.
    top_p       : Nucleus sampling threshold in (0, 1].
    device      : Torch device.
    eos_token   : End-of-sequence special token string.

    Returns
    -------
    The generated string (prompt + completion).
    """
    model.eval()
    eos_id = tokenizer.bytes_to_id.get(eos_token.encode("utf-8"))
    context_length = model.context_length

    if prompt:
        ids = tokenizer.encode(prompt)
    else:
        ids = []

    for _ in range(max_tokens):
        # Truncate to context_length
        input_ids = ids[-context_length:] if len(ids) > context_length else ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        logits = model(x)  # (1, T, V)
        next_logits = logits[0, -1, :]  # (V,)

        # Temperature scaling
        if temperature != 1.0 and temperature > 0:
            next_logits = next_logits / temperature

        probs = softmax(next_logits, dim=-1)

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # Find the cutoff index (smallest set with cumulative prob >= top_p)
            mask = cumsum - sorted_probs >= top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            # Sample from the truncated distribution
            idx_in_sorted = torch.multinomial(sorted_probs, 1).item()
            next_id = sorted_indices[idx_in_sorted].item()
        else:
            next_id = torch.multinomial(probs, 1).item()

        ids.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(ids)
