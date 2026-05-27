"""Transformer language model — all components built from scratch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# =========================================================================
# Utility: softmax
# =========================================================================

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax along *dim*."""
    x_max = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)


# =========================================================================
# Linear
# =========================================================================

class Linear(nn.Module):
    """Linear layer y = Wx (no bias)."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


# =========================================================================
# Embedding
# =========================================================================

class Embedding(nn.Module):
    """Token embedding lookup table."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


# =========================================================================
# LayerNorm
# =========================================================================

class LayerNorm(nn.Module):
    """Standard layer normalization with learnable affine transform."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        result = x_norm * self.weight.float() + self.bias.float()
        return result.to(in_dtype)


# =========================================================================
# Position-wise Feed-Forward Network
# =========================================================================

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = W2 ReLU(W1 x)."""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU implemented manually
        h = self.w1(x)
        h = h * (h > 0).to(h.dtype)  # max(0, h)
        return self.w2(h)


# =========================================================================
# Sinusoidal Positional Encoding
# =========================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional embeddings (Vaswani et al.)."""

    def __init__(self, d_model: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model, device=device, dtype=dtype)
        position = torch.arange(0, max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if dtype is not None:
            pe = pe.to(dtype)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        return self.pe[token_positions]


# =========================================================================
# Scaled Dot-Product Attention
# =========================================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Q : (..., seq_len_q, d_k)
    K : (..., seq_len_k, d_k)
    V : (..., seq_len_k, d_v)
    mask : (seq_len_q, seq_len_k) boolean — True = attend, False = ignore
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (..., seq_q, seq_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    # Replace NaN from all-masked rows (full -inf row softmaxes to nan)
    attn_weights = attn_weights.nan_to_num(0.0)

    return attn_weights @ V


# =========================================================================
# Multi-Head Self-Attention
# =========================================================================

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h, dk = self.num_heads, self.d_k

        Q = self.W_Q(x).view(B, T, h, dk).transpose(1, 2)  # (B, h, T, dk)
        K = self.W_K(x).view(B, T, h, dk).transpose(1, 2)
        V = self.W_V(x).view(B, T, h, dk).transpose(1, 2)

        # Causal mask: position i attends to positions j <= i
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        causal_mask = ~causal_mask  # True where we *do* attend

        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)  # (B, h, T, dk)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(attn_out)


# =========================================================================
# Transformer Block (pre-norm)
# =========================================================================

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.ff = PositionwiseFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# =========================================================================
# Full Transformer LM
# =========================================================================

class TransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        use_pos_emb: bool = True,
        use_layer_norm: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = SinusoidalPositionalEncoding(d_model, context_length, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.final_ln = LayerNorm(d_model, device=device, dtype=dtype)

        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids : (B, T) long tensor
        Returns   : (B, T, vocab_size) logits
        """
        B, T = token_ids.shape
        x = self.token_emb(token_ids)

        if self.use_pos_emb:
            positions = torch.arange(T, device=token_ids.device)
            x = x + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        if self.use_layer_norm:
            x = self.final_ln(x)

        return self.lm_head(x)
