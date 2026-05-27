from __future__ import annotations

import os
from typing import Any

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eecs148b_hw1.bpe import train_bpe as _train_bpe, Tokenizer as _Tokenizer
from eecs148b_hw1.model import (
    Linear as _Linear,
    Embedding as _Embedding,
    LayerNorm as _LayerNorm,
    PositionwiseFeedForward as _FFN,
    SinusoidalPositionalEncoding as _SinPE,
    softmax as _softmax,
    scaled_dot_product_attention as _sdpa,
    MultiHeadSelfAttention as _MHA,
    TransformerBlock as _Block,
    TransformerLM as _LM,
)
from eecs148b_hw1.loss import cross_entropy_loss as _ce_loss
from eecs148b_hw1.data import get_batch as _get_batch


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    module = _Linear(d_in, d_out)
    module.load_state_dict({"weight": weights})
    return module(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    module = _Embedding(vocab_size, d_model)
    module.load_state_dict({"weight": weights})
    return module(token_ids)


def run_ffn(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a 2-layer ReLU FFN, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Inner dimensionality of the feed-forward network.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for the first linear layer.
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for the second linear layer.
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    module = _FFN(d_model, d_ff)
    module.load_state_dict({"w1.weight": w1_weight, "w2.weight": w2_weight})
    return module(in_features)


def run_layernorm(
    d_model: int,
    eps: float,
    weight: Float[Tensor, " d_model"],
    bias: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the affine parameters of LayerNorm, return the output of running
    LayerNorm on the input features.

    Args:
        d_model (int): The dimensionality of the LayerNorm input.
        eps (float): A value added to the denominator for numerical stability.
        weight (Float[Tensor, "d_model"]): LayerNorm scale parameters.
        bias (Float[Tensor, "d_model"]): LayerNorm bias parameters.
        in_features (Float[Tensor, "... d_model"]): Input features to run LayerNorm on.

    Returns:
        Float[Tensor, "... d_model"]: Tensor with the output of running LayerNorm on `in_features`.
    """
    module = _LayerNorm(d_model, eps=eps)
    module.load_state_dict({"weight": weight, "bias": bias})
    return module(in_features)


def run_sinusoidal_pe(
    d_model: int,
    max_seq_len: int,
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """Return sinusoidal positional embeddings for the given token positions."""
    module = _SinPE(d_model, max_seq_len)
    return module(token_positions)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return _sdpa(Q, K, V, mask=mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    module = _MHA(d_model, num_heads)
    module.load_state_dict({
        "W_Q.weight": q_proj_weight,
        "W_K.weight": k_proj_weight,
        "W_V.weight": v_proj_weight,
        "W_O.weight": o_proj_weight,
    })
    return module(in_features)


def _remap_block_keys(weights: dict[str, Tensor]) -> dict[str, Tensor]:
    """Remap reference block state-dict keys to our TransformerBlock keys."""
    mapping = {
        "attn.q_proj.weight": "mha.W_Q.weight",
        "attn.k_proj.weight": "mha.W_K.weight",
        "attn.v_proj.weight": "mha.W_V.weight",
        "attn.output_proj.weight": "mha.W_O.weight",
        "ln1.weight": "ln1.weight",
        "ln1.bias": "ln1.bias",
        "ffn.fc1.weight": "ff.w1.weight",
        "ffn.fc2.weight": "ff.w2.weight",
        "ln2.weight": "ln2.weight",
        "ln2.bias": "ln2.bias",
    }
    return {mapping[k]: v for k, v in weights.items()}


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use LayerNorm and the 2-layer ReLU FFN.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln1.bias`
                Bias of affine transform for the first LayerNorm.
                Shape is (d_model,).
            - `ffn.fc1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.fc2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln2.bias`
                Bias of affine transform for the second LayerNorm.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features.
    """
    module = _Block(d_model, num_heads, d_ff)
    module.load_state_dict(_remap_block_keys(weights))
    return module(in_features)


def _remap_lm_keys(weights: dict[str, Tensor]) -> dict[str, Tensor]:
    """Remap reference TransformerLM state-dict keys to our TransformerLM keys."""
    new = {}
    for k, v in weights.items():
        nk = k
        # Token embeddings
        nk = nk.replace("token_embeddings.weight", "token_emb.weight")
        # Layers → blocks
        nk = nk.replace("layers.", "blocks.")
        # Attention projections
        nk = nk.replace("attn.q_proj.weight", "mha.W_Q.weight")
        nk = nk.replace("attn.k_proj.weight", "mha.W_K.weight")
        nk = nk.replace("attn.v_proj.weight", "mha.W_V.weight")
        nk = nk.replace("attn.output_proj.weight", "mha.W_O.weight")
        # FFN
        nk = nk.replace("ffn.fc1.weight", "ff.w1.weight")
        nk = nk.replace("ffn.fc2.weight", "ff.w2.weight")
        # Final layer norm
        nk = nk.replace("ln_final.", "final_ln.")
        new[nk] = v
    return new


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
            - `layers.{num_layers}.attn.k_proj.weight`
            - `layers.{num_layers}.attn.v_proj.weight`
            - `layers.{num_layers}.attn.output_proj.weight`
            - `layers.{num_layers}.ln1.weight`
            - `layers.{num_layers}.ln1.bias`
            - `layers.{num_layers}.ffn.fc1.weight`
            - `layers.{num_layers}.ffn.fc2.weight`
            - `layers.{num_layers}.ln2.weight`
            - `layers.{num_layers}.ln2.bias`
            - `ln_final.weight`
            - `ln_final.bias`
            - `lm_head.weight`
        in_indices (Int[Tensor, "batch_size sequence_length"]): Input indices.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Predicted unnormalized
        next-word distribution for each token.
    """
    model = _LM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff)
    model.load_state_dict(_remap_lm_keys(weights))
    return model(in_indices)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    return _get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return _softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return _ce_loss(inputs, targets)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return _Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return _train_bpe(str(input_path), vocab_size, special_tokens)