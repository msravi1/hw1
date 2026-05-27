"""Data loading for language model training."""

import numpy as np
import torch


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) pairs from *data*.

    Parameters
    ----------
    data           : 1-D numpy array of token IDs.
    batch_size     : number of sequences per batch.
    context_length : length of each sequence.
    device         : torch device string.

    Returns
    -------
    inputs  : (batch_size, context_length) LongTensor
    targets : (batch_size, context_length) LongTensor
    """
    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=(batch_size,))

    inputs = np.stack([data[s : s + context_length] for s in starts])
    targets = np.stack([data[s + 1 : s + 1 + context_length] for s in starts])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs, targets
