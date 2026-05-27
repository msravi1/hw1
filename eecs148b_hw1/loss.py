"""Cross-entropy loss (numerically stable, from scratch)."""

import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean cross-entropy loss.

    Parameters
    ----------
    logits  : (..., vocab_size) — unnormalised scores.
    targets : (...,)            — integer class labels.

    Returns
    -------
    Scalar mean loss.
    """
    # Subtract max for numerical stability
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - logits_max                          # (..., V)

    # log-sum-exp
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))  # (...)

    # Gather the logit corresponding to the target class
    # targets_logit = shifted[..., target]
    targets_logit = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (...)

    # NLL = log_sum_exp - targets_logit  (cancels log and exp where possible)
    loss = log_sum_exp - targets_logit
    return loss.mean()
