"""Numerically-stable softmax, activation functions, and cross-entropy loss."""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of the same shape summing to 1 along ``dim``.
    """

    # Needed to subtract by x to prevent errors (required on pdf)
    x_shifted = x - x.max(dim=dim, keepdim=True).values

    # Get tensor values
    exp_vals = torch.exp(x_shifted)

    # Sum exp_vals into new tensor to divide by
    exp_vals_sum = exp_vals.sum(dim=dim, keepdim=True)

    # Divide by
    return exp_vals / exp_vals_sum



def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish) activation.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of the same shape.
    """
    
    # Create sigmoid bottom
    bottom = 1 + torch.exp(-x)

    # Divide x by it
    return x / bottom



def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """Token-level cross-entropy loss (numerically stable).

    Args:
        logits: ``(B, T, V)`` — raw scores.
        targets: ``(B, T)`` — ground-truth token IDs.

    Returns:
        Scalar mean cross-entropy loss.
    """

    # Needed to subtract by x to prevent errors (required on pdf, even in cross entropy loss)
    logits_shifted = logits - logits.max(dim=-1, keepdim=True).values

    # Get exp tensor values.
    exp_vals = torch.exp(logits_shifted)
    
    # Sum across v dimension
    summed_values = exp_vals.sum(dim=-1)

    # Apply log
    logged_values = torch.log(summed_values)

    # Indexed logits
    indexed_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Apply max and subtract
    pre_final_sum_values = logits.max(dim=-1).values + logged_values - indexed_logits

    # Sum over all values (no dimensions needed because we need a 0 dimensional scalar output)
    final_sum = pre_final_sum_values.sum()

    # Divide by B and T dimension sizes to get average
    final_loss = final_sum / (logits.shape[0] * logits.shape[1])

    return final_loss

