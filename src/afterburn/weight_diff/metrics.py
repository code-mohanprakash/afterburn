"""Pure metric functions for weight comparison."""

from __future__ import annotations

import torch


def l2_norm_diff(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute L2 norm of (trained - base)."""
    diff = trained.float() - base.float()
    return torch.linalg.vector_norm(diff.flatten()).item()


def cosine_similarity(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute cosine similarity between flattened weight tensors.

    Returns a value in [-1, 1]. 1.0 means identical direction.
    """
    b = base.flatten().float()
    t = trained.flatten().float()

    b_norm = torch.linalg.vector_norm(b)
    t_norm = torch.linalg.vector_norm(t)

    if b_norm < 1e-10 or t_norm < 1e-10:
        return 0.0

    return torch.dot(b, t).item() / (b_norm.item() * t_norm.item())


def frobenius_norm_diff(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute Frobenius norm of the difference.

    For 1D tensors, this is equivalent to L2 norm.
    For 2D+ tensors, computes the matrix Frobenius norm.
    """
    diff = trained.float() - base.float()
    if diff.dim() < 2:
        return torch.linalg.vector_norm(diff).item()
    return torch.linalg.matrix_norm(diff.reshape(diff.shape[0], -1), ord="fro").item()


def relative_change(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute relative change: frobenius(diff) / frobenius(base).

    Returns 0.0 if the base tensor is near-zero.
    """
    base_f = base.float()
    if base_f.dim() < 2:
        base_norm = torch.linalg.vector_norm(base_f).item()
    else:
        base_norm = torch.linalg.matrix_norm(
            base_f.reshape(base_f.shape[0], -1), ord="fro"
        ).item()

    if base_norm < 1e-10:
        return 0.0

    return frobenius_norm_diff(base, trained) / base_norm


def max_abs_diff(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute maximum absolute element-wise difference."""
    diff = (trained.float() - base.float()).abs()
    return diff.max().item()


def mean_abs_diff(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute mean absolute element-wise difference."""
    diff = (trained.float() - base.float()).abs()
    return diff.mean().item()
