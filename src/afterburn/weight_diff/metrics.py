"""Pure metric functions for weight comparison."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class SVDResult:
    """SVD decomposition of a weight diff matrix."""

    top_singular_values: list[float]
    effective_rank: int
    concentration_ratio: float
    stable_rank: float


def svd_analysis(
    base: torch.Tensor,
    trained: torch.Tensor,
    top_k: int = 10,
    energy_threshold: float = 0.9,
) -> SVDResult | None:
    """Compute SVD of the weight difference matrix.

    Returns top-k singular values, effective rank (number of singular values
    needed to capture energy_threshold of total energy), concentration ratio
    (top-1 / sum), and stable rank (sum of squares / max squared).

    Returns None for 1D tensors (biases) where SVD is not meaningful.
    """
    diff = trained.float() - base.float()

    # SVD only meaningful for 2D+ tensors
    if diff.dim() < 2:
        return None

    # Reshape to 2D for SVD
    mat = diff.reshape(diff.shape[0], -1)

    # Use truncated SVD for efficiency — we only need top singular values
    k = min(top_k, min(mat.shape))
    try:
        # torch.svd_lowrank is faster than full SVD for large matrices
        _, s, _ = torch.svd_lowrank(mat, q=k)
    except Exception:
        # Fallback to full SVD if lowrank fails
        try:
            _, s, _ = torch.linalg.svd(mat, full_matrices=False)
        except Exception:
            return None

    if s.numel() == 0:
        return None

    s_list = s[:top_k].tolist()
    s_sum = s.sum().item()
    s_sq_sum = (s ** 2).sum().item()
    s_max_sq = (s[0] ** 2).item()

    # Effective rank: how many singular values to capture threshold of energy
    cumulative_energy = torch.cumsum(s ** 2, dim=0) / max(s_sq_sum, 1e-10)
    effective_rank = int((cumulative_energy < energy_threshold).sum().item()) + 1
    effective_rank = min(effective_rank, s.numel())

    # Concentration ratio: how much of total is in top-1
    concentration = s[0].item() / max(s_sum, 1e-10)

    # Stable rank: frobenius^2 / spectral_norm^2
    stable_rank = s_sq_sum / max(s_max_sq, 1e-10)

    return SVDResult(
        top_singular_values=s_list,
        effective_rank=effective_rank,
        concentration_ratio=concentration,
        stable_rank=stable_rank,
    )
