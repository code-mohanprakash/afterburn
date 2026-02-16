"""Pure metric functions for weight comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def l2_norm_diff(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute L2 norm of (trained - base)."""
    diff = trained.float() - base.float()
    return float(torch.linalg.vector_norm(diff.flatten()).item())


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

    return float(torch.dot(b, t).item()) / (float(b_norm.item()) * float(t_norm.item()))


def frobenius_norm_diff(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute Frobenius norm of the difference.

    For 1D tensors, this is equivalent to L2 norm.
    For 2D+ tensors, computes the matrix Frobenius norm.
    """
    diff = trained.float() - base.float()
    if diff.dim() < 2:
        return float(torch.linalg.vector_norm(diff).item())
    return float(torch.linalg.matrix_norm(diff.reshape(diff.shape[0], -1), ord="fro").item())


def relative_change(base: torch.Tensor, trained: torch.Tensor) -> float:
    """Compute relative change: frobenius(diff) / frobenius(base).

    Returns 0.0 if the base tensor is near-zero.
    """
    base_f = base.float()
    if base_f.dim() < 2:
        base_norm = float(torch.linalg.vector_norm(base_f).item())
    else:
        base_norm = float(torch.linalg.matrix_norm(
            base_f.reshape(base_f.shape[0], -1), ord="fro"
        ).item())

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
    # Direction vectors for behavioral analysis
    top_left_vectors: list[list[float]] | None = None   # U[:, :k] columns as nested lists
    top_right_vectors: list[list[float]] | None = None   # V[:k, :] rows as nested lists


def svd_analysis(
    base: torch.Tensor,
    trained: torch.Tensor,
    top_k: int = 10,
    energy_threshold: float = 0.9,
    return_vectors: bool = False,
) -> SVDResult | None:
    """Compute SVD of the weight difference matrix.

    Returns top-k singular values, effective rank (number of singular values
    needed to capture energy_threshold of total energy), concentration ratio
    (top-1 / sum), and stable rank (sum of squares / max squared).

    Args:
        base: Base model weight tensor
        trained: Trained model weight tensor
        top_k: Number of top singular values to return
        energy_threshold: Energy threshold for effective rank calculation
        return_vectors: If True, also return U and V matrices from SVD

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
    u_mat = None
    v_mat = None
    try:
        # torch.svd_lowrank is faster than full SVD for large matrices
        u_mat, s, v_mat = torch.svd_lowrank(mat, q=k)
    except (torch.linalg.LinAlgError, RuntimeError) as e:
        # Fallback to full SVD if lowrank fails
        logger.warning("svd_lowrank failed (%s), trying full SVD", e)
        try:
            u_mat, s, v_mat = torch.linalg.svd(mat, full_matrices=False)
        except torch.linalg.LinAlgError as e:
            logger.warning("LinAlgError in full SVD: %s", e)
            return None
        except RuntimeError as e:
            logger.warning("RuntimeError in full SVD (possible CUDA error): %s", e)
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

    # Extract vectors if requested
    top_left_vecs = None
    top_right_vecs = None
    if return_vectors and u_mat is not None and v_mat is not None:
        # U[:, :top_k] columns
        top_left_vecs = u_mat[:, :top_k].tolist()
        # V from torch.svd_lowrank is [n, k], transpose to get [k, n] rows
        top_right_vecs = v_mat[:, :top_k].T.tolist()

    return SVDResult(
        top_singular_values=s_list,
        effective_rank=effective_rank,
        concentration_ratio=concentration,
        stable_rank=stable_rank,
        top_left_vectors=top_left_vecs,
        top_right_vectors=top_right_vecs,
    )


def compute_direction_coherence(
    layer_vectors: dict[str, list[list[float]]],
) -> float:
    """Measure cross-layer coherence of principal change directions.

    Computes average pairwise absolute cosine similarity between the top-1
    right singular vector across all layer pairs. High coherence (>0.5)
    suggests a systematic directional shift.

    Args:
        layer_vectors: Mapping from layer_name to list of right singular vectors.
                      Each vector is a list[float]. We use the first vector (top-1).

    Returns:
        Coherence score in [0, 1]. Returns 0.0 if fewer than 2 layers.
    """
    if len(layer_vectors) < 2:
        return 0.0

    # Extract top-1 vector from each layer
    layer_names = list(layer_vectors.keys())
    top_vectors: list[tuple[str, list[float]]] = []

    for name in layer_names:
        vectors = layer_vectors[name]
        if vectors and len(vectors) > 0:
            top_vectors.append((name, vectors[0]))

    if len(top_vectors) < 2:
        return 0.0

    # Compute pairwise absolute cosine similarities
    similarities: list[float] = []

    for i in range(len(top_vectors)):
        for j in range(i + 1, len(top_vectors)):
            name_i, vec_i = top_vectors[i]
            name_j, vec_j = top_vectors[j]

            # Skip if different dimensions
            if len(vec_i) != len(vec_j):
                continue

            # Compute cosine similarity
            dot_product = sum(a * b for a, b in zip(vec_i, vec_j, strict=False))
            norm_i = sum(x ** 2 for x in vec_i) ** 0.5
            norm_j = sum(x ** 2 for x in vec_j) ** 0.5

            if norm_i < 1e-10 or norm_j < 1e-10:
                continue

            cosine_sim = dot_product / (norm_i * norm_j)
            # Take absolute value for coherence measure
            similarities.append(abs(cosine_sim))

    if not similarities:
        return 0.0

    # Return average similarity
    return sum(similarities) / len(similarities)
