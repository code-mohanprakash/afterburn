"""Spectral analysis of weight matrices — alpha metric and stable rank.

Inspired by WeightWatcher (Marchenko-Pastur / Heavy-Tailed Self-Regularization).

The alpha metric measures the power-law exponent of the eigenvalue spectrum:
- alpha ~ 2.0: well-generalized layer (heavy-tailed, good capacity)
- alpha ~ 2-4: typical healthy range
- alpha > 6: possible over-fitting or under-training
- alpha < 2: highly heavy-tailed, unusual

Stable rank = ||W||_F^2 / ||W||_2^2 measures the effective dimensionality
of the weight matrix. A higher stable rank means the weight's energy is
more evenly distributed across singular values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectralResult:
    """Spectral analysis of a single weight matrix."""

    alpha: float
    alpha_quality: str  # "good", "fair", "poor", "unstable"
    stable_rank: float
    spectral_norm: float
    log_spectral_norm: float
    num_eigenvalues: int


def spectral_analysis(weight: torch.Tensor, min_size: int = 50) -> SpectralResult | None:
    """Compute spectral metrics for a weight matrix.

    Args:
        weight: A 2D+ weight tensor.
        min_size: Minimum matrix dimension for meaningful analysis.

    Returns:
        SpectralResult or None if the matrix is too small or 1D.
    """
    if weight.dim() < 2:
        return None

    mat = weight.float().reshape(weight.shape[0], -1)
    m, n = mat.shape

    if min(m, n) < min_size:
        return None

    # Compute singular values (more numerically stable than eigenvalues of W^T W)
    try:
        s = torch.linalg.svdvals(mat)
    except Exception:
        return None

    if s.numel() == 0 or s[0].item() < 1e-10:
        return None

    spectral_norm = s[0].item()
    frobenius_sq = (s ** 2).sum().item()
    stable_rank = frobenius_sq / (spectral_norm ** 2)

    # Fit power-law alpha to eigenvalue spectrum
    # eigenvalues = singular_values^2
    eigenvalues = (s ** 2).cpu()
    alpha = _fit_power_law_alpha(eigenvalues)

    # Quality assessment based on alpha
    if 2.0 <= alpha <= 4.0:
        quality = "good"
    elif 1.5 <= alpha < 2.0 or 4.0 < alpha <= 6.0:
        quality = "fair"
    elif alpha > 6.0:
        quality = "poor"
    else:
        quality = "unstable"

    import math

    return SpectralResult(
        alpha=alpha,
        alpha_quality=quality,
        stable_rank=stable_rank,
        spectral_norm=spectral_norm,
        log_spectral_norm=math.log10(max(spectral_norm, 1e-10)),
        num_eigenvalues=s.numel(),
    )


def _fit_power_law_alpha(eigenvalues: torch.Tensor) -> float:
    """Fit power-law exponent to the tail of the eigenvalue spectrum.

    Uses the Hill estimator on the tail of the log-log distribution.
    The Hill estimator is: alpha = n / sum(ln(x_i / x_min))
    where x_i are the tail observations and x_min is the tail threshold.

    We use the top 20% of eigenvalues as the tail (following WeightWatcher convention).
    """
    # Sort descending
    eigs = eigenvalues[eigenvalues > 1e-10].sort(descending=True).values

    if eigs.numel() < 10:
        return 0.0

    # Use top 20% as the tail
    tail_size = max(int(eigs.numel() * 0.2), 5)
    tail = eigs[:tail_size]

    x_min = tail[-1].item()
    if x_min <= 0:
        return 0.0

    # Hill estimator
    log_ratios = torch.log(tail / x_min)
    mean_log = log_ratios.mean().item()

    if mean_log < 1e-10:
        return 0.0

    alpha = 1.0 + 1.0 / mean_log
    # Clamp to reasonable range
    return max(0.5, min(alpha, 20.0))
