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

import logging
import math
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectralResult:
    """Spectral analysis of a single weight matrix."""

    alpha: float
    alpha_quality: str  # "good", "fair", "poor", "unstable"
    stable_rank: float
    spectral_norm: float
    log_spectral_norm: float
    num_eigenvalues: int


@dataclass(frozen=True)
class MPResult:
    """Marchenko-Pastur law comparison for a weight matrix."""

    sigma_sq: float  # Fitted noise variance
    lambda_plus: float  # MP upper edge: sigma^2 * (1 + sqrt(gamma))^2
    lambda_minus: float  # MP lower edge: sigma^2 * (1 - sqrt(gamma))^2
    gamma: float  # Aspect ratio: rows / cols
    num_spikes: int  # Eigenvalues above lambda_plus (learned structure)
    spike_eigenvalues: list[float]  # The actual spike values
    bulk_fraction: float  # Fraction of eigenvalues within MP bulk
    kl_divergence: float  # KL(empirical || MP) on the bulk — lower = more random


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
    except torch.linalg.LinAlgError as e:
        logger.warning("LinAlgError computing SVD for spectral analysis: %s", e)
        return None
    except RuntimeError as e:
        logger.warning("RuntimeError in spectral SVD: %s", e)
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


def marchenko_pastur_fit(weight: torch.Tensor, min_size: int = 50) -> MPResult | None:
    """Fit Marchenko-Pastur law to a weight matrix to detect random vs. learned structure.

    The Marchenko-Pastur (MP) law describes the eigenvalue distribution of a random
    matrix with i.i.d. Gaussian entries. Deviations from this law (especially eigenvalues
    above the upper edge lambda_plus) indicate learned structure.

    Args:
        weight: A 2D+ weight tensor.
        min_size: Minimum matrix dimension for meaningful analysis.

    Returns:
        MPResult or None if the matrix is too small, 1D, or all zeros.
    """
    if weight.dim() < 2:
        return None

    mat = weight.float().reshape(weight.shape[0], -1)
    m, n = mat.shape

    if min(m, n) < min_size:
        return None

    # Compute singular values
    try:
        s = torch.linalg.svdvals(mat)
    except torch.linalg.LinAlgError as e:
        logger.warning("LinAlgError computing SVD for Marchenko-Pastur analysis: %s", e)
        return None
    except RuntimeError as e:
        logger.warning("RuntimeError in MP SVD: %s", e)
        return None

    if s.numel() == 0 or s.max().item() < 1e-10:
        return None

    # Eigenvalues = singular_values^2
    eigenvalues = (s**2).cpu()

    # Aspect ratio
    gamma = float(m) / float(n)
    sqrt_gamma = math.sqrt(gamma)

    # For Marchenko-Pastur law, we normalize by the number of columns
    # The eigenvalues of (1/sqrt(n)) * X for X with i.i.d. N(0, sigma^2) entries
    # follow MP law with these parameters
    # We estimate sigma^2 from the matrix variance directly
    sigma_sq = (mat.var().item()) * n  # Variance of entries × n

    if sigma_sq <= 0 or math.isnan(sigma_sq):
        return None

    # Marchenko-Pastur edges
    lambda_plus = sigma_sq * (1.0 + sqrt_gamma) ** 2
    lambda_minus = max(0.0, sigma_sq * (1.0 - sqrt_gamma) ** 2)

    # Count spikes (eigenvalues above upper edge)
    spikes_mask = eigenvalues > lambda_plus
    num_spikes = spikes_mask.sum().item()
    spike_eigenvalues = eigenvalues[spikes_mask].tolist()

    # Bulk fraction (eigenvalues within MP bulk)
    bulk_mask = (eigenvalues >= lambda_minus) & (eigenvalues <= lambda_plus)
    total_eigenvalues = eigenvalues.numel()
    bulk_fraction = bulk_mask.sum().item() / total_eigenvalues

    # KL divergence between empirical bulk distribution and theoretical MP distribution
    kl_divergence = _compute_mp_kl_divergence(
        eigenvalues, lambda_minus, lambda_plus, sigma_sq, gamma
    )

    return MPResult(
        sigma_sq=sigma_sq,
        lambda_plus=lambda_plus,
        lambda_minus=lambda_minus,
        gamma=gamma,
        num_spikes=num_spikes,
        spike_eigenvalues=spike_eigenvalues,
        bulk_fraction=bulk_fraction,
        kl_divergence=kl_divergence,
    )


def _compute_mp_kl_divergence(
    eigenvalues: torch.Tensor,
    lambda_minus: float,
    lambda_plus: float,
    sigma_sq: float,
    gamma: float,
) -> float:
    """Compute KL(empirical || theoretical MP) on the bulk eigenvalues.

    Lower KL divergence indicates the matrix is more random (closer to MP law).
    Higher KL divergence indicates more structure/learning.
    """
    # Extract bulk eigenvalues
    bulk_mask = (eigenvalues >= lambda_minus) & (eigenvalues <= lambda_plus)
    bulk_eigs = eigenvalues[bulk_mask]

    if bulk_eigs.numel() == 0:
        return 0.0

    # Create histogram of bulk eigenvalues
    num_bins = min(50, max(20, bulk_eigs.numel() // 10))
    counts, bin_edges = torch.histogram(
        bulk_eigs.float(), bins=num_bins, range=(lambda_minus, lambda_plus)
    )

    # Normalize to get empirical probabilities
    empirical_probs = counts.float() / counts.sum()

    # Compute theoretical MP PDF at bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    theoretical_probs = _mp_pdf(bin_centers, lambda_minus, lambda_plus, sigma_sq, gamma)

    # Normalize theoretical probabilities
    theoretical_probs = theoretical_probs / theoretical_probs.sum()

    # Compute KL divergence: KL(P || Q) = sum(P * log(P / Q))
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    empirical_probs = empirical_probs + eps
    theoretical_probs = theoretical_probs + eps

    kl = (empirical_probs * torch.log(empirical_probs / theoretical_probs)).sum().item()

    return max(0.0, kl)


def _mp_pdf(
    lambda_vals: torch.Tensor,
    lambda_minus: float,
    lambda_plus: float,
    sigma_sq: float,
    gamma: float,
) -> torch.Tensor:
    """Marchenko-Pastur probability density function.

    rho(l) = sqrt((l+ - l)(l - l-)) / (2*pi*sigma^2*gamma*l)

    for lambda in [lambda_minus, lambda_plus], 0 otherwise.
    """
    result = torch.zeros_like(lambda_vals)

    # Only compute for values within the MP support
    valid_mask = (lambda_vals >= lambda_minus) & (lambda_vals <= lambda_plus)

    if valid_mask.sum() == 0:
        return result

    valid_lambdas = lambda_vals[valid_mask]

    # Avoid division by zero
    valid_lambdas = torch.clamp(valid_lambdas, min=1e-10)

    numerator = torch.sqrt(
        (lambda_plus - valid_lambdas) * (valid_lambdas - lambda_minus)
    )
    denominator = 2.0 * math.pi * sigma_sq * gamma * valid_lambdas

    result[valid_mask] = numerator / denominator

    return result
