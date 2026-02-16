"""Confidence interval and multiple testing correction utilities.

Provides:
- Non-central t-distribution CI for Cohen's d
- Wilson score interval for proportions
- Bootstrap percentile CI for arbitrary statistics
- Benjamini-Hochberg FDR correction for multiple hypothesis testing
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def cohens_d_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute CI for Cohen's d using non-central t-distribution.

    Based on Hedges & Olkin (1985) approximation.

    Returns:
        (lower, upper) bounds of the CI.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return (0.0, 0.0)

    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return (0.0, 0.0)

    d = float((np.mean(group2) - np.mean(group1)) / pooled_std)
    df = n1 + n2 - 2

    # Standard error of Cohen's d (Hedges & Olkin approx)
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * df))

    alpha = 1 - confidence
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))

    lower = d - t_crit * se_d
    upper = d + t_crit * se_d
    return (round(lower, 6), round(upper, 6))


def wilson_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than the normal approximation, especially for small n
    or extreme proportions.

    Returns:
        (lower, upper) bounds of the CI.
    """
    if total == 0:
        return (0.0, 0.0)

    p_hat = successes / total
    alpha = 1 - confidence
    z = float(stats.norm.ppf(1 - alpha / 2))
    z2 = z * z

    denom = 1 + z2 / total
    centre = p_hat + z2 / (2 * total)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * total)) / total)

    lower = max(0.0, (centre - margin) / denom)
    upper = min(1.0, (centre + margin) / denom)
    return (round(lower, 6), round(upper, 6))


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Bootstrap percentile CI for an arbitrary statistic.

    Args:
        data: 1D array of observations.
        statistic: One of "mean", "median", or a callable.
        confidence: Confidence level (0-1).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the CI.
    """
    if len(data) < 2:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(data)

    stat_fn = {
        "mean": np.mean,
        "median": np.median,
    }.get(statistic, statistic)

    if not callable(stat_fn):
        msg = f"Unknown statistic: {statistic}"
        raise ValueError(msg)

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (round(lower, 6), round(upper, 6))


def benjamini_hochberg(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Benjamini-Hochberg FDR correction for multiple hypothesis testing.

    Controls the false discovery rate (FDR) — the expected proportion of
    false positives among all rejected hypotheses. Preferred over Bonferroni
    when testing many hypotheses, as it's less conservative.

    Args:
        p_values: List of raw p-values from independent tests.
        alpha: Desired FDR level (default 0.05).

    Returns:
        List of (adjusted_p_value, is_significant) tuples, in the same
        order as the input p_values.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])

    # Compute adjusted p-values (Benjamini-Hochberg step-up)
    adjusted = [0.0] * m
    for rank, idx in enumerate(sorted_indices, 1):
        adjusted[idx] = p_values[idx] * m / rank

    # Enforce monotonicity: adjusted p-values should be non-decreasing
    # when sorted by original p-values (step-up enforcement)
    prev = 1.0
    for idx in reversed(sorted_indices):
        adjusted[idx] = min(adjusted[idx], prev, 1.0)
        prev = adjusted[idx]

    return [(adj, adj < alpha) for adj in adjusted]
