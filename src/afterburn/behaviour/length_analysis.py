"""Output length distribution comparison with statistical testing."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import stats

from afterburn.ci import bootstrap_ci, cohens_d_ci
from afterburn.types import ConfidenceInterval, LengthAnalysis, PromptResult


def analyze_length_distribution(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
    significance_level: float = 0.05,
    effect_size_threshold: float = 0.2,
) -> LengthAnalysis:
    """Compare output length distributions between base and trained models.

    Computes:
    - Mean, median, std for each model
    - Skewness and kurtosis for distribution shape analysis
    - Percentiles (p10, p25, p50, p75, p90) for detailed distribution understanding
    - Mann-Whitney U test for significant difference
    - Cohen's d effect size
    - Per-category breakdowns with effect sizes
    """
    base_lengths = np.array([r.output_tokens for r in base_results], dtype=np.float64)
    trained_lengths = np.array([r.output_tokens for r in trained_results], dtype=np.float64)

    if len(base_lengths) == 0 or len(trained_lengths) == 0:
        return LengthAnalysis(
            base_mean=0.0,
            base_median=0.0,
            base_std=0.0,
            trained_mean=0.0,
            trained_median=0.0,
            trained_std=0.0,
            mean_diff=0.0,
            p_value=1.0,
            cohens_d=0.0,
            is_significant=False,
            base_skewness=0.0,
            trained_skewness=0.0,
            base_kurtosis=0.0,
            trained_kurtosis=0.0,
            base_percentiles={},
            trained_percentiles={},
        )

    # Basic statistics
    base_mean = float(np.mean(base_lengths))
    base_median = float(np.median(base_lengths))
    base_std = float(np.std(base_lengths, ddof=1)) if len(base_lengths) > 1 else 0.0
    trained_mean = float(np.mean(trained_lengths))
    trained_median = float(np.median(trained_lengths))
    trained_std = float(np.std(trained_lengths, ddof=1)) if len(trained_lengths) > 1 else 0.0

    mean_diff = trained_mean - base_mean

    # Skewness and kurtosis for distribution shape analysis
    # Skewness: measures asymmetry (0 = symmetric, >0 = right tail, <0 = left tail)
    # Kurtosis: measures tail heaviness (0 = normal, >0 = heavy tails, <0 = light tails)
    base_skewness = _compute_skewness(base_lengths)
    trained_skewness = _compute_skewness(trained_lengths)
    base_kurtosis = _compute_kurtosis(base_lengths)
    trained_kurtosis = _compute_kurtosis(trained_lengths)

    # Percentile analysis for detailed distribution understanding
    base_percentiles = _compute_percentiles(base_lengths)
    trained_percentiles = _compute_percentiles(trained_lengths)

    # Mann-Whitney U test (non-parametric, doesn't assume normality)
    if len(base_lengths) >= 2 and len(trained_lengths) >= 2:
        _, p_value = stats.mannwhitneyu(
            base_lengths, trained_lengths, alternative="two-sided"
        )
        p_value = float(p_value)
    else:
        p_value = 1.0

    # Cohen's d effect size
    cohens_d = _compute_cohens_d(base_lengths, trained_lengths)

    # Confidence intervals
    d_lower, d_upper = cohens_d_ci(base_lengths, trained_lengths)
    ci_cohens_d = ConfidenceInterval(lower=d_lower, upper=d_upper)

    diff_lower, diff_upper = bootstrap_ci(
        trained_lengths - base_lengths[:len(trained_lengths)]
        if len(base_lengths) == len(trained_lengths)
        else np.concatenate([trained_lengths, -base_lengths]),
        statistic="mean",
    )
    ci_mean_diff = ConfidenceInterval(lower=diff_lower, upper=diff_upper)

    # Per-category analysis with enhanced metrics
    per_category = _per_category_analysis(base_results, trained_results)

    is_significant = p_value < significance_level and abs(cohens_d) > effect_size_threshold

    return LengthAnalysis(
        base_mean=base_mean,
        base_median=base_median,
        base_std=base_std,
        trained_mean=trained_mean,
        trained_median=trained_median,
        trained_std=trained_std,
        mean_diff=mean_diff,
        p_value=p_value,
        cohens_d=cohens_d,
        is_significant=is_significant,
        per_category=per_category,
        base_skewness=base_skewness,
        trained_skewness=trained_skewness,
        base_kurtosis=base_kurtosis,
        trained_kurtosis=trained_kurtosis,
        base_percentiles=base_percentiles,
        trained_percentiles=trained_percentiles,
        cohens_d_ci=ci_cohens_d,
        mean_diff_ci=ci_mean_diff,
    )


def _compute_skewness(data: object) -> float:  # type: ignore[misc]  # type: ignore[type-arg]
    """Compute skewness using scipy.stats.skew.

    Skewness measures the asymmetry of the distribution:
    - 0: symmetric distribution
    - > 0: right-skewed (long tail to the right)
    - < 0: left-skewed (long tail to the left)

    Requires at least 3 data points for meaningful computation.
    """
    if len(data) < 3:  # type: ignore[arg-type]
        return 0.0

    # bias=True returns the Fisher-Pearson coefficient (default)
    # This is the standard definition used in statistics
    return float(stats.skew(data, bias=True))  # type: ignore[arg-type]


def _compute_kurtosis(data: object) -> float:  # type: ignore[misc]
    """Compute kurtosis using scipy.stats.kurtosis.

    Kurtosis measures the "tailedness" of the distribution:
    - 0: normal distribution (excess kurtosis definition)
    - > 0: heavy tails (leptokurtic) - more extreme values
    - < 0: light tails (platykurtic) - fewer extreme values

    Requires at least 4 data points for meaningful computation.
    """
    if len(data) < 4:  # type: ignore[arg-type]
        return 0.0

    # fisher=True returns excess kurtosis (kurtosis - 3)
    # This makes normal distribution have kurtosis of 0
    return float(stats.kurtosis(data, fisher=True, bias=True))  # type: ignore[arg-type]


def _compute_percentiles(data: object) -> dict[str, float]:  # type: ignore[misc]
    """Compute key percentiles for distribution analysis.

    Returns percentiles at 10th, 25th, 50th (median), 75th, and 90th.
    These provide a comprehensive view of the distribution spread.
    """
    if len(data) == 0:  # type: ignore[arg-type]
        return {}

    percentiles = np.percentile(data, [10, 25, 50, 75, 90])  # type: ignore[call-overload]

    return {
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
    }


def _compute_cohens_d(
    group1: object, group2: object
) -> float:  # type: ignore[misc]
    """Compute Cohen's d effect size.

    Effect size interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)  # type: ignore[arg-type]
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = np.var(group1, ddof=1)  # type: ignore[call-overload]
    var2 = np.var(group2, ddof=1)  # type: ignore[call-overload]

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float((np.mean(group2) - np.mean(group1)) / pooled_std)  # type: ignore[call-overload]


def _per_category_analysis(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
) -> dict[str, dict[str, float]]:
    """Compute per-category length statistics with Cohen's d effect sizes.

    For each category, computes:
    - base_mean: average length for base model
    - trained_mean: average length for trained model
    - mean_diff: trained_mean - base_mean
    - cohens_d: effect size of the difference
    """
    base_by_cat: dict[str, list[int]] = defaultdict(list)
    trained_by_cat: dict[str, list[int]] = defaultdict(list)

    for r in base_results:
        base_by_cat[r.category].append(r.output_tokens)
    for r in trained_results:
        trained_by_cat[r.category].append(r.output_tokens)

    categories = set(base_by_cat.keys()) | set(trained_by_cat.keys())
    result: dict[str, dict[str, float]] = {}

    for cat in sorted(categories):
        base_lens = base_by_cat.get(cat, [])
        trained_lens = trained_by_cat.get(cat, [])

        base_mean = float(np.mean(base_lens)) if base_lens else 0.0
        trained_mean = float(np.mean(trained_lens)) if trained_lens else 0.0
        mean_diff = trained_mean - base_mean if base_lens and trained_lens else 0.0

        # Compute Cohen's d for this category
        base_arr = np.array(base_lens, dtype=np.float64)
        trained_arr = np.array(trained_lens, dtype=np.float64)
        cohens_d = _compute_cohens_d(base_arr, trained_arr)

        result[cat] = {
            "base_mean": base_mean,
            "trained_mean": trained_mean,
            "mean_diff": mean_diff,
            "cohens_d": cohens_d,
        }

    return result
