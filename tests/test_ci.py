"""Tests for confidence interval computation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from afterburn.ci import benjamini_hochberg, bootstrap_ci, cohens_d_ci, wilson_ci
from afterburn.types import ConfidenceInterval


class TestConfidenceInterval:
    """Tests for the ConfidenceInterval dataclass."""

    def test_basic_creation(self):
        ci = ConfidenceInterval(lower=0.1, upper=0.9)
        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.confidence == 0.95

    def test_custom_confidence(self):
        ci = ConfidenceInterval(lower=0.2, upper=0.8, confidence=0.99)
        assert ci.confidence == 0.99

    def test_width(self):
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        assert abs(ci.width - 0.6) < 1e-10

    def test_frozen(self):
        ci = ConfidenceInterval(lower=0.1, upper=0.9)
        with pytest.raises(AttributeError):
            ci.lower = 0.5  # type: ignore[misc]


class TestCohensD_CI:
    """Tests for Cohen's d confidence interval."""

    def test_identical_groups(self):
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = cohens_d_ci(g1, g2)
        # d = 0, CI should span zero
        assert lo <= 0 <= hi

    def test_clearly_different_groups(self):
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        lo, hi = cohens_d_ci(g1, g2)
        # Large difference, CI should not include zero
        assert lo > 0

    def test_returns_zeros_for_tiny_groups(self):
        g1 = np.array([1.0])
        g2 = np.array([2.0])
        lo, hi = cohens_d_ci(g1, g2)
        assert lo == 0.0
        assert hi == 0.0

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 50)
        g2 = rng.normal(0.5, 1, 50)
        lo, hi = cohens_d_ci(g1, g2)
        # Compute point estimate
        n1, n2 = len(g1), len(g2)
        pooled = np.sqrt(((n1-1)*np.var(g1,ddof=1) + (n2-1)*np.var(g2,ddof=1)) / (n1+n2-2))
        d = float((np.mean(g2) - np.mean(g1)) / pooled)
        assert lo <= d <= hi

    def test_wider_ci_with_smaller_sample(self):
        rng = np.random.default_rng(42)
        g1_small = rng.normal(0, 1, 5)
        g2_small = rng.normal(0.5, 1, 5)
        lo_s, hi_s = cohens_d_ci(g1_small, g2_small)
        width_small = hi_s - lo_s

        g1_large = rng.normal(0, 1, 200)
        g2_large = rng.normal(0.5, 1, 200)
        lo_l, hi_l = cohens_d_ci(g1_large, g2_large)
        width_large = hi_l - lo_l

        assert width_small > width_large

    def test_zero_variance_groups(self):
        g1 = np.array([5.0, 5.0, 5.0])
        g2 = np.array([5.0, 5.0, 5.0])
        lo, hi = cohens_d_ci(g1, g2)
        assert lo == 0.0
        assert hi == 0.0


class TestWilsonCI:
    """Tests for Wilson score interval."""

    def test_zero_out_of_zero(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_all_successes(self):
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.9
        assert hi == 1.0

    def test_no_successes(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.1

    def test_half_successes(self):
        lo, hi = wilson_ci(50, 100)
        assert lo < 0.5
        assert hi > 0.5

    def test_bounds_respected(self):
        lo, hi = wilson_ci(1, 2)
        assert lo >= 0.0
        assert hi <= 1.0

    def test_small_sample(self):
        lo, hi = wilson_ci(1, 3)
        # Wilson should give wider interval than naive CI
        assert hi - lo > 0.2

    def test_ci_narrows_with_larger_sample(self):
        lo_s, hi_s = wilson_ci(5, 10)
        lo_l, hi_l = wilson_ci(50, 100)
        assert (hi_s - lo_s) > (hi_l - lo_l)


class TestBootstrapCI:
    """Tests for bootstrap percentile CI."""

    def test_basic_mean(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        lo, hi = bootstrap_ci(data, statistic="mean")
        assert lo < 5.5
        assert hi > 5.5

    def test_basic_median(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        lo, hi = bootstrap_ci(data, statistic="median")
        assert lo < 5.5
        assert hi > 5.5

    def test_insufficient_data(self):
        data = np.array([1.0])
        lo, hi = bootstrap_ci(data)
        assert lo == 0.0
        assert hi == 0.0

    def test_reproducible_with_seed(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo1, hi1 = bootstrap_ci(data, seed=42)
        lo2, hi2 = bootstrap_ci(data, seed=42)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seeds_give_different_results(self):
        data = np.arange(100, dtype=np.float64)
        lo1, hi1 = bootstrap_ci(data, seed=42)
        lo2, hi2 = bootstrap_ci(data, seed=99)
        # Very unlikely to be identical with different seeds
        assert lo1 != lo2 or hi1 != hi2

    def test_unknown_statistic_raises(self):
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown statistic"):
            bootstrap_ci(data, statistic="invalid")

    def test_narrower_with_more_data(self):
        rng = np.random.default_rng(42)
        small = rng.normal(0, 1, 10)
        large = rng.normal(0, 1, 500)
        lo_s, hi_s = bootstrap_ci(small)
        lo_l, hi_l = bootstrap_ci(large)
        assert (hi_s - lo_s) > (hi_l - lo_l)


class TestCIInLengthAnalysis:
    """Test that LengthAnalysis includes CI fields."""

    def test_length_analysis_has_ci(self):
        from afterburn.behaviour.length_analysis import analyze_length_distribution
        from afterburn.types import PromptResult

        rng = np.random.default_rng(42)
        base = [
            PromptResult(
                prompt_id=f"b{i}", prompt_text="q", category="math",
                output_text="a", output_tokens=int(rng.normal(50, 10)),
                generation_time_ms=10.0,
            ) for i in range(20)
        ]
        trained = [
            PromptResult(
                prompt_id=f"t{i}", prompt_text="q", category="math",
                output_text="a", output_tokens=int(rng.normal(80, 10)),
                generation_time_ms=10.0,
            ) for i in range(20)
        ]

        result = analyze_length_distribution(base, trained)
        assert result.cohens_d_ci is not None
        assert result.mean_diff_ci is not None
        assert result.cohens_d_ci.lower < result.cohens_d_ci.upper

    def test_length_analysis_ci_contains_point(self):
        from afterburn.behaviour.length_analysis import analyze_length_distribution
        from afterburn.types import PromptResult

        rng = np.random.default_rng(42)
        base = [
            PromptResult(
                prompt_id=f"b{i}", prompt_text="q", category="math",
                output_text="a", output_tokens=int(rng.normal(50, 10)),
                generation_time_ms=10.0,
            ) for i in range(30)
        ]
        trained = [
            PromptResult(
                prompt_id=f"t{i}", prompt_text="q", category="math",
                output_text="a", output_tokens=int(rng.normal(80, 10)),
                generation_time_ms=10.0,
            ) for i in range(30)
        ]

        result = analyze_length_distribution(base, trained)
        # Point estimate should be within CI
        assert result.cohens_d_ci.lower <= result.cohens_d <= result.cohens_d_ci.upper


class TestCIInLengthBias:
    """Test that LengthBiasResult includes CI fields."""

    def test_length_bias_has_ci(self):
        from afterburn.reward_hack.length_bias import detect_length_bias
        from afterburn.types import PromptResult

        base = [
            PromptResult(
                prompt_id=f"b{i}", prompt_text="q", category="math",
                output_text="a" * 50, output_tokens=50, generation_time_ms=10.0,
            ) for i in range(20)
        ]
        trained = [
            PromptResult(
                prompt_id=f"t{i}", prompt_text="q", category="math",
                output_text="a" * 80, output_tokens=80, generation_time_ms=10.0,
            ) for i in range(20)
        ]

        result = detect_length_bias(base, trained)
        assert result.cohens_d_ci is not None
        assert result.cohens_d_ci.lower <= result.cohens_d <= result.cohens_d_ci.upper


class TestCIInSycophancy:
    """Test that SycophancyResult includes CI fields."""

    def test_sycophancy_has_rate_cis(self):
        from afterburn.reward_hack.sycophancy import detect_sycophancy
        from afterburn.types import PromptResult

        base = [
            PromptResult(
                prompt_id=f"b{i}", prompt_text="q", category="safety",
                output_text="I disagree with that claim." if i % 2 else "Yes, you're right.",
                output_tokens=20, generation_time_ms=10.0,
            ) for i in range(20)
        ]
        trained = [
            PromptResult(
                prompt_id=f"t{i}", prompt_text="q", category="safety",
                output_text="Absolutely! You're correct." if i < 15 else "I disagree.",
                output_tokens=20, generation_time_ms=10.0,
            ) for i in range(20)
        ]

        result = detect_sycophancy(base, trained)
        assert result.agreement_rate_ci is not None
        assert result.pushback_rate_ci is not None
        assert 0.0 <= result.agreement_rate_ci.lower <= result.agreement_rate_ci.upper <= 1.0


class TestCIInStrategyCollapse:
    """Test that StrategyCollapseResult includes CI field."""

    def test_strategy_collapse_has_ci(self):
        from afterburn.reward_hack.strategy_collapse import detect_strategy_collapse
        from afterburn.types import PromptResult

        # Base model: diverse strategies
        base = [
            PromptResult(
                prompt_id=f"b{i}", prompt_text="Solve this",
                category="math",
                output_text=(
                    "Let me think step by step..." if i % 3 == 0
                    else "```python\nprint(42)\n```" if i % 3 == 1
                    else "The answer is 42."
                ),
                output_tokens=50, generation_time_ms=10.0,
            ) for i in range(30)
        ]
        # Trained model: mostly one strategy
        trained = [
            PromptResult(
                prompt_id=f"t{i}", prompt_text="Solve this",
                category="math",
                output_text="```python\nprint(42)\n```",
                output_tokens=50, generation_time_ms=10.0,
            ) for i in range(30)
        ]

        result = detect_strategy_collapse(base, trained)
        assert result.entropy_drop_ci is not None
        assert result.entropy_drop_ci.lower <= result.entropy_drop_ci.upper


class TestBenjaminiHochberg:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_empty_input(self):
        assert benjamini_hochberg([]) == []

    def test_single_significant(self):
        result = benjamini_hochberg([0.01], alpha=0.05)
        assert len(result) == 1
        assert result[0][0] == 0.01  # No adjustment for single test
        assert result[0][1] is True

    def test_single_not_significant(self):
        result = benjamini_hochberg([0.10], alpha=0.05)
        assert len(result) == 1
        assert result[0][0] == 0.10
        assert result[0][1] is False

    def test_multiple_all_significant(self):
        result = benjamini_hochberg([0.001, 0.005, 0.01], alpha=0.05)
        assert len(result) == 3
        # All should remain significant after correction
        assert all(sig for _, sig in result)

    def test_adjusted_p_values_bounded_by_one(self):
        result = benjamini_hochberg([0.5, 0.8, 0.99])
        for adj_p, _ in result:
            assert adj_p <= 1.0

    def test_adjusted_p_values_geq_raw(self):
        raw = [0.01, 0.04, 0.03, 0.08]
        result = benjamini_hochberg(raw)
        for i, (adj_p, _) in enumerate(result):
            assert adj_p >= raw[i] or abs(adj_p - raw[i]) < 1e-10

    def test_preserves_order(self):
        """Results should be in same order as input."""
        raw = [0.05, 0.01, 0.10]
        result = benjamini_hochberg(raw)
        assert len(result) == 3
        # The smallest raw p-value (index 1) should have the smallest adjusted p
        assert result[1][0] <= result[0][0]

    def test_monotonicity_enforcement(self):
        """Adjusted p-values should be non-decreasing when sorted by raw p-value."""
        raw = [0.01, 0.04, 0.03, 0.08, 0.50]
        result = benjamini_hochberg(raw)
        # Sort by raw p-value and check monotonicity of adjusted
        sorted_pairs = sorted(zip(raw, result), key=lambda x: x[0])
        adj_sorted = [adj for _, (adj, _) in sorted_pairs]
        for i in range(1, len(adj_sorted)):
            assert adj_sorted[i] >= adj_sorted[i - 1] - 1e-10

    def test_some_rejected_some_not(self):
        raw = [0.001, 0.01, 0.30, 0.80]
        result = benjamini_hochberg(raw, alpha=0.05)
        # First two should be significant, last two should not
        assert result[0][1] is True
        assert result[1][1] is True
        assert result[3][1] is False

    def test_identical_p_values(self):
        raw = [0.03, 0.03, 0.03]
        result = benjamini_hochberg(raw, alpha=0.05)
        assert len(result) == 3
        # All same input → all same output
        adj_values = [adj for adj, _ in result]
        assert all(abs(a - adj_values[0]) < 1e-10 for a in adj_values)

    def test_corrected_p_in_length_analysis(self):
        """Test that BH correction integrates into BehaviourAnalyser."""
        from afterburn.types import LengthAnalysis

        la = LengthAnalysis(
            base_mean=50.0, base_median=50.0, base_std=10.0,
            trained_mean=80.0, trained_median=80.0, trained_std=10.0,
            mean_diff=30.0, p_value=0.001, cohens_d=2.5,
            is_significant=True,
        )
        # Simulate what the analyser does
        corrected = benjamini_hochberg([la.p_value], alpha=0.05)
        la.corrected_p_value = corrected[0][0]
        assert la.corrected_p_value == la.p_value  # Single test: no change
        assert corrected[0][1] is True

    def test_corrected_p_in_length_bias(self):
        """Test that BH correction integrates into RewardHackDetector."""
        from afterburn.types import LengthBiasResult

        lb = LengthBiasResult(
            score=75.0, cohens_d=1.2, p_value=0.002,
            mean_length_ratio=1.5, is_flagged=True,
        )
        corrected = benjamini_hochberg([lb.p_value], alpha=0.05)
        lb.corrected_p_value = corrected[0][0]
        assert lb.corrected_p_value == lb.p_value  # Single test: no change
        assert corrected[0][1] is True
