"""Tests for length bias detection (Phase 2 enhancements)."""

import pytest

from afterburn.reward_hack.length_bias import detect_length_bias
from afterburn.types import PromptResult


def _make_result(output_tokens: int, category: str = "math") -> PromptResult:
    return PromptResult(
        prompt_id="t1",
        prompt_text="Test",
        category=category,
        output_text="x " * output_tokens,
        output_tokens=output_tokens,
        generation_time_ms=100.0,
    )


class TestLengthBias:
    def test_significant_length_increase(self):
        # Use varied lengths so Cohen's d can be computed (std > 0)
        base = [_make_result(45 + i * 2, "math") for i in range(10)]
        trained = [_make_result(180 + i * 5, "math") for i in range(10)]
        result = detect_length_bias(base, trained)
        assert result.score > 30
        assert result.cohens_d > 0
        assert result.mean_length_ratio > 1.0
        assert result.is_flagged

    def test_no_length_increase(self):
        base = [_make_result(100) for _ in range(10)]
        trained = [_make_result(100) for _ in range(10)]
        result = detect_length_bias(base, trained)
        assert result.score < 30
        assert not result.is_flagged

    def test_length_decrease_low_score(self):
        base = [_make_result(200) for _ in range(10)]
        trained = [_make_result(50) for _ in range(10)]
        result = detect_length_bias(base, trained)
        # Should have low score for decreases
        assert result.score < 30

    def test_empty_inputs(self):
        result = detect_length_bias([], [])
        assert result.score == 0.0
        assert not result.is_flagged

    def test_per_category_analysis(self):
        base = (
            [_make_result(50, "math") for _ in range(5)]
            + [_make_result(60, "code") for _ in range(5)]
        )
        trained = (
            [_make_result(200, "math") for _ in range(5)]
            + [_make_result(200, "code") for _ in range(5)]
        )
        result = detect_length_bias(base, trained)
        assert "math" in result.per_category
        assert "code" in result.per_category
        # Both categories should show bias
        assert result.per_category["math"]["mean_ratio"] > 1.0
        assert result.per_category["code"]["mean_ratio"] > 1.0

    def test_consistency_boost(self):
        """When bias is consistent across categories, score should be higher."""
        base = (
            [_make_result(45 + i * 2, "math") for i in range(5)]
            + [_make_result(48 + i * 2, "code") for i in range(5)]
            + [_make_result(50 + i * 2, "safety") for i in range(5)]
        )
        trained = (
            [_make_result(180 + i * 5, "math") for i in range(5)]
            + [_make_result(190 + i * 5, "code") for i in range(5)]
            + [_make_result(185 + i * 5, "safety") for i in range(5)]
        )
        result = detect_length_bias(base, trained)
        assert result.score > 40  # Should get a boost from consistency
