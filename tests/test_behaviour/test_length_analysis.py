"""Tests for length distribution analysis."""

import pytest

from afterburn.behaviour.length_analysis import analyze_length_distribution
from afterburn.types import PromptResult


def _make_results(lengths: list[int], category: str = "test") -> list[PromptResult]:
    return [
        PromptResult(
            prompt_id=f"p{i}",
            prompt_text=f"Prompt {i}",
            category=category,
            output_text="x " * length,
            output_tokens=length,
            generation_time_ms=100.0,
        )
        for i, length in enumerate(lengths)
    ]


class TestLengthDistribution:
    def test_identical_distributions(self):
        results = _make_results([100] * 10)
        analysis = analyze_length_distribution(results, results)
        assert analysis.cohens_d == pytest.approx(0.0, abs=0.1)
        assert analysis.mean_diff == pytest.approx(0.0, abs=0.1)
        assert not analysis.is_significant

    def test_significant_difference(self):
        base = _make_results([50, 55, 60, 45, 52, 58, 47, 53, 49, 51])
        trained = _make_results([150, 155, 160, 145, 152, 158, 147, 153, 149, 151])
        analysis = analyze_length_distribution(base, trained)
        assert analysis.is_significant
        assert analysis.mean_diff > 0
        assert analysis.cohens_d > 0.5

    def test_empty_inputs(self):
        analysis = analyze_length_distribution([], [])
        assert analysis.base_mean == 0.0
        assert not analysis.is_significant

    def test_per_category(self):
        base = _make_results([50, 60], "math") + _make_results([70, 80], "code")
        trained = _make_results([100, 110], "math") + _make_results([75, 85], "code")
        analysis = analyze_length_distribution(base, trained)
        assert "math" in analysis.per_category
        assert "code" in analysis.per_category
        # Math should show larger increase
        assert analysis.per_category["math"]["mean_diff"] > analysis.per_category["code"]["mean_diff"]
