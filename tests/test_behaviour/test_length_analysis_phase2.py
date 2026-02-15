"""Tests for Phase 2 length analysis enhancements (skewness, kurtosis, percentiles)."""

import pytest

from afterburn.behaviour.length_analysis import (
    _compute_kurtosis,
    _compute_percentiles,
    _compute_skewness,
    analyze_length_distribution,
)
from afterburn.types import PromptResult

import numpy as np


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


class TestSkewness:
    def test_symmetric_distribution(self):
        data = np.array([50, 60, 70, 80, 90], dtype=np.float64)
        skew = _compute_skewness(data)
        assert abs(skew) < 0.5  # Roughly symmetric

    def test_right_skewed(self):
        data = np.array([10, 11, 12, 13, 14, 15, 100], dtype=np.float64)
        skew = _compute_skewness(data)
        assert skew > 0  # Right-skewed

    def test_too_few_samples(self):
        data = np.array([10, 20], dtype=np.float64)
        skew = _compute_skewness(data)
        assert skew == 0.0


class TestKurtosis:
    def test_normal_like(self):
        np.random.seed(42)
        data = np.random.normal(100, 10, 100)
        kurtosis = _compute_kurtosis(data)
        # Excess kurtosis of normal distribution is ~0
        assert abs(kurtosis) < 1.5

    def test_too_few_samples(self):
        data = np.array([10, 20, 30], dtype=np.float64)
        kurtosis = _compute_kurtosis(data)
        assert kurtosis == 0.0


class TestPercentiles:
    def test_basic_percentiles(self):
        data = np.arange(1, 101, dtype=np.float64)  # 1 to 100
        p = _compute_percentiles(data)
        assert "p10" in p
        assert "p25" in p
        assert "p50" in p
        assert "p75" in p
        assert "p90" in p
        assert p["p50"] == pytest.approx(50.5, abs=0.5)

    def test_empty_data(self):
        p = _compute_percentiles(np.array([], dtype=np.float64))
        assert p == {}


class TestLengthAnalysisPhase2:
    def test_skewness_in_analysis(self):
        base = _make_results([50, 55, 60, 65, 70])
        trained = _make_results([50, 55, 60, 65, 200])  # Right-skewed
        analysis = analyze_length_distribution(base, trained)
        # Trained should be more right-skewed
        assert analysis.trained_skewness > analysis.base_skewness

    def test_percentiles_in_analysis(self):
        base = _make_results([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        trained = _make_results([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
        analysis = analyze_length_distribution(base, trained)
        assert "p50" in analysis.base_percentiles
        assert "p50" in analysis.trained_percentiles
        assert analysis.trained_percentiles["p50"] > analysis.base_percentiles["p50"]

    def test_kurtosis_in_analysis(self):
        base = _make_results([50, 55, 60, 65, 70, 50, 55, 60, 65, 70])
        trained = _make_results([10, 50, 55, 60, 65, 70, 50, 55, 60, 200])
        analysis = analyze_length_distribution(base, trained)
        # Both should have kurtosis computed
        assert isinstance(analysis.base_kurtosis, float)
        assert isinstance(analysis.trained_kurtosis, float)

    def test_per_category_cohens_d(self):
        base = (
            _make_results([50, 55, 60], "math")
            + _make_results([70, 75, 80], "code")
        )
        trained = (
            _make_results([150, 155, 160], "math")
            + _make_results([72, 77, 82], "code")
        )
        analysis = analyze_length_distribution(base, trained)
        assert "math" in analysis.per_category
        assert "cohens_d" in analysis.per_category["math"]
        # Math should have large effect size
        assert analysis.per_category["math"]["cohens_d"] > 1.0

    def test_empty_inputs_have_defaults(self):
        analysis = analyze_length_distribution([], [])
        assert analysis.base_skewness == 0.0
        assert analysis.trained_skewness == 0.0
        assert analysis.base_kurtosis == 0.0
        assert analysis.trained_kurtosis == 0.0
        assert analysis.base_percentiles == {}
        assert analysis.trained_percentiles == {}
