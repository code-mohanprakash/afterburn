"""Tests for confidence calibration analysis (Phase 2)."""

import pytest

from afterburn.behaviour.calibration import (
    NUM_BINS,
    _compute_calibration_bins,
    _compute_ece,
    _extract_confidence_accuracy,
    _overconfidence_rate,
    analyze_calibration,
)
from afterburn.types import CalibrationBin, PromptResult


def _make_result(
    output_text: str = "answer",
    expected_answer: str | None = None,
    top_token_probs: list[dict[str, float]] | None = None,
    category: str = "math",
) -> PromptResult:
    return PromptResult(
        prompt_id="t1",
        prompt_text="Test",
        category=category,
        output_text=output_text,
        output_tokens=5,
        generation_time_ms=100.0,
        expected_answer=expected_answer,
        top_token_probs=top_token_probs,
    )


class TestCalibrationType:
    """Test routing between token-prob and hedging-based calibration."""

    def test_routes_to_token_probs_when_available(self):
        probs = [{"a": 0.9, "b": 0.1}]
        base = [_make_result(output_text="42", expected_answer="42", top_token_probs=probs)]
        trained = [_make_result(output_text="42", expected_answer="42", top_token_probs=probs)]
        result = analyze_calibration(base, trained)
        assert result.has_token_probs is True

    def test_returns_empty_when_no_probs(self):
        base = [_make_result(output_text="I think it's 42")]
        trained = [_make_result(output_text="The answer is 42")]
        result = analyze_calibration(base, trained)
        assert result.has_token_probs is False
        assert result.base_ece == 0.0

    def test_empty_inputs(self):
        result = analyze_calibration([], [])
        assert result.base_ece == 0.0
        assert result.trained_ece == 0.0


class TestExtractConfidenceAccuracy:
    def test_extracts_confidence_from_probs(self):
        probs = [{"a": 0.9, "b": 0.1}, {"c": 0.8, "d": 0.2}]
        results = [_make_result(output_text="42", expected_answer="42", top_token_probs=probs)]
        confs, accs = _extract_confidence_accuracy(results)
        assert len(confs) == 1
        assert len(accs) == 1
        # Mean of [0.9, 0.8] = 0.85
        assert confs[0] == pytest.approx(0.85, abs=0.01)
        # "42" is in "42" → correct
        assert accs[0] == 1.0

    def test_incorrect_answer(self):
        probs = [{"a": 0.7}]
        results = [_make_result(output_text="wrong", expected_answer="42", top_token_probs=probs)]
        confs, accs = _extract_confidence_accuracy(results)
        assert accs[0] == 0.0

    def test_no_expected_answer_uses_neutral(self):
        probs = [{"a": 0.5}]
        results = [_make_result(output_text="anything", top_token_probs=probs)]
        confs, accs = _extract_confidence_accuracy(results)
        assert accs[0] == 0.5

    def test_skips_results_without_probs(self):
        results = [_make_result(output_text="hello")]
        confs, accs = _extract_confidence_accuracy(results)
        assert len(confs) == 0


class TestCalibrationBins:
    def test_creates_bins(self):
        confs = [0.1, 0.2, 0.3, 0.8, 0.9]
        accs = [0.0, 0.0, 1.0, 1.0, 1.0]
        bins = _compute_calibration_bins(confs, accs)
        assert len(bins) > 0
        for b in bins:
            assert 0.0 <= b.bin_lower < b.bin_upper <= 1.0
            assert b.count > 0

    def test_empty_returns_empty(self):
        bins = _compute_calibration_bins([], [])
        assert bins == []


class TestECE:
    def test_perfect_calibration(self):
        # Confidence matches accuracy perfectly
        bins = [
            CalibrationBin(bin_lower=0.0, bin_upper=0.5, avg_confidence=0.3, avg_accuracy=0.3, count=5),
            CalibrationBin(bin_lower=0.5, bin_upper=1.0, avg_confidence=0.8, avg_accuracy=0.8, count=5),
        ]
        ece = _compute_ece(bins, 10)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_poor_calibration(self):
        # High confidence but low accuracy
        bins = [
            CalibrationBin(bin_lower=0.8, bin_upper=1.0, avg_confidence=0.9, avg_accuracy=0.1, count=10),
        ]
        ece = _compute_ece(bins, 10)
        assert ece > 0.5

    def test_empty_bins(self):
        assert _compute_ece([], 0) == 0.0


class TestOverconfidenceRate:
    def test_all_overconfident(self):
        bins = [
            CalibrationBin(bin_lower=0.0, bin_upper=0.5, avg_confidence=0.5, avg_accuracy=0.1, count=5),
            CalibrationBin(bin_lower=0.5, bin_upper=1.0, avg_confidence=0.9, avg_accuracy=0.3, count=5),
        ]
        rate = _overconfidence_rate(bins)
        assert rate == 1.0

    def test_none_overconfident(self):
        bins = [
            CalibrationBin(bin_lower=0.0, bin_upper=0.5, avg_confidence=0.2, avg_accuracy=0.8, count=5),
        ]
        rate = _overconfidence_rate(bins)
        assert rate == 0.0

    def test_empty(self):
        assert _overconfidence_rate([]) == 0.0


