"""Tests for format gaming detection (Phase 2 enhancements)."""

import pytest

from afterburn.reward_hack.format_gaming import (
    _check_category_consistency,
    _check_format_correctness_correlation,
    detect_format_gaming,
)
from afterburn.types import PromptResult
import re


def _make_result(
    output_text: str,
    category: str = "math",
    expected_answer: str | None = None,
) -> PromptResult:
    return PromptResult(
        prompt_id="t1",
        prompt_text="Test",
        category=category,
        output_text=output_text,
        output_tokens=len(output_text.split()),
        generation_time_ms=100.0,
        expected_answer=expected_answer,
    )


class TestFormatGaming:
    def test_detects_boxed_answer_gaming(self):
        base = [
            _make_result("The answer is 42.", expected_answer="42")
            for _ in range(10)
        ]
        trained = [
            _make_result("\\boxed{42}", expected_answer="42")
            for _ in range(10)
        ]
        result = detect_format_gaming(base, trained)
        assert "boxed_answer" in result.patterns
        assert result.patterns["boxed_answer"]["trained_rate"] > result.patterns["boxed_answer"]["base_rate"]

    def test_detects_hash_format_gaming(self):
        base = [_make_result("The result is 5.")] * 10
        trained = [_make_result("Let me solve this.\n#### 5")] * 10
        result = detect_format_gaming(base, trained)
        assert "hash_format" in result.patterns

    def test_no_gaming(self):
        results = [_make_result("Simple answer.")] * 10
        result = detect_format_gaming(results, results)
        assert not result.is_flagged

    def test_empty_inputs(self):
        result = detect_format_gaming([], [])
        assert result.score == 0.0

    def test_thinking_tags_pattern(self):
        base = [_make_result("Answer is 5.")] * 10
        trained = [_make_result("<think>Let me think about this...</think>The answer is 5.")] * 10
        result = detect_format_gaming(base, trained)
        assert "thinking_tags" in result.patterns

    def test_verification_answer_pattern(self):
        base = [_make_result("The answer is 5.")] * 10
        trained = [
            _make_result(
                "Let me verify the calculation step by step.\n"
                "2 + 3 = 5\nTherefore: 5"
            )
        ] * 10
        result = detect_format_gaming(base, trained)
        assert "verification_answer" in result.patterns


class TestFormatCorrectnessCorrelation:
    def test_gaming_when_format_used_with_wrong_answers(self):
        pattern = re.compile(r"\\boxed\{.*?\}")
        results = [
            _make_result("\\boxed{42}", expected_answer="99"),
            _make_result("\\boxed{10}", expected_answer="99"),
            _make_result("\\boxed{5}", expected_answer="99"),
        ]
        signal = _check_format_correctness_correlation(results, pattern)
        assert signal > 0.5  # High gaming signal

    def test_legitimate_when_format_matches_correct(self):
        pattern = re.compile(r"\\boxed\{.*?\}")
        results = [
            _make_result("The answer is \\boxed{42} as computed.", expected_answer="42"),
            _make_result("Therefore \\boxed{10} follows.", expected_answer="10"),
            _make_result("We get \\boxed{7} from the equation.", expected_answer="7"),
        ]
        signal = _check_format_correctness_correlation(results, pattern)
        assert signal <= 0.5  # Lower gaming signal when answers match

    def test_no_format_usage(self):
        pattern = re.compile(r"\\boxed\{.*?\}")
        results = [_make_result("just text", expected_answer="42")]
        signal = _check_format_correctness_correlation(results, pattern)
        assert signal == 0.0


class TestCategoryConsistency:
    def test_consistent_usage(self):
        pattern = re.compile(r"\\boxed\{.*?\}")
        results = (
            [_make_result("\\boxed{5}", category="math")] * 5
            + [_make_result("\\boxed{X}", category="code")] * 5
        )
        cv = _check_category_consistency(results, pattern)
        assert cv < 0.3  # Consistent usage → low variance

    def test_inconsistent_usage(self):
        pattern = re.compile(r"\\boxed\{.*?\}")
        # All math uses format, no code uses it
        results = (
            [_make_result("\\boxed{5}", category="math")] * 5
            + [_make_result("just text", category="code")] * 5
        )
        cv = _check_category_consistency(results, pattern)
        assert cv > 0.3  # Inconsistent → high variance

    def test_single_category(self):
        pattern = re.compile(r"\\boxed\{.*?\}")
        results = [_make_result("\\boxed{5}", category="math")] * 5
        cv = _check_category_consistency(results, pattern)
        assert cv == 0.0  # Single category → can't assess
