"""Tests for format analysis."""

import pytest

from afterburn.behaviour.format_analysis import analyze_format_compliance
from afterburn.types import PromptResult


def _make_result(output_text: str, category: str = "math") -> PromptResult:
    return PromptResult(
        prompt_id="t1",
        prompt_text="Test",
        category=category,
        output_text=output_text,
        output_tokens=len(output_text.split()),
        generation_time_ms=100.0,
    )


class TestFormatAnalysis:
    def test_no_format_patterns(self):
        results = [_make_result("Simple answer without formatting.") for _ in range(5)]
        analysis = analyze_format_compliance(results, results)
        assert analysis.format_increase == pytest.approx(0.0, abs=0.01)

    def test_increased_code_blocks(self):
        base = [_make_result("I think the result is 42, but I am not sure.") for _ in range(5)]
        trained = [_make_result("```python\nresult = 42\nprint(result)\n```") for _ in range(5)]
        analysis = analyze_format_compliance(base, trained)
        assert "code_block" in analysis.patterns_detected
        assert analysis.patterns_detected["code_block"]["trained_rate"] > analysis.patterns_detected["code_block"]["base_rate"]

    def test_empty_results(self):
        analysis = analyze_format_compliance([], [])
        assert analysis.format_increase == 0.0
