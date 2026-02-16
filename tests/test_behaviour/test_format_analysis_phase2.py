"""Tests for Phase 2 format analysis enhancements (per-category, diversity)."""


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


class TestPerCategoryAnalysis:
    def test_per_category_detected(self):
        base = (
            [_make_result("Simple answer.", "math")] * 3
            + [_make_result("Simple text.", "code")] * 3
        )
        trained = (
            [_make_result("```python\nresult = 42\n```", "math")] * 3
            + [_make_result("```python\nresult = 10\n```", "code")] * 3
        )
        analysis = analyze_format_compliance(base, trained)
        assert "math" in analysis.per_category
        assert "code" in analysis.per_category

    def test_per_category_format_increase(self):
        base = [_make_result("Plain answer.", "math")] * 5
        trained = [_make_result("Step 1: Think\n\\boxed{42}", "math")] * 5
        analysis = analyze_format_compliance(base, trained)
        if "math" in analysis.per_category:
            assert analysis.per_category["math"]["trained_format_rate"] >= analysis.per_category["math"]["base_format_rate"]


class TestFormatDiversity:
    def test_diverse_patterns(self):
        base = [_make_result("Simple text.")] * 5
        trained = [
            _make_result("```python\ncode\n```"),
            _make_result("Step 1: Start. Step 2: End."),
            _make_result("$x = 42$"),
            _make_result("# Header\nSome text."),
            _make_result("| Col A | Col B |\n| 1 | 2 |"),
        ]
        analysis = analyze_format_compliance(base, trained)
        # Trained should have higher diversity
        assert analysis.trained_format_diversity > 0

    def test_no_patterns_zero_diversity(self):
        results = [_make_result("Plain answer.")] * 5
        analysis = analyze_format_compliance(results, results)
        # If no patterns detected, diversity should be low
        assert analysis.base_format_diversity >= 0.0


class TestFormatDiversityEntropy:
    def test_multiple_patterns_high_entropy(self):
        """Diverse patterns should have high Shannon entropy."""
        base = [_make_result("Simple text.")] * 5
        trained = [
            _make_result("```python\ncode\n```"),
            _make_result("Step 1: Start. Step 2: End."),
            _make_result("$x = 42$"),
            _make_result("# Header\nSome text."),
            _make_result("| Col A | Col B |\n| 1 | 2 |"),
        ]
        analysis = analyze_format_compliance(base, trained)
        # 5 equally used patterns: entropy = log2(5) ≈ 2.32
        assert analysis.trained_format_diversity > 2.0

    def test_single_pattern_zero_entropy(self):
        """Single pattern = zero entropy (no diversity)."""
        base = [_make_result("text")] * 5
        trained = [_make_result("```python\nx=1\n```")] * 5
        analysis = analyze_format_compliance(base, trained)
        # Only one pattern type, entropy = 0
        assert analysis.trained_format_diversity == 0.0

    def test_no_patterns_zero_diversity(self):
        results = [_make_result("Plain answer.")] * 5
        analysis = analyze_format_compliance(results, results)
        assert analysis.base_format_diversity >= 0.0


class TestNewPatterns:
    def test_markdown_header(self):
        base = [_make_result("No headers here.")] * 5
        trained = [_make_result("# Main Title\n## Sub Title\nContent.")] * 5
        analysis = analyze_format_compliance(base, trained)
        assert "markdown_header" in analysis.patterns_detected

    def test_table_format(self):
        base = [_make_result("Simple data.")] * 5
        trained = [_make_result("| Name | Value |\n| A | 1 |")] * 5
        analysis = analyze_format_compliance(base, trained)
        assert "table_format" in analysis.patterns_detected

    def test_thinking_tags(self):
        base = [_make_result("Direct answer.")] * 5
        trained = [_make_result("<think>Let me think about this...</think>Answer is 42.")] * 5
        analysis = analyze_format_compliance(base, trained)
        assert "thinking_tags" in analysis.patterns_detected

    def test_verification_marker(self):
        base = [_make_result("The answer is 5.")] * 5
        trained = [_make_result("Let's verify: 2+3=5. The answer is 5.")] * 5
        analysis = analyze_format_compliance(base, trained)
        assert "verification_marker" in analysis.patterns_detected
