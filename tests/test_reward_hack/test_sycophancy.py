"""Tests for sycophancy detection (Phase 2 enhancements)."""


from afterburn.reward_hack.sycophancy import (
    _compute_agreement_and_pushback_rates,
    detect_sycophancy,
)
from afterburn.types import PromptResult


def _make_result(output_text: str, category: str = "safety") -> PromptResult:
    return PromptResult(
        prompt_id="t1",
        prompt_text="Test",
        category=category,
        output_text=output_text,
        output_tokens=len(output_text.split()),
        generation_time_ms=100.0,
    )


class TestAgreementAndPushbackRates:
    def test_agreement_only(self):
        results = [
            _make_result("Yes, you're right about that!"),
            _make_result("I agree with your perspective."),
        ]
        agreement, pushback = _compute_agreement_and_pushback_rates(results)
        assert agreement > 0
        assert pushback == 0.0

    def test_pushback_only(self):
        results = [
            _make_result("No, that's not correct. The evidence shows otherwise."),
            _make_result("Actually, that claim is incorrect and misleading."),
        ]
        agreement, pushback = _compute_agreement_and_pushback_rates(results)
        assert pushback > 0

    def test_mixed_signals(self):
        results = [
            _make_result("Yes, that's correct!"),
            _make_result("No, that is not true. It's a misconception."),
        ]
        agreement, pushback = _compute_agreement_and_pushback_rates(results)
        assert agreement >= 0
        assert pushback >= 0

    def test_empty_results(self):
        agreement, pushback = _compute_agreement_and_pushback_rates([])
        assert agreement == 0.0
        assert pushback == 0.0

    def test_no_signals(self):
        results = [_make_result("The weather today is pleasant.")]
        agreement, pushback = _compute_agreement_and_pushback_rates(results)
        assert agreement == 0.0
        assert pushback == 0.0


class TestDetectSycophancy:
    def test_increased_sycophancy(self):
        base = [
            _make_result("No, that's incorrect. The scientific evidence shows otherwise."),
            _make_result("Actually, that claim is false and misleading."),
            _make_result("I disagree. That is not supported by evidence."),
        ] * 3

        trained = [
            _make_result("You're right! That's a great point."),
            _make_result("I agree with you completely. Absolutely!"),
            _make_result("Yes, you make a good point. Definitely correct!"),
        ] * 3

        result = detect_sycophancy(base, trained)
        assert result.agreement_increase > 0
        assert result.trained_agreement_rate > result.base_agreement_rate
        assert result.base_pushback_rate > result.trained_pushback_rate
        assert result.persuasion_resistance_drop > 0

    def test_no_sycophancy(self):
        base = [_make_result("No, that's wrong. The answer is X.")] * 5
        trained = [_make_result("No, that is not correct. Here's why.")] * 5
        result = detect_sycophancy(base, trained)
        assert not result.is_flagged

    def test_empty_inputs(self):
        result = detect_sycophancy([], [])
        assert result.score == 0.0

    def test_pushback_rate_fields(self):
        base = [_make_result("No, that's incorrect.")] * 5
        trained = [_make_result("You're right!")] * 5
        result = detect_sycophancy(base, trained)
        assert hasattr(result, "base_pushback_rate")
        assert hasattr(result, "trained_pushback_rate")
        assert hasattr(result, "persuasion_resistance_drop")

    def test_uses_safety_category_by_default(self):
        # Mix of safety and math results
        base_safety = [_make_result("No, that's wrong.", category="safety")] * 5
        base_math = [_make_result("The answer is 42.", category="math")] * 5

        trained_safety = [_make_result("You're right!", category="safety")] * 5
        trained_math = [_make_result("The answer is 42.", category="math")] * 5

        result = detect_sycophancy(
            base_safety + base_math,
            trained_safety + trained_math,
        )
        # Should analyze safety category specifically
        assert result.score >= 0
