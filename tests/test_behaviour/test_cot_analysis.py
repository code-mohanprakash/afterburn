"""Tests for chain-of-thought analysis (Phase 2 enhancements)."""

import pytest

from afterburn.behaviour.cot_analysis import (
    _count_reasoning_steps,
    _detect_self_correction,
    _detect_verification,
    _estimate_reasoning_depth,
    analyze_chain_of_thought,
)
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


class TestCountReasoningSteps:
    def test_explicit_steps(self):
        text = "Step 1: Do this\nStep 2: Do that\nStep 3: Done"
        steps = _count_reasoning_steps(text)
        assert steps >= 2

    def test_numbered_list(self):
        text = "1. First item\n2. Second item\n3. Third item"
        steps = _count_reasoning_steps(text)
        assert steps >= 2

    def test_ordinal_words(self):
        text = "First, we need to identify. Then, we compute. Finally, we check."
        steps = _count_reasoning_steps(text)
        assert steps >= 2

    def test_empty_text(self):
        assert _count_reasoning_steps("") == 0
        assert _count_reasoning_steps("   ") == 0

    def test_short_text_fallback(self):
        text = "Just a short answer."
        steps = _count_reasoning_steps(text)
        assert steps >= 0


class TestEstimateReasoningDepth:
    def test_deep_reasoning(self):
        text = (
            "Because of this, we need to consider the implications. "
            "Therefore, it follows that the result is correct. "
            "Since the base case holds, consequently the induction step works. "
            "Thus we can conclude the proof."
        )
        depth = _estimate_reasoning_depth(text)
        assert depth > 3.0

    def test_shallow_text(self):
        depth = _estimate_reasoning_depth("The answer is 42.")
        assert depth < 3.0

    def test_empty(self):
        assert _estimate_reasoning_depth("") == 0.0


class TestDetectSelfCorrection:
    def test_wait_pattern(self):
        assert _detect_self_correction("Wait, that's not right. Let me redo this.")

    def test_actually_pattern(self):
        assert _detect_self_correction("Actually, I made a mistake earlier.")

    def test_explicit_correction(self):
        assert _detect_self_correction("I was wrong about the sign. Correction: it's positive.")

    def test_oops_pattern(self):
        assert _detect_self_correction("Oops, I forgot to carry the one.")

    def test_let_me_reconsider(self):
        assert _detect_self_correction("Let me reconsider this approach.")

    def test_no_correction(self):
        assert not _detect_self_correction("The answer is straightforward: 42.")

    def test_empty(self):
        assert not _detect_self_correction("")


class TestDetectVerification:
    def test_lets_verify(self):
        assert _detect_verification("Let's verify our result by substituting back.")

    def test_let_me_check(self):
        assert _detect_verification("Let me check this calculation again.")

    def test_double_check(self):
        assert _detect_verification("We should double-check this answer.")

    def test_sanity_check(self):
        assert _detect_verification("Sanity check: does this make sense?")

    def test_to_confirm(self):
        assert _detect_verification("To confirm, we plug the value back in.")

    def test_no_verification(self):
        assert not _detect_verification("The answer is simply 42.")

    def test_empty(self):
        assert not _detect_verification("")


class TestAnalyzeChainOfThought:
    def test_basic_analysis(self):
        base = [
            _make_result("Step 1: Start. Step 2: Compute. The answer is 10."),
            _make_result("First, identify. Then solve. Therefore, 5."),
        ]
        trained = [
            _make_result(
                "Step 1: Start. Step 2: Compute. Step 3: Verify. "
                "Let's verify our answer. Wait, I need to reconsider."
            ),
            _make_result(
                "First, identify. Then solve. Let me check the result. "
                "Actually, I made a mistake. The corrected answer is 5."
            ),
        ]
        analysis = analyze_chain_of_thought(base, trained)

        assert analysis.trained_avg_steps >= analysis.base_avg_steps
        assert analysis.trained_verification_rate > 0
        assert analysis.trained_self_correction_rate > 0

    def test_depth_change(self):
        base = [_make_result("42.")]
        trained = [
            _make_result(
                "Because x=2, therefore y=4. Since this implies z=8, "
                "consequently the answer follows. Thus it is 42."
            )
        ]
        analysis = analyze_chain_of_thought(base, trained)
        assert analysis.depth_change > 0

    def test_empty_inputs(self):
        analysis = analyze_chain_of_thought([], [])
        assert analysis.base_avg_steps == 0.0
        assert analysis.trained_avg_steps == 0.0
        assert analysis.base_self_correction_rate == 0.0
        assert analysis.trained_verification_rate == 0.0
