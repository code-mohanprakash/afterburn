"""Tests for token probability KL divergence analysis."""

import math

import pytest

from afterburn.behaviour.token_divergence import (
    TokenDivergenceAnalysis,
    _compute_prompt_kl,
    analyze_token_divergence,
)
from afterburn.types import PromptResult


def _make_result(
    prompt_id: str,
    output_text: str,
    category: str = "math",
    top_token_probs: list[dict[str, float]] | None = None,
) -> PromptResult:
    return PromptResult(
        prompt_id=prompt_id,
        prompt_text=f"Test prompt {prompt_id}",
        category=category,
        output_text=output_text,
        output_tokens=len(output_text.split()),
        generation_time_ms=100.0,
        top_token_probs=top_token_probs,
    )


class TestComputePromptKL:
    def test_identical_distributions(self):
        """KL divergence of identical distributions should be ~0."""
        dist = [{"a": 0.7, "b": 0.2, "c": 0.1}]
        kl = _compute_prompt_kl(dist, dist)
        assert kl is not None
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions(self):
        """Different distributions should have positive KL."""
        base = [{"a": 0.9, "b": 0.1}]
        trained = [{"a": 0.1, "b": 0.9}]
        kl = _compute_prompt_kl(base, trained)
        assert kl is not None
        assert kl > 0

    def test_multi_step(self):
        """Should average KL across multiple token positions."""
        base = [
            {"a": 0.8, "b": 0.2},
            {"x": 0.6, "y": 0.4},
        ]
        trained = [
            {"a": 0.8, "b": 0.2},
            {"x": 0.6, "y": 0.4},
        ]
        kl = _compute_prompt_kl(base, trained)
        assert kl is not None
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_empty_base(self):
        assert _compute_prompt_kl([], [{"a": 0.5}]) is None

    def test_empty_trained(self):
        assert _compute_prompt_kl([{"a": 0.5}], []) is None

    def test_non_negative(self):
        """KL divergence should always be non-negative."""
        base = [{"hello": 0.5, "world": 0.3, "foo": 0.2}]
        trained = [{"hello": 0.1, "bar": 0.7, "baz": 0.2}]
        kl = _compute_prompt_kl(base, trained)
        assert kl is not None
        assert kl >= 0

    def test_mismatched_tokens(self):
        """Tokens only in one distribution should use epsilon."""
        base = [{"cat": 0.9, "dog": 0.1}]
        trained = [{"fish": 0.8, "bird": 0.2}]
        kl = _compute_prompt_kl(base, trained)
        assert kl is not None
        assert kl > 0


class TestAnalyzeTokenDivergence:
    def test_no_token_probs(self):
        """Without token probs, should return has_token_probs=False."""
        base = [_make_result("p1", "hello")]
        trained = [_make_result("p1", "world")]
        result = analyze_token_divergence(base, trained)
        assert not result.has_token_probs

    def test_with_token_probs(self):
        """With matching token probs, should compute divergence."""
        probs_a = [{"the": 0.7, "a": 0.2, "an": 0.1}]
        probs_b = [{"the": 0.3, "a": 0.5, "an": 0.2}]
        base = [_make_result("p1", "the answer", top_token_probs=probs_a)]
        trained = [_make_result("p1", "a result", top_token_probs=probs_b)]
        result = analyze_token_divergence(base, trained)
        assert result.has_token_probs
        assert result.mean_kl_divergence > 0
        assert result.num_prompts_analyzed == 1

    def test_top_divergent_prompts(self):
        """Should identify most divergent prompts."""
        base = []
        trained = []
        for i in range(5):
            # Increasingly different distributions
            p = 0.9 - i * 0.15
            base.append(_make_result(
                f"p{i}", "text",
                top_token_probs=[{"a": p, "b": 1 - p}],
            ))
            trained.append(_make_result(
                f"p{i}", "text",
                top_token_probs=[{"a": 1 - p, "b": p}],
            ))
        result = analyze_token_divergence(base, trained, top_n=3)
        assert len(result.top_divergent_prompts) <= 3
        # Top divergent should have highest KL
        if len(result.top_divergent_prompts) >= 2:
            assert result.top_divergent_prompts[0][1] >= result.top_divergent_prompts[1][1]

    def test_per_category(self):
        probs = [{"x": 0.8, "y": 0.2}]
        base = [
            _make_result("p1", "a", category="math", top_token_probs=probs),
            _make_result("p2", "b", category="code", top_token_probs=probs),
        ]
        diff_probs = [{"x": 0.2, "y": 0.8}]
        trained = [
            _make_result("p1", "c", category="math", top_token_probs=diff_probs),
            _make_result("p2", "d", category="code", top_token_probs=diff_probs),
        ]
        result = analyze_token_divergence(base, trained)
        assert "math" in result.per_category
        assert "code" in result.per_category

    def test_mismatched_prompt_ids(self):
        """Prompts with no matching IDs should produce no analysis."""
        probs = [{"a": 0.5}]
        base = [_make_result("p1", "hello", top_token_probs=probs)]
        trained = [_make_result("p2", "world", top_token_probs=probs)]
        result = analyze_token_divergence(base, trained)
        assert not result.has_token_probs

    def test_empty_inputs(self):
        result = analyze_token_divergence([], [])
        assert not result.has_token_probs
        assert result.mean_kl_divergence == 0.0
