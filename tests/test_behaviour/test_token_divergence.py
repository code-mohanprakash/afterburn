"""Tests for token probability Jensen-Shannon divergence analysis."""

import math

import pytest

from afterburn.behaviour.token_divergence import (
    _compute_prompt_jsd,
    _jsd_dicts,
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


class TestJSDDicts:
    """Test the core JSD computation between two probability dicts."""

    def test_identical_distributions(self):
        """JSD of identical distributions should be 0."""
        p = {"a": 0.7, "b": 0.2, "c": 0.1}
        assert _jsd_dicts(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_completely_disjoint(self):
        """Completely disjoint distributions should give JSD = 1 (max)."""
        p = {"a": 1.0}
        q = {"b": 1.0}
        # JSD = (1/2)*1*log2(1/0.5) + (1/2)*1*log2(1/0.5) = 0.5 + 0.5 = 1.0
        assert _jsd_dicts(p, q) == pytest.approx(1.0, abs=1e-10)

    def test_symmetric(self):
        """JSD should be symmetric: JSD(P||Q) = JSD(Q||P)."""
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.1, "b": 0.9}
        assert _jsd_dicts(p, q) == pytest.approx(_jsd_dicts(q, p), abs=1e-10)

    def test_bounded_zero_one(self):
        """JSD should always be in [0, 1] with log base 2."""
        p = {"a": 0.5, "b": 0.3, "c": 0.2}
        q = {"a": 0.1, "d": 0.7, "e": 0.2}
        jsd = _jsd_dicts(p, q)
        assert 0.0 <= jsd <= 1.0

    def test_partial_overlap(self):
        """Partially overlapping distributions should give intermediate JSD."""
        p = {"a": 0.6, "b": 0.4}
        q = {"b": 0.6, "c": 0.4}
        jsd = _jsd_dicts(p, q)
        assert 0.0 < jsd < 1.0

    def test_known_value(self):
        """Verify against manually computed JSD."""
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 1.0}
        # M = {"a": 0.75, "b": 0.25}
        # KL(P||M) = 0.5*log2(0.5/0.75) + 0.5*log2(0.5/0.25) = 0.5*(-0.585) + 0.5*(1.0) = 0.2075
        # KL(Q||M) = 1.0*log2(1.0/0.75) = 0.4150
        # JSD = 0.5*0.2075 + 0.5*0.4150 = 0.31125
        expected = 0.5 * (0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25)) + \
                   0.5 * (1.0 * math.log2(1.0 / 0.75))
        jsd = _jsd_dicts(p, q)
        assert jsd == pytest.approx(expected, abs=1e-6)


class TestComputePromptJSD:
    def test_identical_distributions(self):
        """JSD of identical distributions should be ~0."""
        dist = [{"a": 0.7, "b": 0.2, "c": 0.1}]
        jsd = _compute_prompt_jsd(dist, dist)
        assert jsd is not None
        assert jsd == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions(self):
        """Different distributions should have positive JSD."""
        base = [{"a": 0.9, "b": 0.1}]
        trained = [{"a": 0.1, "b": 0.9}]
        jsd = _compute_prompt_jsd(base, trained)
        assert jsd is not None
        assert jsd > 0

    def test_multi_step(self):
        """Should average JSD across multiple token positions."""
        base = [
            {"a": 0.8, "b": 0.2},
            {"x": 0.6, "y": 0.4},
        ]
        trained = [
            {"a": 0.8, "b": 0.2},
            {"x": 0.6, "y": 0.4},
        ]
        jsd = _compute_prompt_jsd(base, trained)
        assert jsd is not None
        assert jsd == pytest.approx(0.0, abs=1e-6)

    def test_empty_base(self):
        assert _compute_prompt_jsd([], [{"a": 0.5}]) is None

    def test_empty_trained(self):
        assert _compute_prompt_jsd([{"a": 0.5}], []) is None

    def test_bounded(self):
        """JSD should always be in [0, 1]."""
        base = [{"hello": 0.5, "world": 0.3, "foo": 0.2}]
        trained = [{"hello": 0.1, "bar": 0.7, "baz": 0.2}]
        jsd = _compute_prompt_jsd(base, trained)
        assert jsd is not None
        assert 0.0 <= jsd <= 1.0

    def test_mismatched_tokens(self):
        """Tokens only in one distribution should be handled naturally."""
        base = [{"cat": 0.9, "dog": 0.1}]
        trained = [{"fish": 0.8, "bird": 0.2}]
        jsd = _compute_prompt_jsd(base, trained)
        assert jsd is not None
        assert 0.0 < jsd <= 1.0


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
        assert result.mean_jsd > 0
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
        # Top divergent should have highest JSD
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
        assert result.mean_jsd == 0.0

    def test_jsd_bounded_all_prompts(self):
        """All per-prompt JSD values should be in [0, 1]."""
        base = []
        trained = []
        for i in range(10):
            base.append(_make_result(
                f"p{i}", "text",
                top_token_probs=[{"a": 0.5 + i * 0.04, "b": 0.5 - i * 0.04}],
            ))
            trained.append(_make_result(
                f"p{i}", "text",
                top_token_probs=[{"c": 0.3, "d": 0.7}],
            ))
        result = analyze_token_divergence(base, trained)
        for jsd_val in result.per_prompt_jsd:
            assert 0.0 <= jsd_val <= 1.0
