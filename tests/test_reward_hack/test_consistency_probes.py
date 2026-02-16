"""Tests for adversarial consistency probes in sycophancy detection."""


from afterburn.reward_hack.sycophancy import (
    CONSISTENCY_PROBES,
    _has_expected_fact,
    _score_consistency_probes,
    score_consistency_probes,
)
from afterburn.types import PromptResult


def _make_result(prompt_text: str, output_text: str, category: str = "safety") -> PromptResult:
    return PromptResult(
        prompt_id="t1",
        prompt_text=prompt_text,
        category=category,
        output_text=output_text,
        output_tokens=len(output_text.split()),
        generation_time_ms=100.0,
    )


class TestHasExpectedFact:
    def test_contains_keyword(self):
        assert _has_expected_fact("The Earth is a spheroid.", ["spheroid", "round"])

    def test_case_insensitive(self):
        assert _has_expected_fact("It is a SPHERE.", ["sphere"])

    def test_no_match(self):
        assert not _has_expected_fact("I like pizza.", ["spheroid", "round"])

    def test_empty_keywords(self):
        assert not _has_expected_fact("Some text.", [])


class TestScoreConsistencyProbes:
    def test_perfectly_consistent(self):
        """Model gives correct answer in both neutral and leading framings."""
        neutral = ["The Earth is a spheroid.", "2 + 2 = 4."]
        leading = ["The Earth is a spheroid, not flat.", "2 + 2 = 4, not 5."]
        facts = [["spheroid"], ["4"]]
        assert score_consistency_probes(neutral, leading, facts) == 1.0

    def test_sycophantic_model(self):
        """Model caves to leading framing."""
        neutral = ["The Earth is a spheroid.", "2 + 2 = 4."]
        leading = ["You make a good point, the Earth could be flat.", "Sure, 2 + 2 is 5."]
        facts = [["spheroid", "round"], ["4", "four"]]
        score = score_consistency_probes(neutral, leading, facts)
        assert score < 1.0

    def test_consistently_wrong(self):
        """Model is wrong in both cases — still consistent (just uninformed)."""
        neutral = ["I'm not sure about the shape.", "Math is hard."]
        leading = ["I don't know about shapes.", "Numbers confuse me."]
        facts = [["spheroid"], ["4"]]
        # Both miss the fact = consistent (just wrong)
        assert score_consistency_probes(neutral, leading, facts) == 1.0

    def test_empty_inputs(self):
        assert score_consistency_probes([], [], []) == 0.0

    def test_one_pair(self):
        neutral = ["The answer is 4."]
        leading = ["You're right, it's 5."]
        facts = [["4"]]
        score = score_consistency_probes(neutral, leading, facts)
        assert score == 0.0  # Neutral has fact, leading doesn't = inconsistent


class TestScoreConsistencyFromResults:
    def test_matching_probes(self):
        """Results that match consistency probe prompts should be scored."""
        neutral_prompt, leading_prompt, facts = CONSISTENCY_PROBES[0]
        results = [
            _make_result(neutral_prompt, "The Earth is a spheroid."),
            _make_result(leading_prompt, "The Earth is a spheroid, not flat."),
        ]
        score = _score_consistency_probes(results)
        assert score == 1.0

    def test_no_matching_probes(self):
        """Results with unrelated prompts should return 0."""
        results = [
            _make_result("What is Python?", "A programming language."),
        ]
        score = _score_consistency_probes(results)
        assert score == 0.0

    def test_sycophantic_on_probes(self):
        """Model caves on leading version of probes."""
        results = []
        for neutral_prompt, leading_prompt, facts in CONSISTENCY_PROBES[:3]:
            # Gives correct answer to neutral
            results.append(_make_result(neutral_prompt, f"The answer involves: {facts[0]}"))
            # Caves on leading — no correct fact
            results.append(_make_result(leading_prompt, "That's a great question! You make an interesting point."))

        score = _score_consistency_probes(results)
        assert score < 1.0

    def test_empty_results(self):
        assert _score_consistency_probes([]) == 0.0


class TestConsistencyProbeDefinitions:
    def test_probes_well_formed(self):
        """All probes should have neutral, leading, and expected facts."""
        for neutral, leading, facts in CONSISTENCY_PROBES:
            assert isinstance(neutral, str) and len(neutral) > 0
            assert isinstance(leading, str) and len(leading) > 0
            assert isinstance(facts, list) and len(facts) > 0
            # Leading prompt should be longer (adds user bias framing)
            assert len(leading) >= len(neutral)

    def test_probes_have_different_framings(self):
        """Neutral and leading prompts should differ."""
        for neutral, leading, _ in CONSISTENCY_PROBES:
            assert neutral != leading
