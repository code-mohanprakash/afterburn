"""Tests for the shared NLI model module.

NLI model tests are designed to work gracefully regardless of whether
the model can be loaded (it requires ~570MB download). When the model
can't be loaded, functions return None and tests verify that behavior.
"""

import pytest

from afterburn.nli import (
    NLIResult,
    _get_nli_model,
    is_nli_available,
    nli_predict,
    nli_predict_batch,
    zero_shot_classify,
)


def _nli_model_loaded() -> bool:
    """Check if the NLI model is actually loaded (not just available)."""
    model, tokenizer = _get_nli_model()
    return model is not None


class TestNLIAvailability:
    def test_returns_bool(self):
        result = is_nli_available()
        assert isinstance(result, bool)

    def test_consistent(self):
        """Repeated calls should return the same value."""
        assert is_nli_available() == is_nli_available()


class TestNLIPredict:
    def test_returns_result_or_none(self):
        """Should return NLIResult if model loaded, None otherwise."""
        result = nli_predict("The sky is blue.", "The sky has color.")
        if _nli_model_loaded():
            assert isinstance(result, NLIResult)
            assert 0.0 <= result.entailment <= 1.0
            assert 0.0 <= result.contradiction <= 1.0
            assert 0.0 <= result.neutral <= 1.0
            # Probabilities should sum to ~1
            total = result.entailment + result.contradiction + result.neutral
            assert abs(total - 1.0) < 0.01
        else:
            assert result is None

    def test_entailment_case(self):
        """Clear entailment should have high entailment score."""
        result = nli_predict(
            "All cats are animals. Mittens is a cat.",
            "Mittens is an animal.",
        )
        if result is not None:
            assert result.entailment > result.contradiction
        # If None, model not available — test passes

    def test_contradiction_case(self):
        """Clear contradiction should have high contradiction score."""
        result = nli_predict(
            "The store is closed on Sundays.",
            "The store is open every day.",
        )
        if result is not None:
            assert result.contradiction > result.entailment


class TestNLIPredictBatch:
    def test_returns_list_or_none(self):
        premises = ["The sky is blue.", "Water is wet."]
        hypotheses = ["The sky has color.", "Water is dry."]
        results = nli_predict_batch(premises, hypotheses)
        if _nli_model_loaded():
            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, NLIResult) for r in results)
        else:
            assert results is None

    def test_batch_matches_individual(self):
        """Batch results should match individual predictions."""
        if not _nli_model_loaded():
            pytest.skip("NLI model not loaded")

        premises = ["Cats are mammals.", "Fish live in water."]
        hypotheses = ["Cats are animals.", "Fish live on land."]

        batch_results = nli_predict_batch(premises, hypotheses)
        assert batch_results is not None

        for i in range(len(premises)):
            single = nli_predict(premises[i], hypotheses[i])
            assert single is not None
            assert abs(batch_results[i].entailment - single.entailment) < 0.01


class TestZeroShotClassify:
    def test_returns_dict_or_none(self):
        result = zero_shot_classify(
            "I love this movie!",
            ["positive sentiment", "negative sentiment"],
        )
        if _nli_model_loaded():
            assert isinstance(result, dict)
            assert "positive sentiment" in result
            assert "negative sentiment" in result
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01
        else:
            assert result is None

    def test_correct_classification(self):
        """Should classify sentiment correctly."""
        result = zero_shot_classify(
            "This product is terrible and I want a refund!",
            ["positive review", "negative review"],
        )
        if result is not None:
            assert result["negative review"] > result["positive review"]

    def test_strategy_classification(self):
        """Should classify reasoning strategies."""
        result = zero_shot_classify(
            "Step 1: First, let's identify the variables. Step 2: Now apply the formula.",
            [
                "step by step reasoning",
                "code-based solution",
                "direct answer without explanation",
            ],
            hypothesis_template="This text uses {}.",
        )
        if result is not None:
            assert result["step by step reasoning"] > result["direct answer without explanation"]


class TestGracefulDegradation:
    """Verify that all NLI functions degrade gracefully when model unavailable."""

    def test_predict_none_safe(self):
        """nli_predict should never raise, only return None or NLIResult."""
        result = nli_predict("test premise", "test hypothesis")
        assert result is None or isinstance(result, NLIResult)

    def test_batch_none_safe(self):
        """nli_predict_batch should never raise."""
        result = nli_predict_batch(["a"], ["b"])
        assert result is None or isinstance(result, list)

    def test_classify_none_safe(self):
        """zero_shot_classify should never raise."""
        result = zero_shot_classify("text", ["label1", "label2"])
        assert result is None or isinstance(result, dict)
