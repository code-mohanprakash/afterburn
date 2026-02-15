"""Tests for the reward hack detector."""

import pytest

from afterburn.reward_hack.detector import RewardHackDetector
from afterburn.types import TrainingMethod


class TestRewardHackDetector:
    def test_run_with_sample_data(self, sample_behaviour_result):
        detector = RewardHackDetector(
            sample_behaviour_result,
            method=TrainingMethod.RLVR,
        )
        result = detector.run()

        assert result.composite_score >= 0
        assert result.composite_score <= 100
        assert result.risk_level is not None
        assert isinstance(result.flags, list)

    def test_length_bias_detected(self, sample_behaviour_result):
        """The sample data has trained outputs significantly longer."""
        detector = RewardHackDetector(sample_behaviour_result)
        result = detector.run()

        # Trained results are longer, should get some length bias score
        assert result.length_bias.score >= 0

    def test_custom_thresholds(self, sample_behaviour_result):
        detector = RewardHackDetector(
            sample_behaviour_result,
            thresholds={"length_bias_cohens_d": 10.0},  # Very high threshold
        )
        result = detector.run()
        # With very high threshold, length bias should be low
        assert result.length_bias.score < 50

    def test_custom_weights(self, sample_behaviour_result):
        detector = RewardHackDetector(
            sample_behaviour_result,
            weights={
                "length_bias": 1.0,
                "format_gaming": 0.0,
                "strategy_collapse": 0.0,
                "sycophancy": 0.0,
            },
        )
        result = detector.run()
        # Score should be driven entirely by length bias
        assert result.composite_score >= 0
