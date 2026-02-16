"""Tests for summary and recommendation generation."""


from afterburn.report.summary import generate_recommendations, generate_summary
from afterburn.types import (
    DiagnosticReport,
    ModelPair,
    TrainingMethod,
)


class TestGenerateSummary:
    def test_minimal_report(self):
        report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="base",
                trained_model="trained",
                method=TrainingMethod.SFT,
            )
        )
        summary = generate_summary(report)
        assert "base" in summary
        assert "trained" in summary
        assert "SFT" in summary

    def test_with_behaviour(self, sample_behaviour_result):
        report = DiagnosticReport(
            model_pair=ModelPair(base_model="a", trained_model="b"),
            behaviour=sample_behaviour_result,
        )
        summary = generate_summary(report)
        assert len(summary) > 0


class TestGenerateRecommendations:
    def test_no_issues(self):
        report = DiagnosticReport(
            model_pair=ModelPair(base_model="a", trained_model="b"),
        )
        recs = generate_recommendations(report)
        assert len(recs) >= 1
        assert "No specific concerns" in recs[0]

    def test_with_behaviour_issues(self, sample_behaviour_result):
        report = DiagnosticReport(
            model_pair=ModelPair(base_model="a", trained_model="b"),
            behaviour=sample_behaviour_result,
        )
        recs = generate_recommendations(report)
        # Should have recommendations about length, format, strategy
        assert len(recs) >= 1
