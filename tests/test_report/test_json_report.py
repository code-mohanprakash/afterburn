"""Tests for JSON report generation."""

import json

import pytest

from afterburn.report.json_report import JSONReport
from afterburn.types import (
    DiagnosticReport,
    ModelPair,
    TrainingMethod,
)


class TestJSONReport:
    def test_minimal_report(self, tmp_path):
        report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="test/base",
                trained_model="test/trained",
                method=TrainingMethod.SFT,
            ),
            summary="Test summary.",
            recommendations=["Test recommendation."],
        )

        output = tmp_path / "report.json"
        JSONReport(report).generate(output)

        assert output.exists()
        data = json.loads(output.read_text())

        assert data["model_pair"]["base_model"] == "test/base"
        assert data["model_pair"]["method"] == "sft"
        assert data["summary"] == "Test summary."
        assert len(data["recommendations"]) == 1
        assert "afterburn_version" in data

    def test_with_reward_hack(self, tmp_path, sample_behaviour_result):
        from afterburn.reward_hack.detector import RewardHackDetector

        detector = RewardHackDetector(sample_behaviour_result)
        hack_result = detector.run()

        report = DiagnosticReport(
            model_pair=ModelPair(base_model="a", trained_model="b"),
            behaviour=sample_behaviour_result,
            reward_hack=hack_result,
            hack_score=hack_result.composite_score,
        )

        output = tmp_path / "report.json"
        JSONReport(report).generate(output)

        data = json.loads(output.read_text())
        assert "reward_hack" in data
        assert "composite_score" in data["reward_hack"]
        assert "length_bias" in data["reward_hack"]

    def test_valid_json(self, tmp_path):
        report = DiagnosticReport(
            model_pair=ModelPair(base_model="x", trained_model="y"),
        )
        output = tmp_path / "report.json"
        JSONReport(report).generate(output)

        # Should be valid JSON
        data = json.loads(output.read_text())
        assert isinstance(data, dict)
