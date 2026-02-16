"""Tests for shared types and dataclasses."""

from pathlib import Path

import pytest

from afterburn.types import (
    LayerDiff,
    ModelPair,
    ReportFormat,
    RiskLevel,
    TrainingMethod,
    _detect_format,
)


class TestTrainingMethod:
    def test_all_values(self):
        assert TrainingMethod.SFT.value == "sft"
        assert TrainingMethod.RLVR.value == "rlvr"
        assert TrainingMethod.UNKNOWN.value == "unknown"

    def test_from_string(self):
        assert TrainingMethod("sft") == TrainingMethod.SFT
        assert TrainingMethod("rlvr") == TrainingMethod.RLVR

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            TrainingMethod("invalid_method")


class TestRiskLevel:
    def test_from_score_low(self):
        assert RiskLevel.from_score(0) == RiskLevel.LOW
        assert RiskLevel.from_score(25) == RiskLevel.LOW

    def test_from_score_moderate(self):
        assert RiskLevel.from_score(26) == RiskLevel.MODERATE
        assert RiskLevel.from_score(50) == RiskLevel.MODERATE

    def test_from_score_high(self):
        assert RiskLevel.from_score(51) == RiskLevel.HIGH
        assert RiskLevel.from_score(75) == RiskLevel.HIGH

    def test_from_score_critical(self):
        assert RiskLevel.from_score(76) == RiskLevel.CRITICAL
        assert RiskLevel.from_score(100) == RiskLevel.CRITICAL


class TestModelPair:
    def test_creation(self):
        pair = ModelPair(base_model="base", trained_model="trained")
        assert pair.base_model == "base"
        assert pair.method == TrainingMethod.UNKNOWN

    def test_frozen(self):
        pair = ModelPair(base_model="a", trained_model="b")
        with pytest.raises(AttributeError):
            pair.base_model = "c"


class TestLayerDiff:
    def test_creation(self):
        ld = LayerDiff(
            layer_name="layer_0",
            layer_index=0,
            l2_norm=1.5,
            cosine_similarity=0.99,
            frobenius_norm=2.0,
            relative_change=0.01,
            param_count=1000,
        )
        assert ld.layer_name == "layer_0"
        assert ld.l2_norm == 1.5


class TestDetectFormat:
    def test_html(self):
        assert _detect_format(Path("report.html")) == ReportFormat.HTML
        assert _detect_format(Path("report.htm")) == ReportFormat.HTML

    def test_json(self):
        assert _detect_format(Path("report.json")) == ReportFormat.JSON

    def test_markdown(self):
        assert _detect_format(Path("report.md")) == ReportFormat.MARKDOWN

    def test_pdf(self):
        assert _detect_format(Path("report.pdf")) == ReportFormat.PDF

    def test_unknown_defaults_to_html(self):
        assert _detect_format(Path("report.xyz")) == ReportFormat.HTML
