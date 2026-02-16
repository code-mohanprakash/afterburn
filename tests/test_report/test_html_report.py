"""Tests for HTML report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from afterburn.exceptions import ReportGenerationError
from afterburn.report.html_report import HTMLReport
from afterburn.types import (
    ChainOfThoughtAnalysis,
    DiagnosticReport,
    EmbeddingDrift,
    FormatAnalysis,
    FormatGamingResult,
    LengthAnalysis,
    LengthBiasResult,
    ModelPair,
    RewardHackResult,
    RiskLevel,
    StrategyCollapseResult,
    StrategyShiftAnalysis,
    SycophancyResult,
    TrainingMethod,
    WeightDiffResult,
    BehaviourResult,
)
from afterburn.version import __version__


@pytest.fixture
def minimal_report():
    """Create a minimal DiagnosticReport for testing."""
    model_pair = ModelPair(
        base_model="test/base",
        trained_model="test/trained",
        method=TrainingMethod.SFT,
    )
    return DiagnosticReport(
        model_pair=model_pair,
        summary="Test diagnostic report",
    )


@pytest.fixture
def full_report():
    """Create a full DiagnosticReport with all components."""
    model_pair = ModelPair(
        base_model="test/base",
        trained_model="test/trained",
        method=TrainingMethod.RLHF,
    )

    # Weight diff
    embedding_drift = EmbeddingDrift(
        input_embedding_l2=0.1,
        input_embedding_cosine=0.998,
        output_embedding_l2=0.2,
        output_embedding_cosine=0.997,
        top_drifted_tokens=[(0, 0.5)],
    )
    weight_diff = WeightDiffResult(
        layer_diffs=[],
        attention_heads=[],
        layernorm_shifts=[],
        embedding_drift=embedding_drift,
        lora_analysis=None,
        total_param_count=1000,
        changed_param_count=100,
    )

    # Behaviour
    length_analysis = LengthAnalysis(
        base_mean=50.0,
        base_median=48.0,
        base_std=10.0,
        trained_mean=55.0,
        trained_median=52.0,
        trained_std=11.0,
        mean_diff=5.0,
        p_value=0.01,
        cohens_d=0.5,
        is_significant=True,
    )
    behaviour = BehaviourResult(
        base_results=[],
        trained_results=[],
        length_analysis=length_analysis,
        format_analysis=FormatAnalysis(),
        strategy_analysis=StrategyShiftAnalysis(),
        cot_analysis=ChainOfThoughtAnalysis(),
    )

    # Reward hack
    length_bias = LengthBiasResult(
        score=30.0,
        cohens_d=0.5,
        p_value=0.01,
        mean_length_ratio=1.2,
        is_flagged=False,
    )
    format_gaming = FormatGamingResult(score=20.0)
    strategy_collapse = StrategyCollapseResult(
        score=15.0,
        base_entropy=1.5,
        trained_entropy=1.3,
        entropy_drop=0.2,
    )
    sycophancy = SycophancyResult(score=25.0)

    reward_hack = RewardHackResult(
        length_bias=length_bias,
        format_gaming=format_gaming,
        strategy_collapse=strategy_collapse,
        sycophancy=sycophancy,
        composite_score=22.5,
        risk_level=RiskLevel.LOW,
    )

    return DiagnosticReport(
        model_pair=model_pair,
        weight_diff=weight_diff,
        behaviour=behaviour,
        reward_hack=reward_hack,
        summary="Full test diagnostic report",
        hack_score=22.5,
    )


def test_html_report_generates_valid_file(minimal_report, tmp_path):
    """Test that HTMLReport generates a valid HTML file."""
    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    result_path = html_report.generate(output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.is_file()


def test_html_report_contains_version(minimal_report, tmp_path):
    """Test that generated HTML includes version."""
    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    html_report.generate(output_path)
    content = output_path.read_text()

    assert __version__ in content


def test_html_report_contains_timestamp(minimal_report, tmp_path):
    """Test that generated HTML includes timestamp."""
    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    html_report.generate(output_path)
    content = output_path.read_text()

    # Should contain a date in YYYY-MM-DD format or similar
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", content) is not None


def test_html_report_creates_parent_directories(minimal_report, tmp_path):
    """Test that parent directories are created if needed."""
    output_path = tmp_path / "nested" / "dir" / "report.html"
    html_report = HTMLReport(minimal_report)

    result_path = html_report.generate(output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.parent.exists()


def test_html_report_contains_model_info(minimal_report, tmp_path):
    """Test that generated HTML includes model information."""
    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    html_report.generate(output_path)
    content = output_path.read_text()

    assert "test/base" in content
    assert "test/trained" in content


def test_html_report_handles_no_weight_diff(minimal_report, tmp_path):
    """Test HTML generation with no weight_diff data."""
    # minimal_report has no weight_diff
    assert minimal_report.weight_diff is None

    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    result_path = html_report.generate(output_path)

    assert result_path.exists()
    # Should not crash, should handle gracefully


def test_html_report_handles_no_behaviour(minimal_report, tmp_path):
    """Test HTML generation with no behaviour data."""
    # minimal_report has no behaviour
    assert minimal_report.behaviour is None

    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    result_path = html_report.generate(output_path)

    assert result_path.exists()


def test_html_report_handles_no_reward_hack(minimal_report, tmp_path):
    """Test HTML generation with no reward_hack data."""
    # minimal_report has no reward_hack
    assert minimal_report.reward_hack is None

    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    result_path = html_report.generate(output_path)

    assert result_path.exists()


def test_html_report_with_full_data(full_report, tmp_path):
    """Test HTML generation with all data populated."""
    output_path = tmp_path / "full_report.html"
    html_report = HTMLReport(full_report)

    result_path = html_report.generate(output_path)

    assert result_path.exists()
    content = output_path.read_text()

    # Check for presence of various sections
    assert "test/base" in content
    assert "test/trained" in content
    # Should have some indication of the data being present
    assert len(content) > 1000  # Full report should be substantial


def test_html_report_contains_summary(minimal_report, tmp_path):
    """Test that generated HTML includes summary."""
    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    html_report.generate(output_path)
    content = output_path.read_text()

    assert "Test diagnostic report" in content


def test_html_report_invalid_template_raises_error(minimal_report, tmp_path, monkeypatch):
    """Test that missing template raises ReportGenerationError."""
    # Simulate missing template by pointing to wrong directory
    from afterburn.report import html_report

    original_template_dir = html_report.TEMPLATE_DIR
    monkeypatch.setattr(html_report, "TEMPLATE_DIR", tmp_path / "nonexistent")

    output_path = tmp_path / "report.html"
    html_report_obj = HTMLReport(minimal_report)

    with pytest.raises(ReportGenerationError):
        html_report_obj.generate(output_path)

    # Restore for other tests
    monkeypatch.setattr(html_report, "TEMPLATE_DIR", original_template_dir)


def test_html_report_is_valid_html(minimal_report, tmp_path):
    """Test that generated file contains basic HTML structure."""
    output_path = tmp_path / "report.html"
    html_report = HTMLReport(minimal_report)

    html_report.generate(output_path)
    content = output_path.read_text()

    # Basic HTML validation
    assert "<html" in content.lower()
    assert "</html>" in content.lower()
    assert "<body" in content.lower()
    assert "</body>" in content.lower()
