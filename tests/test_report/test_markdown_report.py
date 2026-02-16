"""Tests for Markdown report generation."""

from __future__ import annotations

import pytest

from afterburn.report.markdown_report import MarkdownReport
from afterburn.types import (
    BehaviourResult,
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
)
from afterburn.version import __version__


@pytest.fixture
def minimal_report():
    """Create a minimal DiagnosticReport for testing."""
    model_pair = ModelPair(
        base_model="test/base",
        trained_model="test/trained",
        method=TrainingMethod.DPO,
    )
    return DiagnosticReport(
        model_pair=model_pair,
        summary="Test markdown report",
    )


@pytest.fixture
def full_report():
    """Create a full DiagnosticReport with all components."""
    model_pair = ModelPair(
        base_model="test/base",
        trained_model="test/trained",
        method=TrainingMethod.RLVR,
    )

    # Weight diff
    embedding_drift = EmbeddingDrift(
        input_embedding_l2=0.15,
        input_embedding_cosine=0.995,
        output_embedding_l2=None,
        output_embedding_cosine=None,
        top_drifted_tokens=[],
    )
    weight_diff = WeightDiffResult(
        layer_diffs=[],
        attention_heads=[],
        layernorm_shifts=[],
        embedding_drift=embedding_drift,
        lora_analysis=None,
        total_param_count=5000,
        changed_param_count=250,
    )

    # Behaviour
    length_analysis = LengthAnalysis(
        base_mean=60.0,
        base_median=58.0,
        base_std=12.0,
        trained_mean=65.0,
        trained_median=62.0,
        trained_std=13.0,
        mean_diff=5.0,
        p_value=0.02,
        cohens_d=0.4,
        is_significant=True,
    )
    strategy_analysis = StrategyShiftAnalysis(
        base_distribution={"direct_answer": 0.6},
        trained_distribution={"step_by_step": 0.7},
        dominant_shift="step_by_step",
        base_entropy=1.3,
        trained_entropy=1.0,
        entropy_change=-0.3,
    )
    cot_analysis = ChainOfThoughtAnalysis(
        base_avg_steps=2.5,
        trained_avg_steps=3.2,
        step_count_change=0.7,
    )
    behaviour = BehaviourResult(
        base_results=[],
        trained_results=[],
        length_analysis=length_analysis,
        format_analysis=FormatAnalysis(),
        strategy_analysis=strategy_analysis,
        cot_analysis=cot_analysis,
    )

    # Reward hack
    length_bias = LengthBiasResult(
        score=40.0,
        cohens_d=0.6,
        p_value=0.005,
        mean_length_ratio=1.3,
        is_flagged=True,
        detail="Moderate length bias",
    )
    format_gaming = FormatGamingResult(
        score=30.0,
        is_flagged=False,
        detail="Minor format gaming",
    )
    strategy_collapse = StrategyCollapseResult(
        score=50.0,
        base_entropy=1.8,
        trained_entropy=0.5,
        entropy_drop=1.3,
        is_flagged=True,
        detail="Strategy collapse detected",
    )
    sycophancy = SycophancyResult(
        score=20.0,
        is_flagged=False,
        detail="No sycophancy",
    )

    reward_hack = RewardHackResult(
        length_bias=length_bias,
        format_gaming=format_gaming,
        strategy_collapse=strategy_collapse,
        sycophancy=sycophancy,
        composite_score=35.0,
        risk_level=RiskLevel.MODERATE,
        flags=["Moderate length bias", "Strategy collapse detected"],
    )

    return DiagnosticReport(
        model_pair=model_pair,
        weight_diff=weight_diff,
        behaviour=behaviour,
        reward_hack=reward_hack,
        summary="Full markdown test report",
        hack_score=35.0,
        recommendations=["Recommendation 1", "Recommendation 2"],
    )


def test_markdown_report_generates_file(minimal_report, tmp_path):
    """Test that MarkdownReport generates a .md file."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(minimal_report)

    result_path = md_report.generate(output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.is_file()


def test_markdown_report_contains_header(minimal_report, tmp_path):
    """Test that markdown report contains header sections."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(minimal_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    assert "# Afterburn Diagnostic Report" in content
    assert "## Executive Summary" in content


def test_markdown_report_contains_model_info(minimal_report, tmp_path):
    """Test that markdown report contains model information."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(minimal_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    assert "test/base" in content
    assert "test/trained" in content
    assert "DPO" in content  # method


def test_markdown_report_contains_version(minimal_report, tmp_path):
    """Test that markdown report includes version."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(minimal_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    assert __version__ in content


def test_markdown_report_contains_timestamp(minimal_report, tmp_path):
    """Test that markdown report includes timestamp."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(minimal_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    # Should contain a date in YYYY-MM-DD format
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", content) is not None


def test_markdown_report_creates_parent_directories(minimal_report, tmp_path):
    """Test that parent directories are created if needed."""
    output_path = tmp_path / "reports" / "nested" / "report.md"
    md_report = MarkdownReport(minimal_report)

    result_path = md_report.generate(output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.parent.exists()


def test_markdown_report_handles_minimal_data(minimal_report, tmp_path):
    """Test markdown generation with minimal data."""
    assert minimal_report.weight_diff is None
    assert minimal_report.behaviour is None
    assert minimal_report.reward_hack is None

    output_path = tmp_path / "minimal.md"
    md_report = MarkdownReport(minimal_report)

    result_path = md_report.generate(output_path)

    assert result_path.exists()
    content = output_path.read_text()

    # Should still have basic structure
    assert "# Afterburn Diagnostic Report" in content
    assert "Test markdown report" in content


def test_markdown_report_with_weight_diff(full_report, tmp_path):
    """Test markdown report includes weight diff section."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(full_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    assert "## Weight Diff Analysis" in content
    assert "5,000" in content or "5000" in content  # total params
    assert "250" in content  # changed params
    assert "### Embedding Drift" in content


def test_markdown_report_with_behaviour(full_report, tmp_path):
    """Test markdown report includes behaviour section."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(full_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    assert "## Behavioural Analysis" in content
    assert "### Output Length" in content
    assert "60.0" in content  # base mean
    assert "65.0" in content  # trained mean
    assert "### Strategy Shift" in content
    assert "step_by_step" in content


def test_markdown_report_with_reward_hack(full_report, tmp_path):
    """Test markdown report includes reward hack section."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(full_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    assert "## Reward Hacking Assessment" in content
    assert "35" in content  # composite score
    assert "MODERATE" in content  # risk level
    assert "Length Bias" in content
    assert "Format Gaming" in content
    assert "Strategy Collapse" in content
    assert "Sycophancy" in content


def test_markdown_report_with_recommendations(full_report, tmp_path):
    """Test markdown report includes recommendations section."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(full_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    assert "## Recommendations" in content
    assert "Recommendation 1" in content
    assert "Recommendation 2" in content


def test_markdown_report_with_flags(full_report, tmp_path):
    """Test markdown report includes flags."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(full_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    assert "### Flags" in content
    assert "Moderate length bias" in content
    assert "Strategy collapse detected" in content


def test_markdown_report_has_valid_markdown_tables(full_report, tmp_path):
    """Test that markdown report contains valid table syntax."""
    output_path = tmp_path / "report.md"
    md_report = MarkdownReport(full_report)

    md_report.generate(output_path)
    content = output_path.read_text()

    # Check for table structure (header separator)
    assert "|----" in content or "|---" in content
    # Check for table rows
    assert content.count("|") > 10  # Should have multiple table rows
