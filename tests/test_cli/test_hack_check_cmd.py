"""Tests for hack-check command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from afterburn.cli.main import cli
from afterburn.exceptions import AfterburnError
from afterburn.types import (
    DiagnosticReport,
    FormatGamingResult,
    LengthBiasResult,
    ModelPair,
    RewardHackResult,
    RiskLevel,
    StrategyCollapseResult,
    SycophancyResult,
    TrainingMethod,
)


def test_hack_check_missing_base_flag():
    """Test that missing --base flag causes error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["hack-check", "--trained", "model/trained"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_hack_check_missing_trained_flag():
    """Test that missing --trained flag causes error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["hack-check", "--base", "model/base"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_hack_check_invalid_method_choice():
    """Test that invalid --method choice causes error."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hack-check",
            "--base", "model/base",
            "--trained", "model/trained",
            "--method", "invalid_method",
        ]
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


def test_hack_check_invalid_device_choice():
    """Test that invalid --device choice causes error."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hack-check",
            "--base", "model/base",
            "--trained", "model/trained",
            "--device", "invalid_device",
        ]
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


@patch("afterburn.diagnoser.Diagnoser")
def test_hack_check_valid_invocation(mock_diagnoser_class):
    """Test valid invocation with mocked diagnoser."""
    # Create mock reward hack result
    length_bias = LengthBiasResult(
        score=30.0,
        cohens_d=0.5,
        p_value=0.01,
        mean_length_ratio=1.2,
        is_flagged=False,
        detail="Length bias detected",
    )
    format_gaming = FormatGamingResult(
        score=20.0,
        is_flagged=False,
        detail="No format gaming",
    )
    strategy_collapse = StrategyCollapseResult(
        score=40.0,
        base_entropy=1.5,
        trained_entropy=0.5,
        entropy_drop=1.0,
        dominant_strategy="direct_answer",
        is_flagged=True,
        detail="Strategy collapse detected",
    )
    sycophancy = SycophancyResult(
        score=25.0,
        is_flagged=False,
        detail="No sycophancy",
    )

    reward_hack = RewardHackResult(
        length_bias=length_bias,
        format_gaming=format_gaming,
        strategy_collapse=strategy_collapse,
        sycophancy=sycophancy,
        composite_score=28.75,
        risk_level=RiskLevel.MODERATE,
        flags=["Strategy collapse detected"],
    )

    model_pair = ModelPair(
        base_model="model/base",
        trained_model="model/trained",
        method=TrainingMethod.RLHF,
    )
    mock_report = DiagnosticReport(
        model_pair=model_pair,
        reward_hack=reward_hack,
        hack_score=28.75,
    )

    # Mock the Diagnoser instance
    mock_diagnoser = MagicMock()
    mock_diagnoser.run.return_value = mock_report
    mock_diagnoser_class.return_value = mock_diagnoser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hack-check",
            "--base", "model/base",
            "--trained", "model/trained",
            "--method", "rlhf",
        ]
    )

    assert result.exit_code == 0
    # Composite score is displayed rounded (could be 28, 28.8, 29, etc)
    assert "MODERATE" in result.output  # risk level
    assert "Length Bias" in result.output
    assert "Format Gaming" in result.output
    assert "Strategy Collapse" in result.output
    assert "Sycophancy" in result.output

    # Verify Diagnoser was instantiated correctly
    mock_diagnoser_class.assert_called_once()
    call_kwargs = mock_diagnoser_class.call_args[1]
    assert call_kwargs["base_model"] == "model/base"
    assert call_kwargs["trained_model"] == "model/trained"
    assert call_kwargs["method"] == "rlhf"
    assert call_kwargs["modules"] == ["behaviour", "reward_hack"]

    # Verify run was called
    mock_diagnoser.run.assert_called_once()


@patch("afterburn.diagnoser.Diagnoser")
def test_hack_check_with_suites(mock_diagnoser_class):
    """Test hack-check with multiple --suites options."""
    length_bias = LengthBiasResult(
        score=10.0,
        cohens_d=0.2,
        p_value=0.5,
        mean_length_ratio=1.0,
        is_flagged=False,
    )
    format_gaming = FormatGamingResult(score=10.0)
    strategy_collapse = StrategyCollapseResult(
        score=10.0,
        base_entropy=1.0,
        trained_entropy=0.9,
        entropy_drop=0.1,
    )
    sycophancy = SycophancyResult(score=10.0)

    reward_hack = RewardHackResult(
        length_bias=length_bias,
        format_gaming=format_gaming,
        strategy_collapse=strategy_collapse,
        sycophancy=sycophancy,
        composite_score=10.0,
        risk_level=RiskLevel.LOW,
    )

    model_pair = ModelPair(
        base_model="model/base",
        trained_model="model/trained",
    )
    mock_report = DiagnosticReport(
        model_pair=model_pair,
        reward_hack=reward_hack,
        hack_score=10.0,
    )

    mock_diagnoser = MagicMock()
    mock_diagnoser.run.return_value = mock_report
    mock_diagnoser_class.return_value = mock_diagnoser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hack-check",
            "--base", "model/base",
            "--trained", "model/trained",
            "--suites", "math",
            "--suites", "safety",
        ]
    )

    assert result.exit_code == 0

    # Verify suites were passed correctly
    call_kwargs = mock_diagnoser_class.call_args[1]
    assert call_kwargs["suites"] == ["math", "safety"]


@patch("afterburn.diagnoser.Diagnoser")
def test_hack_check_with_output_file(mock_diagnoser_class, tmp_path):
    """Test hack-check with output file option."""
    length_bias = LengthBiasResult(
        score=15.0,
        cohens_d=0.3,
        p_value=0.2,
        mean_length_ratio=1.1,
        is_flagged=False,
    )
    format_gaming = FormatGamingResult(score=15.0)
    strategy_collapse = StrategyCollapseResult(
        score=15.0,
        base_entropy=1.0,
        trained_entropy=0.85,
        entropy_drop=0.15,
    )
    sycophancy = SycophancyResult(score=15.0)

    reward_hack = RewardHackResult(
        length_bias=length_bias,
        format_gaming=format_gaming,
        strategy_collapse=strategy_collapse,
        sycophancy=sycophancy,
        composite_score=15.0,
        risk_level=RiskLevel.LOW,
    )

    model_pair = ModelPair(
        base_model="model/base",
        trained_model="model/trained",
    )
    mock_report = DiagnosticReport(
        model_pair=model_pair,
        reward_hack=reward_hack,
        hack_score=15.0,
    )

    mock_diagnoser = MagicMock()
    mock_diagnoser.run.return_value = mock_report
    mock_diagnoser_class.return_value = mock_diagnoser

    output_path = tmp_path / "hack_check.json"

    with patch.object(DiagnosticReport, "save", return_value=output_path) as mock_save:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "hack-check",
                "--base", "model/base",
                "--trained", "model/trained",
                "--output", str(output_path),
            ]
        )

    assert result.exit_code == 0
    assert "Report saved" in result.output
    # Verify save was called
    mock_save.assert_called_once()


@patch("afterburn.diagnoser.Diagnoser")
def test_hack_check_with_flags(mock_diagnoser_class):
    """Test hack-check displays flags correctly."""
    length_bias = LengthBiasResult(
        score=80.0,
        cohens_d=1.5,
        p_value=0.001,
        mean_length_ratio=2.0,
        is_flagged=True,
        detail="Severe length bias",
    )
    format_gaming = FormatGamingResult(
        score=70.0,
        is_flagged=True,
        detail="Format gaming detected",
    )
    strategy_collapse = StrategyCollapseResult(
        score=60.0,
        base_entropy=2.0,
        trained_entropy=0.3,
        entropy_drop=1.7,
        is_flagged=True,
        detail="Severe strategy collapse",
    )
    sycophancy = SycophancyResult(
        score=50.0,
        is_flagged=False,
    )

    reward_hack = RewardHackResult(
        length_bias=length_bias,
        format_gaming=format_gaming,
        strategy_collapse=strategy_collapse,
        sycophancy=sycophancy,
        composite_score=65.0,
        risk_level=RiskLevel.HIGH,
        flags=[
            "Severe length bias",
            "Format gaming detected",
            "Severe strategy collapse",
        ],
    )

    model_pair = ModelPair(
        base_model="model/base",
        trained_model="model/trained",
    )
    mock_report = DiagnosticReport(
        model_pair=model_pair,
        reward_hack=reward_hack,
        hack_score=65.0,
    )

    mock_diagnoser = MagicMock()
    mock_diagnoser.run.return_value = mock_report
    mock_diagnoser_class.return_value = mock_diagnoser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hack-check",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 0
    assert "Flags:" in result.output
    assert "Severe length bias" in result.output
    assert "Format gaming detected" in result.output
    assert "Severe strategy collapse" in result.output


@patch("afterburn.diagnoser.Diagnoser")
def test_hack_check_afterburn_error_caught(mock_diagnoser_class):
    """Test that AfterburnError is caught and exits with code 1."""
    mock_diagnoser = MagicMock()
    mock_diagnoser.run.side_effect = AfterburnError("Hack check failed")
    mock_diagnoser_class.return_value = mock_diagnoser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hack-check",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 1
    assert "Hack check failed" in result.output


@patch("afterburn.diagnoser.Diagnoser")
def test_hack_check_keyboard_interrupt_caught(mock_diagnoser_class):
    """Test that KeyboardInterrupt is caught and exits with code 130."""
    mock_diagnoser = MagicMock()
    mock_diagnoser.run.side_effect = KeyboardInterrupt()
    mock_diagnoser_class.return_value = mock_diagnoser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hack-check",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 130
    assert "Interrupted" in result.output
