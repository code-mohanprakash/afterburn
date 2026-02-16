"""Tests for behaviour command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from afterburn.cli.main import cli
from afterburn.exceptions import AfterburnError
from afterburn.types import (
    BehaviourResult,
    ChainOfThoughtAnalysis,
    DiagnosticReport,
    FormatAnalysis,
    LengthAnalysis,
    ModelPair,
    PromptResult,
    StrategyShiftAnalysis,
)


def test_behaviour_missing_base_flag():
    """Test that missing --base flag causes error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["behaviour", "--trained", "model/trained"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_behaviour_missing_trained_flag():
    """Test that missing --trained flag causes error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["behaviour", "--base", "model/base"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_behaviour_invalid_device_choice():
    """Test that invalid --device choice causes error."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "behaviour",
            "--base", "model/base",
            "--trained", "model/trained",
            "--device", "invalid_device",
        ]
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


@patch("afterburn.behaviour.analyser.BehaviourAnalyser")
@patch("afterburn.device.auto_detect_device")
def test_behaviour_valid_invocation(mock_auto_detect, mock_analyser_class):
    """Test valid invocation with mocked analyser."""
    # Mock device detection
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    # Create mock result
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
    format_analysis = FormatAnalysis()
    strategy_analysis = StrategyShiftAnalysis(
        base_distribution={"direct_answer": 0.7},
        trained_distribution={"step_by_step": 0.8},
        dominant_shift="step_by_step",
        base_entropy=1.2,
        trained_entropy=0.9,
        entropy_change=-0.3,
    )
    cot_analysis = ChainOfThoughtAnalysis()

    mock_result = BehaviourResult(
        base_results=[],
        trained_results=[],
        length_analysis=length_analysis,
        format_analysis=format_analysis,
        strategy_analysis=strategy_analysis,
        cot_analysis=cot_analysis,
    )

    # Mock the analyser
    mock_analyser = MagicMock()
    mock_analyser.run.return_value = mock_result
    mock_analyser_class.return_value = mock_analyser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "behaviour",
            "--base", "model/base",
            "--trained", "model/trained",
            "--device", "cpu",
        ]
    )

    assert result.exit_code == 0
    assert "Behaviour Analysis Results" in result.output
    assert "50.0" in result.output  # base mean
    assert "55.0" in result.output  # trained mean

    # Verify analyser was instantiated correctly
    mock_analyser_class.assert_called_once()
    call_args = mock_analyser_class.call_args[0]
    assert isinstance(call_args[0], ModelPair)
    assert call_args[0].base_model == "model/base"
    assert call_args[0].trained_model == "model/trained"

    # Verify run was called
    mock_analyser.run.assert_called_once()


@patch("afterburn.behaviour.analyser.BehaviourAnalyser")
@patch("afterburn.device.auto_detect_device")
def test_behaviour_with_suites(mock_auto_detect, mock_analyser_class):
    """Test behaviour with multiple --suites options."""
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    # Create minimal result
    length_analysis = LengthAnalysis(
        base_mean=50.0,
        base_median=48.0,
        base_std=10.0,
        trained_mean=50.5,
        trained_median=49.0,
        trained_std=10.5,
        mean_diff=0.5,
        p_value=0.5,
        cohens_d=0.05,
        is_significant=False,
    )
    mock_result = BehaviourResult(
        base_results=[],
        trained_results=[],
        length_analysis=length_analysis,
        format_analysis=FormatAnalysis(),
        strategy_analysis=StrategyShiftAnalysis(),
        cot_analysis=ChainOfThoughtAnalysis(),
    )

    mock_analyser = MagicMock()
    mock_analyser.run.return_value = mock_result
    mock_analyser_class.return_value = mock_analyser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "behaviour",
            "--base", "model/base",
            "--trained", "model/trained",
            "--suites", "math",
            "--suites", "code",
        ]
    )

    assert result.exit_code == 0

    # Verify suites were passed correctly
    call_kwargs = mock_analyser_class.call_args[1]
    assert call_kwargs["suites"] == ["math", "code"]


@patch("afterburn.behaviour.analyser.BehaviourAnalyser")
@patch("afterburn.device.auto_detect_device")
def test_behaviour_with_output_file(mock_auto_detect, mock_analyser_class, tmp_path):
    """Test behaviour with output file option."""
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    # Create minimal result
    length_analysis = LengthAnalysis(
        base_mean=50.0,
        base_median=48.0,
        base_std=10.0,
        trained_mean=50.0,
        trained_median=48.0,
        trained_std=10.0,
        mean_diff=0.0,
        p_value=1.0,
        cohens_d=0.0,
        is_significant=False,
    )
    mock_result = BehaviourResult(
        base_results=[],
        trained_results=[],
        length_analysis=length_analysis,
        format_analysis=FormatAnalysis(),
        strategy_analysis=StrategyShiftAnalysis(),
        cot_analysis=ChainOfThoughtAnalysis(),
    )

    mock_analyser = MagicMock()
    mock_analyser.run.return_value = mock_result
    mock_analyser_class.return_value = mock_analyser

    output_path = tmp_path / "behaviour.json"

    with patch.object(DiagnosticReport, "save", return_value=output_path) as mock_save:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "behaviour",
                "--base", "model/base",
                "--trained", "model/trained",
                "--output", str(output_path),
            ]
        )

    assert result.exit_code == 0
    assert "Report saved" in result.output
    # Verify save was called
    mock_save.assert_called_once()


@patch("afterburn.behaviour.analyser.BehaviourAnalyser")
@patch("afterburn.device.auto_detect_device")
def test_behaviour_afterburn_error_caught(mock_auto_detect, mock_analyser_class):
    """Test that AfterburnError is caught and exits with code 1."""
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    mock_analyser = MagicMock()
    mock_analyser.run.side_effect = AfterburnError("Behaviour analysis failed")
    mock_analyser_class.return_value = mock_analyser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "behaviour",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 1
    assert "Behaviour analysis failed" in result.output


@patch("afterburn.behaviour.analyser.BehaviourAnalyser")
@patch("afterburn.device.auto_detect_device")
def test_behaviour_keyboard_interrupt_caught(mock_auto_detect, mock_analyser_class):
    """Test that KeyboardInterrupt is caught and exits with code 130."""
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    mock_analyser = MagicMock()
    mock_analyser.run.side_effect = KeyboardInterrupt()
    mock_analyser_class.return_value = mock_analyser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "behaviour",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 130
    assert "Interrupted" in result.output
