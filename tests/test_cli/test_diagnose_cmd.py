"""Tests for diagnose command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from afterburn.cli.main import cli
from afterburn.exceptions import AfterburnError
from afterburn.types import DiagnosticReport, ModelPair, RiskLevel, TrainingMethod


def test_diagnose_missing_base_flag():
    """Test that missing --base flag causes error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["diagnose", "--trained", "model/trained"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_diagnose_missing_trained_flag():
    """Test that missing --trained flag causes error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["diagnose", "--base", "model/base"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_diagnose_invalid_method_choice():
    """Test that invalid --method choice causes error."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "diagnose",
            "--base", "model/base",
            "--trained", "model/trained",
            "--method", "invalid_method",
        ]
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


def test_diagnose_invalid_device_choice():
    """Test that invalid --device choice causes error."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "diagnose",
            "--base", "model/base",
            "--trained", "model/trained",
            "--device", "invalid_device",
        ]
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


def test_diagnose_invalid_format_choice():
    """Test that invalid --format choice causes error."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "diagnose",
            "--base", "model/base",
            "--trained", "model/trained",
            "--format", "invalid_format",
        ]
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


@patch("afterburn.diagnoser.Diagnoser")
def test_diagnose_valid_options_accepted(mock_diagnoser_class, tmp_path):
    """Test that valid options are accepted and Diagnoser is invoked."""
    # Create a mock report
    model_pair = ModelPair(
        base_model="model/base",
        trained_model="model/trained",
        method=TrainingMethod.DPO
    )
    mock_report = DiagnosticReport(
        model_pair=model_pair,
        summary="Test summary",
        hack_score=25.0,
    )

    # Mock the Diagnoser instance
    mock_diagnoser = MagicMock()
    mock_diagnoser.run.return_value = mock_report
    mock_diagnoser_class.return_value = mock_diagnoser

    # Mock the save method
    output_path = tmp_path / "report.html"
    with patch.object(DiagnosticReport, "save", return_value=output_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diagnose",
                "--base", "model/base",
                "--trained", "model/trained",
                "--method", "dpo",
                "--device", "cpu",
                "--output", str(output_path),
            ]
        )

    assert result.exit_code == 0
    # Verify Diagnoser was instantiated correctly
    mock_diagnoser_class.assert_called_once()
    call_kwargs = mock_diagnoser_class.call_args[1]
    assert call_kwargs["base_model"] == "model/base"
    assert call_kwargs["trained_model"] == "model/trained"
    assert call_kwargs["method"] == "dpo"
    assert call_kwargs["device"] == "cpu"

    # Verify run was called
    mock_diagnoser.run.assert_called_once()


@patch("afterburn.diagnoser.Diagnoser")
def test_diagnose_format_auto_detection(mock_diagnoser_class, tmp_path):
    """Test that --format is auto-detected from output path extension."""
    model_pair = ModelPair(
        base_model="model/base",
        trained_model="model/trained",
    )
    mock_report = DiagnosticReport(
        model_pair=model_pair,
        summary="Test summary",
    )

    mock_diagnoser = MagicMock()
    mock_diagnoser.run.return_value = mock_report
    mock_diagnoser_class.return_value = mock_diagnoser

    output_path = tmp_path / "report.json"

    with patch.object(DiagnosticReport, "save", return_value=output_path) as mock_save:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diagnose",
                "--base", "model/base",
                "--trained", "model/trained",
                "--output", str(output_path),
            ]
        )

    assert result.exit_code == 0
    # save should be called with None for auto-detection
    mock_save.assert_called_once()
    # First arg is path, second is fmt
    assert mock_save.call_args[0][0] == Path(str(output_path))
    assert mock_save.call_args[1]["fmt"] is None


@patch("afterburn.diagnoser.Diagnoser")
def test_diagnose_explicit_format(mock_diagnoser_class, tmp_path):
    """Test that explicit --format overrides extension."""
    from afterburn.types import ReportFormat

    model_pair = ModelPair(
        base_model="model/base",
        trained_model="model/trained",
    )
    mock_report = DiagnosticReport(
        model_pair=model_pair,
        summary="Test summary",
    )

    mock_diagnoser = MagicMock()
    mock_diagnoser.run.return_value = mock_report
    mock_diagnoser_class.return_value = mock_diagnoser

    output_path = tmp_path / "report.txt"

    with patch.object(DiagnosticReport, "save", return_value=output_path) as mock_save:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diagnose",
                "--base", "model/base",
                "--trained", "model/trained",
                "--output", str(output_path),
                "--format", "json",
            ]
        )

    assert result.exit_code == 0
    # save should be called with explicit format
    mock_save.assert_called_once()
    assert mock_save.call_args[0][0] == Path(str(output_path))
    assert mock_save.call_args[1]["fmt"] == ReportFormat.JSON


@patch("afterburn.diagnoser.Diagnoser")
def test_diagnose_afterburn_error_caught(mock_diagnoser_class):
    """Test that AfterburnError is caught and exits with code 1."""
    mock_diagnoser = MagicMock()
    mock_diagnoser.run.side_effect = AfterburnError("Test error message")
    mock_diagnoser_class.return_value = mock_diagnoser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "diagnose",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 1
    assert "Test error message" in result.output


@patch("afterburn.diagnoser.Diagnoser")
def test_diagnose_keyboard_interrupt_caught(mock_diagnoser_class):
    """Test that KeyboardInterrupt is caught and exits with code 130."""
    mock_diagnoser = MagicMock()
    mock_diagnoser.run.side_effect = KeyboardInterrupt()
    mock_diagnoser_class.return_value = mock_diagnoser

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "diagnose",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 130
    assert "Interrupted" in result.output


@patch("afterburn.diagnoser.Diagnoser")
def test_diagnose_with_all_options(mock_diagnoser_class, tmp_path):
    """Test diagnose with all optional parameters."""
    model_pair = ModelPair(
        base_model="model/base",
        trained_model="model/trained",
        method=TrainingMethod.RLVR
    )
    mock_report = DiagnosticReport(
        model_pair=model_pair,
        summary="Test summary",
    )

    mock_diagnoser = MagicMock()
    mock_diagnoser.run.return_value = mock_report
    mock_diagnoser_class.return_value = mock_diagnoser

    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: config")
    output_path = tmp_path / "report.html"

    with patch.object(DiagnosticReport, "save", return_value=output_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diagnose",
                "--base", "model/base",
                "--trained", "model/trained",
                "--method", "rlvr",
                "--device", "mps",
                "--suites", "math",
                "--suites", "code",
                "--modules", "weight_diff",
                "--modules", "behaviour",
                "--config", str(config_path),
                "--output", str(output_path),
                "--format", "html",
            ]
        )

    assert result.exit_code == 0

    # Verify all parameters were passed correctly
    call_kwargs = mock_diagnoser_class.call_args[1]
    assert call_kwargs["base_model"] == "model/base"
    assert call_kwargs["trained_model"] == "model/trained"
    assert call_kwargs["method"] == "rlvr"
    assert call_kwargs["device"] == "mps"
    assert call_kwargs["suites"] == ["math", "code"]
    assert call_kwargs["modules"] == ["weight_diff", "behaviour"]
    assert call_kwargs["config_path"] == str(config_path)
