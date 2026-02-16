"""Tests for weight-diff command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from afterburn.cli.main import cli
from afterburn.exceptions import AfterburnError
from afterburn.types import (
    DiagnosticReport,
    EmbeddingDrift,
    LayerDiff,
    ModelPair,
    WeightDiffResult,
)


def test_weight_diff_missing_base_flag():
    """Test that missing --base flag causes error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["weight-diff", "--trained", "model/trained"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_weight_diff_missing_trained_flag():
    """Test that missing --trained flag causes error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["weight-diff", "--base", "model/base"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_weight_diff_invalid_device_choice():
    """Test that invalid --device choice causes error."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "weight-diff",
            "--base", "model/base",
            "--trained", "model/trained",
            "--device", "invalid_device",
        ]
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


@patch("afterburn.weight_diff.engine.WeightDiffEngine")
@patch("afterburn.device.auto_detect_device")
def test_weight_diff_valid_invocation(mock_auto_detect, mock_engine_class):
    """Test valid invocation with mocked engine."""
    # Mock device detection
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    # Create mock result
    layer_diff = LayerDiff(
        layer_name="model.layers.0.self_attn.q_proj",
        layer_index=0,
        l2_norm=0.5,
        cosine_similarity=0.999,
        frobenius_norm=0.4,
        relative_change=0.001,
        param_count=1000,
    )
    embedding_drift = EmbeddingDrift(
        input_embedding_l2=0.1,
        input_embedding_cosine=0.998,
        output_embedding_l2=0.2,
        output_embedding_cosine=0.997,
        top_drifted_tokens=[(0, 0.5), (1, 0.4)],
    )
    mock_result = WeightDiffResult(
        layer_diffs=[layer_diff],
        attention_heads=[],
        layernorm_shifts=[],
        embedding_drift=embedding_drift,
        lora_analysis=None,
        total_param_count=10000,
        changed_param_count=500,
    )

    # Mock the engine
    mock_engine = MagicMock()
    mock_engine.run.return_value = mock_result
    mock_engine_class.return_value = mock_engine

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "weight-diff",
            "--base", "model/base",
            "--trained", "model/trained",
            "--device", "cpu",
        ]
    )

    assert result.exit_code == 0
    assert "Weight Diff Results" in result.output
    assert "10,000" in result.output or "10000" in result.output  # total params
    assert "500" in result.output  # changed params

    # Verify engine was instantiated correctly
    mock_engine_class.assert_called_once()
    call_args = mock_engine_class.call_args[0]
    assert isinstance(call_args[0], ModelPair)
    assert call_args[0].base_model == "model/base"
    assert call_args[0].trained_model == "model/trained"

    # Verify run was called
    mock_engine.run.assert_called_once()


@patch("afterburn.weight_diff.engine.WeightDiffEngine")
@patch("afterburn.device.auto_detect_device")
def test_weight_diff_with_output_file(mock_auto_detect, mock_engine_class, tmp_path):
    """Test weight-diff with output file option."""
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    # Create minimal result
    embedding_drift = EmbeddingDrift(
        input_embedding_l2=0.1,
        input_embedding_cosine=0.998,
        output_embedding_l2=None,
        output_embedding_cosine=None,
        top_drifted_tokens=[],
    )
    mock_result = WeightDiffResult(
        layer_diffs=[],
        attention_heads=[],
        layernorm_shifts=[],
        embedding_drift=embedding_drift,
        lora_analysis=None,
        total_param_count=1000,
        changed_param_count=100,
    )

    mock_engine = MagicMock()
    mock_engine.run.return_value = mock_result
    mock_engine_class.return_value = mock_engine

    output_path = tmp_path / "weight_diff.json"

    with patch.object(DiagnosticReport, "save", return_value=output_path) as mock_save:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "weight-diff",
                "--base", "model/base",
                "--trained", "model/trained",
                "--output", str(output_path),
            ]
        )

    assert result.exit_code == 0
    assert "Report saved" in result.output
    # Verify save was called
    mock_save.assert_called_once()


@patch("afterburn.weight_diff.engine.WeightDiffEngine")
@patch("afterburn.device.auto_detect_device")
def test_weight_diff_top_n_option(mock_auto_detect, mock_engine_class):
    """Test --top-n option for number of layers to show."""
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    # Create multiple layer diffs
    layer_diffs = [
        LayerDiff(
            layer_name=f"model.layers.{i}.self_attn.q_proj",
            layer_index=i,
            l2_norm=0.5 - i * 0.1,
            cosine_similarity=0.999,
            frobenius_norm=0.4,
            relative_change=0.001 + i * 0.001,
            param_count=1000,
        )
        for i in range(10)
    ]
    embedding_drift = EmbeddingDrift(
        input_embedding_l2=0.1,
        input_embedding_cosine=0.998,
        output_embedding_l2=None,
        output_embedding_cosine=None,
        top_drifted_tokens=[],
    )
    mock_result = WeightDiffResult(
        layer_diffs=layer_diffs,
        attention_heads=[],
        layernorm_shifts=[],
        embedding_drift=embedding_drift,
        lora_analysis=None,
        total_param_count=10000,
        changed_param_count=1000,
    )

    mock_engine = MagicMock()
    mock_engine.run.return_value = mock_result
    mock_engine_class.return_value = mock_engine

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "weight-diff",
            "--base", "model/base",
            "--trained", "model/trained",
            "--top-n", "3",
        ]
    )

    assert result.exit_code == 0
    assert "Top 3 Changed Layers" in result.output


@patch("afterburn.weight_diff.engine.WeightDiffEngine")
@patch("afterburn.device.auto_detect_device")
def test_weight_diff_afterburn_error_caught(mock_auto_detect, mock_engine_class):
    """Test that AfterburnError is caught and exits with code 1."""
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    mock_engine = MagicMock()
    mock_engine.run.side_effect = AfterburnError("Weight diff failed")
    mock_engine_class.return_value = mock_engine

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "weight-diff",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 1
    assert "Weight diff failed" in result.output


@patch("afterburn.weight_diff.engine.WeightDiffEngine")
@patch("afterburn.device.auto_detect_device")
def test_weight_diff_keyboard_interrupt_caught(mock_auto_detect, mock_engine_class):
    """Test that KeyboardInterrupt is caught and exits with code 130."""
    mock_device_config = MagicMock()
    mock_auto_detect.return_value = mock_device_config

    mock_engine = MagicMock()
    mock_engine.run.side_effect = KeyboardInterrupt()
    mock_engine_class.return_value = mock_engine

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "weight-diff",
            "--base", "model/base",
            "--trained", "model/trained",
        ]
    )

    assert result.exit_code == 130
    assert "Interrupted" in result.output
