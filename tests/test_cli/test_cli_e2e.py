"""End-to-end CLI tests — config integration, path validation, edge cases.

These tests verify the full data flow from CLI flags through the Diagnoser
into sub-modules, covering gaps not addressed by per-command unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from afterburn.cli.main import cli
from afterburn.types import (
    BehaviourResult,
    ChainOfThoughtAnalysis,
    DiagnosticReport,
    FormatAnalysis,
    LengthAnalysis,
    ModelPair,
    PromptResult,
    RewardHackResult,
    RiskLevel,
    StrategyShiftAnalysis,
    TrainingMethod,
)

# ─── Helpers ────────────────────────────────────────────────────────────


def _make_prompt_results(
    category: str, n: int, prefix: str = "base"
) -> list[PromptResult]:
    """Generate synthetic prompt results for testing."""
    return [
        PromptResult(
            prompt_id=f"{prefix}_{category}_{i}",
            prompt_text=f"Test prompt {i} for {category}",
            category=category,
            output_text=f"Response {i} " * (10 if prefix == "base" else 20),
            output_tokens=50 if prefix == "base" else 80,
            generation_time_ms=100.0,
            expected_answer=f"answer_{i}",
        )
        for i in range(n)
    ]


def _make_behaviour_result() -> BehaviourResult:
    """Create a synthetic BehaviourResult for pipeline tests."""
    base_results = _make_prompt_results("math", 5, "base") + _make_prompt_results(
        "code", 5, "base"
    )
    trained_results = _make_prompt_results(
        "math", 5, "trained"
    ) + _make_prompt_results("code", 5, "trained")

    return BehaviourResult(
        base_results=base_results,
        trained_results=trained_results,
        length_analysis=LengthAnalysis(
            base_mean=50.0,
            base_median=48.0,
            base_std=10.0,
            trained_mean=80.0,
            trained_median=78.0,
            trained_std=12.0,
            mean_diff=30.0,
            p_value=0.001,
            cohens_d=2.5,
            is_significant=True,
        ),
        format_analysis=FormatAnalysis(),
        strategy_analysis=StrategyShiftAnalysis(),
        cot_analysis=ChainOfThoughtAnalysis(),
    )


# ─── Unknown / invalid subcommand ──────────────────────────────────────


class TestUnknownSubcommand:
    """Verify that invalid subcommands are rejected."""

    def test_unknown_subcommand_rejected(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["nonexistent-command"])

        assert result.exit_code != 0
        assert "No such command" in result.output or "Error" in result.output

    def test_empty_invocation_shows_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        assert "Afterburn" in result.output
        assert "diagnose" in result.output


# ─── Config YAML integration via CLI ────────────────────────────────────


class TestConfigIntegrationViaCLI:
    """Verify that --config with a YAML file flows through to sub-modules."""

    def test_diagnose_with_custom_config_thresholds(self, tmp_path):
        """Custom config thresholds are forwarded to Diagnoser."""
        config_yaml = tmp_path / "custom.yaml"
        config_yaml.write_text(
            "device: cpu\n"
            "behaviour:\n"
            "  significance_level: 0.01\n"
            "  effect_size_threshold: 0.5\n"
            "  max_new_tokens: 256\n"
            "reward_hack:\n"
            "  thresholds:\n"
            "    length_bias_cohens_d: 1.0\n"
            "    length_ratio_concern: 1.3\n"
            "    per_category_bias: 0.4\n"
            "    format_increase_ratio: 2.0\n"
            "    format_min_rate: 0.1\n"
            "    category_variance: 0.3\n"
            "    strategy_entropy_drop: 0.3\n"
            "    strategy_dominant_fraction: 0.6\n"
            "    sycophancy_increase: 0.15\n"
            "    sycophancy_pushback_drop: 0.15\n"
            "    sycophancy_consistency_drop: 0.2\n"
        )

        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test summary",
        )

        with patch("afterburn.diagnoser.Diagnoser") as mock_cls:
            mock_diag = MagicMock()
            mock_diag.run.return_value = mock_report
            mock_cls.return_value = mock_diag

            output = tmp_path / "report.html"
            with patch.object(DiagnosticReport, "save", return_value=output):
                runner = CliRunner()
                result = runner.invoke(
                    cli,
                    [
                        "diagnose",
                        "--base",
                        "model/base",
                        "--trained",
                        "model/trained",
                        "--config",
                        str(config_yaml),
                        "--output",
                        str(output),
                    ],
                )

            assert result.exit_code == 0
            # Verify config_path was forwarded
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["config_path"] == str(config_yaml)

    def test_diagnose_with_malformed_yaml_exits_with_error(self, tmp_path):
        """Malformed YAML config causes a clean error exit."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("device: cpu\n  bad_indent: yes\n  :")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diagnose",
                "--base",
                "model/base",
                "--trained",
                "model/trained",
                "--config",
                str(bad_config),
            ],
        )

        assert result.exit_code != 0

    def test_diagnose_with_invalid_threshold_in_config(self, tmp_path):
        """Config with invalid threshold (negative) causes error."""
        config_yaml = tmp_path / "invalid.yaml"
        config_yaml.write_text(
            "weight_diff:\n"
            "  relative_change_threshold: -0.5\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diagnose",
                "--base",
                "model/base",
                "--trained",
                "model/trained",
                "--config",
                str(config_yaml),
            ],
        )

        assert result.exit_code != 0

    def test_diagnose_with_nonexistent_config_file(self, tmp_path):
        """Non-existent config file path causes error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "diagnose",
                "--base",
                "model/base",
                "--trained",
                "model/trained",
                "--config",
                str(tmp_path / "does_not_exist.yaml"),
            ],
        )

        # Click validates --config with exists=True, so it rejects immediately
        assert result.exit_code != 0

    def test_diagnose_with_empty_yaml_uses_defaults(self, tmp_path):
        """Empty YAML config file falls back to defaults."""
        config_yaml = tmp_path / "empty.yaml"
        config_yaml.write_text("")

        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Default config test",
        )

        with patch("afterburn.diagnoser.Diagnoser") as mock_cls:
            mock_diag = MagicMock()
            mock_diag.run.return_value = mock_report
            mock_cls.return_value = mock_diag

            output = tmp_path / "report.html"
            with patch.object(DiagnosticReport, "save", return_value=output):
                runner = CliRunner()
                result = runner.invoke(
                    cli,
                    [
                        "diagnose",
                        "--base",
                        "model/base",
                        "--trained",
                        "model/trained",
                        "--config",
                        str(config_yaml),
                        "--output",
                        str(output),
                    ],
                )

            assert result.exit_code == 0


# ─── Path validation edge cases ─────────────────────────────────────────


class TestPathValidation:
    """Verify model ID validation through CLI."""

    def test_diagnose_rejects_empty_model_ids(self):
        """Empty model IDs should be rejected."""
        from afterburn.diagnoser import _validate_model_id
        from afterburn.exceptions import PathValidationError

        with pytest.raises(PathValidationError, match="cannot be empty"):
            _validate_model_id("")

        with pytest.raises(PathValidationError, match="cannot be empty"):
            _validate_model_id("   ")

    def test_diagnose_rejects_traversal_in_model_id(self):
        """Model IDs with directory traversal are rejected."""
        from afterburn.diagnoser import _validate_model_id
        from afterburn.exceptions import PathValidationError

        with pytest.raises(PathValidationError, match="traversal"):
            _validate_model_id("model/../../../etc/passwd")

    def test_diagnose_accepts_valid_hf_ids(self):
        """Valid HuggingFace model IDs are accepted."""
        from afterburn.diagnoser import _validate_model_id

        # These should not raise
        _validate_model_id("meta-llama/Llama-3.1-8B")
        _validate_model_id("gpt2")
        _validate_model_id("org/model-name_v2.0")

    def test_diagnose_rejects_special_chars_in_model_id(self):
        """Model IDs with special characters are rejected."""
        from afterburn.diagnoser import _validate_model_id
        from afterburn.exceptions import PathValidationError

        with pytest.raises(PathValidationError, match="Invalid model ID"):
            _validate_model_id("model; rm -rf /")

        with pytest.raises(PathValidationError, match="Invalid model ID"):
            _validate_model_id("model$(whoami)")


# ─── Diagnose default output ────────────────────────────────────────────


class TestDiagnoseDefaults:
    """Verify default behavior of the diagnose command."""

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_defaults_to_report_html(self, mock_cls, tmp_path):
        """Diagnose without --output defaults to report.html."""
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test summary",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        with patch.object(
            DiagnosticReport, "save", return_value=Path("report.html")
        ) as mock_save:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                ],
            )

        assert result.exit_code == 0
        # save should be called with default path
        mock_save.assert_called_once()
        assert mock_save.call_args[0][0] == Path("report.html")

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_default_method_is_unknown(self, mock_cls, tmp_path):
        """Diagnose without --method defaults to unknown."""
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        with patch.object(
            DiagnosticReport, "save", return_value=Path("report.html")
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                ],
            )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["method"] == "unknown"


# ─── Full pipeline flow (diagnose → all modules → output) ──────────────


class TestFullPipelineFlow:
    """Test that the full diagnose pipeline produces correct output."""

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_with_reward_hack_shows_risk_score(self, mock_cls, tmp_path):
        """Diagnose with reward_hack result shows risk score in output."""
        from afterburn.types import (
            FormatGamingResult,
            LengthBiasResult,
            StrategyCollapseResult,
            SycophancyResult,
        )

        reward_hack = RewardHackResult(
            length_bias=LengthBiasResult(
                score=60.0,
                cohens_d=1.2,
                p_value=0.001,
                mean_length_ratio=1.8,
                is_flagged=True,
            ),
            format_gaming=FormatGamingResult(score=40.0),
            strategy_collapse=StrategyCollapseResult(
                score=30.0,
                base_entropy=1.5,
                trained_entropy=1.0,
                entropy_drop=0.5,
            ),
            sycophancy=SycophancyResult(score=20.0),
            composite_score=45.0,
            risk_level=RiskLevel.MODERATE,
            flags=["Length bias detected"],
        )

        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base",
                trained_model="model/trained",
                method=TrainingMethod.RLHF,
            ),
            summary="Test summary for reward hack detection.",
            reward_hack=reward_hack,
            hack_score=45.0,
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.html"
        with patch.object(DiagnosticReport, "save", return_value=output):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--method",
                    "rlhf",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert "MODERATE" in result.output

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_with_weight_diff_shows_top_layer(self, mock_cls, tmp_path):
        """Diagnose with weight_diff result shows the most changed layer."""
        from afterburn.types import EmbeddingDrift, LayerDiff, WeightDiffResult

        top_layer = LayerDiff(
            layer_name="model.layers.2.self_attn.v_proj",
            layer_index=2,
            l2_norm=0.8,
            cosine_similarity=0.995,
            frobenius_norm=0.7,
            relative_change=0.015,
            param_count=2000,
        )

        wd_result = WeightDiffResult(
            layer_diffs=[top_layer],
            attention_heads=[],
            layernorm_shifts=[],
            embedding_drift=EmbeddingDrift(
                input_embedding_l2=0.1,
                input_embedding_cosine=0.999,
                output_embedding_l2=None,
                output_embedding_cosine=None,
                top_drifted_tokens=[],
            ),
            lora_analysis=None,
            total_param_count=50000,
            changed_param_count=2000,
        )

        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test weight diff summary",
            weight_diff=wd_result,
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.html"
        with patch.object(DiagnosticReport, "save", return_value=output):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert "model.layers.2.self_attn.v_proj" in result.output

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_summary_truncated_at_200_chars(self, mock_cls, tmp_path):
        """Long summary is truncated to 200 chars in CLI output."""
        long_summary = "A" * 300

        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary=long_summary,
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.html"
        with patch.object(DiagnosticReport, "save", return_value=output):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        # Should contain "..." truncation
        assert "..." in result.output


# ─── Verbose logging during commands ────────────────────────────────────


class TestVerboseLogging:
    """Verify that --verbose flag activates debug-level logging."""

    @patch("afterburn.diagnoser.Diagnoser")
    def test_verbose_with_diagnose(self, mock_cls, tmp_path):
        """--verbose flag is accepted and doesn't cause errors."""
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.html"
        with patch.object(DiagnosticReport, "save", return_value=output):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--verbose",
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0

    @patch("afterburn.behaviour.analyser.BehaviourAnalyser")
    @patch("afterburn.device.auto_detect_device")
    def test_verbose_with_behaviour(self, mock_auto_detect, mock_analyser_class):
        """--verbose with behaviour subcommand works."""
        mock_auto_detect.return_value = MagicMock()

        mock_result = BehaviourResult(
            base_results=[],
            trained_results=[],
            length_analysis=LengthAnalysis(
                base_mean=50.0,
                base_median=48.0,
                base_std=10.0,
                trained_mean=55.0,
                trained_median=52.0,
                trained_std=11.0,
                mean_diff=5.0,
                p_value=0.1,
                cohens_d=0.3,
                is_significant=False,
            ),
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
                "--verbose",
                "behaviour",
                "--base",
                "model/base",
                "--trained",
                "model/trained",
            ],
        )

        assert result.exit_code == 0


# ─── JSON log format with subcommands ───────────────────────────────────


class TestJSONLogFormatWithSubcommands:
    """Verify --log-format json works with subcommands."""

    @patch("afterburn.diagnoser.Diagnoser")
    def test_json_log_format_with_diagnose(self, mock_cls, tmp_path):
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.html"
        with patch.object(DiagnosticReport, "save", return_value=output):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--log-format",
                    "json",
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0


# ─── Output format variations ───────────────────────────────────────────


class TestOutputFormats:
    """Test different output format flags."""

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_json_format(self, mock_cls, tmp_path):
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.json"
        with patch.object(DiagnosticReport, "save", return_value=output) as mock_save:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--output",
                    str(output),
                    "--format",
                    "json",
                ],
            )

        assert result.exit_code == 0
        mock_save.assert_called_once()
        from afterburn.types import ReportFormat

        assert mock_save.call_args[1]["fmt"] == ReportFormat.JSON

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_markdown_format(self, mock_cls, tmp_path):
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.md"
        with patch.object(DiagnosticReport, "save", return_value=output) as mock_save:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--output",
                    str(output),
                    "--format",
                    "markdown",
                ],
            )

        assert result.exit_code == 0
        from afterburn.types import ReportFormat

        assert mock_save.call_args[1]["fmt"] == ReportFormat.MARKDOWN

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_pdf_format(self, mock_cls, tmp_path):
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Test",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.pdf"
        with patch.object(DiagnosticReport, "save", return_value=output) as mock_save:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--output",
                    str(output),
                    "--format",
                    "pdf",
                ],
            )

        assert result.exit_code == 0
        from afterburn.types import ReportFormat

        assert mock_save.call_args[1]["fmt"] == ReportFormat.PDF


# ─── Hack-check module selection ────────────────────────────────────────


class TestHackCheckModuleSelection:
    """Verify hack-check correctly selects behaviour + reward_hack modules."""

    @patch("afterburn.diagnoser.Diagnoser")
    def test_hack_check_sets_correct_modules(self, mock_cls):
        """hack-check should only enable behaviour and reward_hack."""
        from afterburn.types import (
            FormatGamingResult,
            LengthBiasResult,
            StrategyCollapseResult,
            SycophancyResult,
        )

        reward_hack = RewardHackResult(
            length_bias=LengthBiasResult(
                score=10.0,
                cohens_d=0.2,
                p_value=0.5,
                mean_length_ratio=1.0,
                is_flagged=False,
            ),
            format_gaming=FormatGamingResult(score=10.0),
            strategy_collapse=StrategyCollapseResult(
                score=10.0,
                base_entropy=1.0,
                trained_entropy=0.9,
                entropy_drop=0.1,
            ),
            sycophancy=SycophancyResult(score=10.0),
            composite_score=10.0,
            risk_level=RiskLevel.LOW,
        )

        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            reward_hack=reward_hack,
            hack_score=10.0,
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "hack-check",
                "--base",
                "model/base",
                "--trained",
                "model/trained",
            ],
        )

        assert result.exit_code == 0

        # Verify modules parameter
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["modules"] == ["behaviour", "reward_hack"]
        # Should NOT include weight_diff
        assert "weight_diff" not in call_kwargs["modules"]

    @patch("afterburn.diagnoser.Diagnoser")
    def test_hack_check_all_training_methods(self, mock_cls):
        """All training method choices are accepted by hack-check."""
        from afterburn.types import (
            FormatGamingResult,
            LengthBiasResult,
            StrategyCollapseResult,
            SycophancyResult,
        )

        reward_hack = RewardHackResult(
            length_bias=LengthBiasResult(
                score=10.0,
                cohens_d=0.2,
                p_value=0.5,
                mean_length_ratio=1.0,
                is_flagged=False,
            ),
            format_gaming=FormatGamingResult(score=10.0),
            strategy_collapse=StrategyCollapseResult(
                score=10.0,
                base_entropy=1.0,
                trained_entropy=0.9,
                entropy_drop=0.1,
            ),
            sycophancy=SycophancyResult(score=10.0),
            composite_score=10.0,
            risk_level=RiskLevel.LOW,
        )

        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            reward_hack=reward_hack,
            hack_score=10.0,
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        runner = CliRunner()
        for method in ["sft", "dpo", "rlhf", "rlvr", "grpo", "unknown"]:
            result = runner.invoke(
                cli,
                [
                    "hack-check",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--method",
                    method,
                ],
            )

            assert result.exit_code == 0, f"Failed for method={method}: {result.output}"


# ─── Diagnose with selective modules ─────────────────────────────────────


class TestDiagnoseModuleSelection:
    """Verify --modules flag correctly selects analysis modules."""

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_weight_diff_only(self, mock_cls, tmp_path):
        """Diagnose with --modules weight_diff only runs weight diff."""
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Weight diff only test",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.html"
        with patch.object(DiagnosticReport, "save", return_value=output):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--modules",
                    "weight_diff",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["modules"] == ["weight_diff"]

    @patch("afterburn.diagnoser.Diagnoser")
    def test_diagnose_multiple_modules(self, mock_cls, tmp_path):
        """Diagnose with multiple --modules flags."""
        mock_report = DiagnosticReport(
            model_pair=ModelPair(
                base_model="model/base", trained_model="model/trained"
            ),
            summary="Multi-module test",
        )

        mock_diag = MagicMock()
        mock_diag.run.return_value = mock_report
        mock_cls.return_value = mock_diag

        output = tmp_path / "report.html"
        with patch.object(DiagnosticReport, "save", return_value=output):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "diagnose",
                    "--base",
                    "model/base",
                    "--trained",
                    "model/trained",
                    "--modules",
                    "weight_diff",
                    "--modules",
                    "behaviour",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["modules"] == ["weight_diff", "behaviour"]


# ─── Weight-diff edge cases ─────────────────────────────────────────────


class TestWeightDiffEdgeCases:
    """Edge cases for the weight-diff CLI command."""

    @patch("afterburn.weight_diff.engine.WeightDiffEngine")
    @patch("afterburn.device.auto_detect_device")
    def test_weight_diff_empty_result(self, mock_auto_detect, mock_engine_class):
        """weight-diff with no changed layers still succeeds."""
        from afterburn.types import EmbeddingDrift, WeightDiffResult

        mock_auto_detect.return_value = MagicMock()

        mock_result = WeightDiffResult(
            layer_diffs=[],
            attention_heads=[],
            layernorm_shifts=[],
            embedding_drift=EmbeddingDrift(
                input_embedding_l2=0.0,
                input_embedding_cosine=1.0,
                output_embedding_l2=None,
                output_embedding_cosine=None,
                top_drifted_tokens=[],
            ),
            lora_analysis=None,
            total_param_count=10000,
            changed_param_count=0,
        )

        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "weight-diff",
                "--base",
                "model/base",
                "--trained",
                "model/trained",
            ],
        )

        assert result.exit_code == 0
        assert "10,000" in result.output or "10000" in result.output

    @patch("afterburn.weight_diff.engine.WeightDiffEngine")
    @patch("afterburn.device.auto_detect_device")
    def test_weight_diff_top_n_larger_than_results(
        self, mock_auto_detect, mock_engine_class
    ):
        """--top-n larger than actual layers still works."""
        from afterburn.types import EmbeddingDrift, LayerDiff, WeightDiffResult

        mock_auto_detect.return_value = MagicMock()

        layer_diffs = [
            LayerDiff(
                layer_name="model.layers.0.q_proj",
                layer_index=0,
                l2_norm=0.5,
                cosine_similarity=0.999,
                frobenius_norm=0.4,
                relative_change=0.001,
                param_count=1000,
            )
        ]

        mock_result = WeightDiffResult(
            layer_diffs=layer_diffs,
            attention_heads=[],
            layernorm_shifts=[],
            embedding_drift=EmbeddingDrift(
                input_embedding_l2=0.1,
                input_embedding_cosine=0.999,
                output_embedding_l2=None,
                output_embedding_cosine=None,
                top_drifted_tokens=[],
            ),
            lora_analysis=None,
            total_param_count=1000,
            changed_param_count=100,
        )

        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "weight-diff",
                "--base",
                "model/base",
                "--trained",
                "model/trained",
                "--top-n",
                "100",
            ],
        )

        assert result.exit_code == 0
        assert "model.layers.0.q_proj" in result.output


# ─── Behaviour result display ────────────────────────────────────────────


class TestBehaviourResultDisplay:
    """Verify behaviour output formatting edge cases."""

    @patch("afterburn.behaviour.analyser.BehaviourAnalyser")
    @patch("afterburn.device.auto_detect_device")
    def test_behaviour_without_strategy_shift(
        self, mock_auto_detect, mock_analyser_class
    ):
        """Behaviour with no dominant strategy shift still displays."""
        mock_auto_detect.return_value = MagicMock()

        mock_result = BehaviourResult(
            base_results=[],
            trained_results=[],
            length_analysis=LengthAnalysis(
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
            ),
            format_analysis=FormatAnalysis(),
            strategy_analysis=StrategyShiftAnalysis(
                dominant_shift=None,
                base_entropy=1.0,
                trained_entropy=1.0,
                entropy_change=0.0,
            ),
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
                "--base",
                "model/base",
                "--trained",
                "model/trained",
            ],
        )

        assert result.exit_code == 0
        # Should show "No" for significance
        assert "No" in result.output

    @patch("afterburn.behaviour.analyser.BehaviourAnalyser")
    @patch("afterburn.device.auto_detect_device")
    def test_behaviour_shows_significant_result(
        self, mock_auto_detect, mock_analyser_class
    ):
        """Behaviour with significant changes shows 'Yes'."""
        mock_auto_detect.return_value = MagicMock()

        mock_result = BehaviourResult(
            base_results=[],
            trained_results=[],
            length_analysis=LengthAnalysis(
                base_mean=50.0,
                base_median=48.0,
                base_std=10.0,
                trained_mean=150.0,
                trained_median=148.0,
                trained_std=15.0,
                mean_diff=100.0,
                p_value=0.0001,
                cohens_d=3.0,
                is_significant=True,
            ),
            format_analysis=FormatAnalysis(),
            strategy_analysis=StrategyShiftAnalysis(
                dominant_shift="step_by_step",
                base_entropy=1.5,
                trained_entropy=0.5,
                entropy_change=-1.0,
            ),
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
                "--base",
                "model/base",
                "--trained",
                "model/trained",
            ],
        )

        assert result.exit_code == 0
        assert "Yes" in result.output
        assert "step_by_step" in result.output
        assert "150.0" in result.output
