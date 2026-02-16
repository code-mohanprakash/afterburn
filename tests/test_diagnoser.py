"""Tests for the top-level Diagnoser API."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from afterburn.config import AfterburnConfig, BehaviourConfig, RewardHackConfig, WeightDiffConfig
from afterburn.diagnoser import Diagnoser
from afterburn.types import TrainingMethod


class TestDiagnoser:
    """Tests for Diagnoser class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with all modules enabled."""
        config = AfterburnConfig()
        config.device = "cpu"
        config.weight_diff = WeightDiffConfig(enabled=True)
        config.behaviour = BehaviourConfig(enabled=True, suites=["math", "code"])
        config.reward_hack = RewardHackConfig(enabled=True)
        return config

    def test_constructor_accepts_valid_method_strings(self):
        """Constructor accepts valid method strings."""
        with patch("afterburn.diagnoser.load_config"):
            with patch("afterburn.diagnoser.auto_detect_device"):
                # Test all valid methods
                for method_str in ["sft", "dpo", "rlhf", "rlvr", "grpo", "lora", "qlora"]:
                    diag = Diagnoser(
                        base_model="base/model",
                        trained_model="trained/model",
                        method=method_str,
                    )
                    assert isinstance(diag.model_pair.method, TrainingMethod)
                    assert diag.model_pair.method.value == method_str

    def test_constructor_handles_invalid_method_defaults_to_unknown(self):
        """Constructor handles invalid method and defaults to UNKNOWN."""
        with patch("afterburn.diagnoser.load_config"):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                    method="invalid_method",
                )
                assert diag.model_pair.method == TrainingMethod.UNKNOWN

    def test_constructor_accepts_training_method_enum(self):
        """Constructor accepts TrainingMethod enum directly."""
        with patch("afterburn.diagnoser.load_config"):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                    method=TrainingMethod.DPO,
                )
                assert diag.model_pair.method == TrainingMethod.DPO

    def test_default_modules_returns_all_three_when_enabled(self, mock_config):
        """_default_modules() returns all 3 when config enables all."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                modules = diag._default_modules()

                assert "weight_diff" in modules
                assert "behaviour" in modules
                assert "reward_hack" in modules
                assert len(modules) == 3

    def test_default_modules_excludes_disabled_weight_diff(self, mock_config):
        """_default_modules() excludes disabled modules."""
        mock_config.weight_diff.enabled = False

        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                modules = diag._default_modules()

                assert "weight_diff" not in modules
                assert "behaviour" in modules
                assert "reward_hack" in modules
                assert len(modules) == 2

    def test_default_modules_excludes_disabled_behaviour(self, mock_config):
        """_default_modules() excludes disabled behaviour module."""
        mock_config.behaviour.enabled = False

        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                modules = diag._default_modules()

                assert "weight_diff" in modules
                assert "behaviour" not in modules
                assert "reward_hack" in modules
                assert len(modules) == 2

    def test_default_modules_excludes_disabled_reward_hack(self, mock_config):
        """_default_modules() excludes disabled reward_hack module."""
        mock_config.reward_hack.enabled = False

        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                modules = diag._default_modules()

                assert "weight_diff" in modules
                assert "behaviour" in modules
                assert "reward_hack" not in modules
                assert len(modules) == 2

    def test_respects_config_path_parameter(self, tmp_path: Path):
        """Respects config_path parameter."""
        config_file = tmp_path / "custom_config.yaml"
        config_file.write_text("device: cpu\n")

        with patch("afterburn.diagnoser.load_config") as mock_load:
            with patch("afterburn.diagnoser.auto_detect_device"):
                mock_load.return_value = AfterburnConfig()

                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                    config_path=config_file,
                )

                # Verify load_config was called with the custom path
                mock_load.assert_called_once_with(config_file)

    def test_respects_device_override(self, mock_config):
        """Respects device override parameter."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device") as mock_detect:
                mock_device_config = MagicMock()
                mock_detect.return_value = mock_device_config

                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                    device="cuda",
                )

                # Verify auto_detect_device was called with force_device
                mock_detect.assert_called_once_with(force_device="cuda")
                assert diag.device_config == mock_device_config

    def test_respects_config_device_when_not_auto(self, mock_config):
        """Uses config device when not 'auto' and no override provided."""
        mock_config.device = "mps"

        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device") as mock_detect:
                mock_device_config = MagicMock()
                mock_detect.return_value = mock_device_config

                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                mock_detect.assert_called_once_with(force_device="mps")

    def test_uses_auto_detect_when_config_device_is_auto(self, mock_config):
        """Uses auto_detect_device with no override when config.device is 'auto'."""
        mock_config.device = "auto"

        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device") as mock_detect:
                mock_device_config = MagicMock()
                mock_detect.return_value = mock_device_config

                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                # Should be called without force_device
                mock_detect.assert_called_once_with()

    def test_respects_suites_override(self, mock_config):
        """Respects suites override parameter."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                    suites=["custom_suite_1", "custom_suite_2"],
                )

                assert diag.suites == ["custom_suite_1", "custom_suite_2"]

    def test_uses_config_suites_when_no_override(self, mock_config):
        """Uses config suites when no override provided."""
        mock_config.behaviour.suites = ["math", "code", "reasoning"]

        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                assert diag.suites == ["math", "code", "reasoning"]

    def test_respects_modules_override(self, mock_config):
        """Respects modules override parameter."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                    modules=["weight_diff"],  # Only run weight_diff
                )

                assert diag.modules == ["weight_diff"]

    def test_uses_default_modules_when_no_override(self, mock_config):
        """Uses _default_modules() when no modules override provided."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                # Should use default modules
                assert "weight_diff" in diag.modules
                assert "behaviour" in diag.modules
                assert "reward_hack" in diag.modules

    def test_collect_logits_parameter(self, mock_config):
        """collect_logits parameter is stored correctly."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                    collect_logits=True,
                )

                assert diag.collect_logits is True

    def test_collect_logits_defaults_to_false(self, mock_config):
        """collect_logits defaults to False."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="base/model",
                    trained_model="trained/model",
                )

                assert diag.collect_logits is False

    def test_stores_model_pair_correctly(self, mock_config):
        """Stores model pair with correct base and trained models."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                diag = Diagnoser(
                    base_model="meta-llama/Llama-2-7b",
                    trained_model="my-org/Llama-2-7b-finetuned",
                    method="sft",
                )

                assert diag.model_pair.base_model == "meta-llama/Llama-2-7b"
                assert diag.model_pair.trained_model == "my-org/Llama-2-7b-finetuned"
                assert diag.model_pair.method == TrainingMethod.SFT

    def test_run_weight_diff_calls_engine(self, mock_config):
        """run_weight_diff() creates and runs WeightDiffEngine."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                with patch("afterburn.diagnoser.WeightDiffEngine") as mock_engine_class:
                    mock_engine = MagicMock()
                    mock_result = MagicMock()
                    mock_engine.run.return_value = mock_result
                    mock_engine_class.return_value = mock_engine

                    diag = Diagnoser(
                        base_model="base/model",
                        trained_model="trained/model",
                    )

                    result = diag.run_weight_diff()

                    mock_engine_class.assert_called_once()
                    mock_engine.run.assert_called_once()
                    assert result == mock_result

    def test_run_behaviour_calls_analyser(self, mock_config):
        """run_behaviour() creates and runs BehaviourAnalyser."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                with patch("afterburn.diagnoser.BehaviourAnalyser") as mock_analyser_class:
                    mock_analyser = MagicMock()
                    mock_result = MagicMock()
                    mock_analyser.run.return_value = mock_result
                    mock_analyser_class.return_value = mock_analyser

                    diag = Diagnoser(
                        base_model="base/model",
                        trained_model="trained/model",
                    )

                    result = diag.run_behaviour()

                    mock_analyser_class.assert_called_once()
                    mock_analyser.run.assert_called_once()
                    assert result == mock_result

    def test_run_hack_check_with_behaviour_result(self, mock_config):
        """run_hack_check() uses provided behaviour_result."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                with patch("afterburn.diagnoser.RewardHackDetector") as mock_detector_class:
                    with patch("afterburn.diagnoser.generate_summary"):
                        with patch("afterburn.diagnoser.generate_recommendations"):
                            mock_detector = MagicMock()
                            mock_reward_hack = MagicMock()
                            mock_reward_hack.composite_score = 45.0
                            mock_detector.run.return_value = mock_reward_hack
                            mock_detector_class.return_value = mock_detector

                            diag = Diagnoser(
                                base_model="base/model",
                                trained_model="trained/model",
                            )

                            mock_behaviour = MagicMock()
                            report = diag.run_hack_check(behaviour_result=mock_behaviour)

                            # Should not run behaviour analysis
                            mock_detector_class.assert_called_once()
                            assert report.behaviour == mock_behaviour
                            assert report.reward_hack == mock_reward_hack
                            assert report.hack_score == 45.0

    def test_run_hack_check_without_behaviour_result_runs_behaviour(self, mock_config):
        """run_hack_check() runs behaviour analysis when not provided."""
        with patch("afterburn.diagnoser.load_config", return_value=mock_config):
            with patch("afterburn.diagnoser.auto_detect_device"):
                with patch("afterburn.diagnoser.BehaviourAnalyser") as mock_analyser_class:
                    with patch("afterburn.diagnoser.RewardHackDetector") as mock_detector_class:
                        with patch("afterburn.diagnoser.generate_summary"):
                            with patch("afterburn.diagnoser.generate_recommendations"):
                                mock_analyser = MagicMock()
                                mock_behaviour = MagicMock()
                                mock_analyser.run.return_value = mock_behaviour
                                mock_analyser_class.return_value = mock_analyser

                                mock_detector = MagicMock()
                                mock_reward_hack = MagicMock()
                                mock_reward_hack.composite_score = 30.0
                                mock_detector.run.return_value = mock_reward_hack
                                mock_detector_class.return_value = mock_detector

                                diag = Diagnoser(
                                    base_model="base/model",
                                    trained_model="trained/model",
                                )

                                report = diag.run_hack_check()

                                # Should run behaviour analysis
                                mock_analyser_class.assert_called_once()
                                mock_analyser.run.assert_called_once()
                                assert report.behaviour == mock_behaviour
