"""Top-level Diagnoser class — the public Python API."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from afterburn.behaviour.analyser import BehaviourAnalyser
from afterburn.config import load_config
from afterburn.device import auto_detect_device
from afterburn.exceptions import PathValidationError
from afterburn.report.summary import generate_recommendations, generate_summary
from afterburn.reward_hack.detector import RewardHackDetector
from afterburn.types import (
    BehaviourResult,
    DiagnosticReport,
    ModelPair,
    TrainingMethod,
    WeightDiffResult,
)
from afterburn.weight_diff.engine import WeightDiffEngine

logger = logging.getLogger(__name__)


def _validate_model_id(model_id: str) -> None:
    """Validate that a model ID is either a valid HuggingFace ID or an existing safe local path.

    Args:
        model_id: Either a HuggingFace model ID or local path.

    Raises:
        PathValidationError: If the model ID is invalid or unsafe.
    """
    # Check for empty or whitespace-only model IDs
    if not model_id or not model_id.strip():
        raise PathValidationError("Model ID cannot be empty")

    # Check if it's a local path
    path = Path(model_id)
    if path.exists():
        # Validate local path for safety
        try:
            resolved = path.resolve()

            # Check for directory traversal patterns
            path_str = str(model_id)
            suspicious_patterns = ["/../", "/..", "\\..\\", "\\.."]

            for pattern in suspicious_patterns:
                if pattern in path_str:
                    raise PathValidationError(
                        f"Model path contains directory traversal pattern: {model_id}"
                    )

            if not resolved.is_absolute():
                raise PathValidationError(f"Model path must be absolute: {model_id}")

        except (OSError, RuntimeError) as e:
            raise PathValidationError(f"Invalid model path: {model_id}. Error: {e}") from e
    else:
        # Validate HuggingFace model ID format
        # Valid format: alphanumeric, hyphens, underscores, forward slashes, dots
        # Examples: "meta-llama/Llama-3.1-8B", "gpt2", "org/model-name_v2.0"
        if not re.match(r"^[\w\-./]+$", model_id):
            raise PathValidationError(
                f"Invalid model ID: {model_id}. Must be alphanumeric with hyphens, "
                "underscores, slashes, and dots only."
            )

        # Additional safety checks
        if ".." in model_id:
            raise PathValidationError(
                f"Model ID contains directory traversal pattern: {model_id}"
            )


class Diagnoser:
    """Top-level API for running Afterburn diagnostics.

    Usage:
        diag = Diagnoser(
            base_model="meta-llama/Llama-3.1-8B",
            trained_model="my-org/Llama-3.1-8B-RLVR",
            method="rlvr",
        )
        report = diag.run()
        report.save("report.html")
    """

    def __init__(
        self,
        base_model: str,
        trained_model: str,
        method: str | TrainingMethod = "unknown",
        suites: list[str] | None = None,
        config_path: str | Path | None = None,
        device: str | None = None,
        modules: list[str] | None = None,
        collect_logits: bool = False,
    ):
        # Validate model IDs for safety
        _validate_model_id(base_model)
        _validate_model_id(trained_model)

        if isinstance(method, str):
            try:
                method = TrainingMethod(method)
            except ValueError:
                method = TrainingMethod.UNKNOWN

        self.model_pair = ModelPair(
            base_model=base_model,
            trained_model=trained_model,
            method=method,
        )

        # Load config
        self._config = load_config(config_path)

        # Setup device
        if device:
            self.device_config = auto_detect_device(force_device=device)
        elif self._config.device != "auto":
            self.device_config = auto_detect_device(force_device=self._config.device)
        else:
            self.device_config = auto_detect_device()

        self.suites = suites or self._config.behaviour.suites
        self.modules = modules or self._default_modules()
        self.collect_logits = collect_logits

    def _default_modules(self) -> list[str]:
        modules = []
        if self._config.weight_diff.enabled:
            modules.append("weight_diff")
        if self._config.behaviour.enabled:
            modules.append("behaviour")
        if self._config.reward_hack.enabled:
            modules.append("reward_hack")
        return modules

    def run(
        self,
        progress: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> DiagnosticReport:
        """Run all configured analysis modules.

        Execution order:
        1. Weight diff (no inference needed, just weight loading)
        2. Behaviour analysis (needs inference, loads models one at a time)
        3. Reward hack detection (reuses behaviour results, no model loading)
        4. Report assembly (aggregates all results)
        """
        report = DiagnosticReport(model_pair=self.model_pair)

        # Step 1: Weight diff
        if "weight_diff" in self.modules:
            logger.info("Running weight diff analysis...")
            engine = WeightDiffEngine(
                self.model_pair,
                self.device_config,
                progress_callback=progress_callback,
                config=self._config.weight_diff,
            )
            report.weight_diff = engine.run()
            report.top_changed_layers = report.weight_diff.top_changed_layers

        # Step 2: Behaviour analysis
        if "behaviour" in self.modules:
            logger.info("Running behaviour analysis...")
            analyser = BehaviourAnalyser(
                self.model_pair,
                self.device_config,
                suites=self.suites,
                max_new_tokens=self._config.behaviour.max_new_tokens,
                batch_size=self._config.behaviour.batch_size,
                temperature=self._config.behaviour.temperature,
                collect_logits=self.collect_logits,
                progress_callback=progress_callback,
                config=self._config.behaviour,
            )
            report.behaviour = analyser.run()

        # Step 3: Reward hack detection
        if "reward_hack" in self.modules and report.behaviour:
            logger.info("Running reward hack detection...")
            detector = RewardHackDetector(
                report.behaviour,
                method=self.model_pair.method,
                thresholds=self._config.reward_hack.thresholds,
                weights=self._config.reward_hack.weights,
            )
            report.reward_hack = detector.run()
            report.hack_score = report.reward_hack.composite_score

        # Step 4: Generate summary and recommendations
        report.summary = generate_summary(report, config=self._config.report)
        report.recommendations = generate_recommendations(report, config=self._config.report)

        logger.info("Diagnostic analysis complete.")
        return report

    def run_weight_diff(self) -> WeightDiffResult:
        """Run only weight diff analysis."""
        engine = WeightDiffEngine(
            self.model_pair, self.device_config, config=self._config.weight_diff
        )
        return engine.run()

    def run_behaviour(self) -> BehaviourResult:
        """Run only behaviour analysis."""
        analyser = BehaviourAnalyser(
            self.model_pair,
            self.device_config,
            suites=self.suites,
            max_new_tokens=self._config.behaviour.max_new_tokens,
            batch_size=self._config.behaviour.batch_size,
            temperature=self._config.behaviour.temperature,
            collect_logits=self.collect_logits,
            config=self._config.behaviour,
        )
        return analyser.run()

    def run_hack_check(self, behaviour_result: BehaviourResult | None = None) -> DiagnosticReport:
        """Run reward hack detection.

        If behaviour_result is not provided, runs behaviour analysis first.
        """
        if behaviour_result is None:
            behaviour_result = self.run_behaviour()

        detector = RewardHackDetector(
            behaviour_result,
            method=self.model_pair.method,
            thresholds=self._config.reward_hack.thresholds,
            weights=self._config.reward_hack.weights,
        )
        reward_hack = detector.run()

        report = DiagnosticReport(
            model_pair=self.model_pair,
            behaviour=behaviour_result,
            reward_hack=reward_hack,
            hack_score=reward_hack.composite_score,
        )
        report.summary = generate_summary(report, config=self._config.report)
        report.recommendations = generate_recommendations(report, config=self._config.report)
        return report
