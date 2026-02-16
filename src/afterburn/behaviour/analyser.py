"""Behaviour analyser — orchestrates behavioural shift detection."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable

import torch

from afterburn.behaviour.calibration import analyze_calibration
from afterburn.behaviour.cot_analysis import analyze_chain_of_thought
from afterburn.behaviour.diversity import analyze_diversity
from afterburn.behaviour.format_analysis import analyze_format_compliance
from afterburn.behaviour.length_analysis import analyze_length_distribution
from afterburn.behaviour.reasoning import analyze_strategy_shift
from afterburn.behaviour.token_divergence import analyze_token_divergence
from afterburn.config import BehaviourConfig
from afterburn.device import DeviceConfig
from afterburn.loading.model_loader import ModelLoader
from afterburn.prompts.runner import PromptRunner
from afterburn.prompts.suite import PromptSuite
from afterburn.types import BehaviourResult, ModelPair

logger = logging.getLogger(__name__)


class BehaviourAnalyser:
    """Orchestrates behavioural analysis between base and trained models.

    Loads models one at a time to minimize memory usage:
    1. Load base model → run prompts → unload
    2. Load trained model → run prompts → unload
    3. Compare results (no model in memory)
    """

    def __init__(
        self,
        model_pair: ModelPair,
        device_config: DeviceConfig,
        suites: list[str] | None = None,
        max_new_tokens: int = 512,
        batch_size: int = 4,
        temperature: float = 0.0,
        collect_logits: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
        config: BehaviourConfig | None = None,
    ):
        self.model_pair = model_pair
        self.device_config = device_config
        self.suite_names = suites or ["math", "code", "reasoning", "safety"]
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.collect_logits = collect_logits
        self._progress = progress_callback
        self.config = config or BehaviourConfig()

    def run(self) -> BehaviourResult:
        """Execute full behaviour analysis."""
        # Load prompt suites
        all_prompts = self._load_prompts()
        logger.info("Loaded %d prompts across %d suites", len(all_prompts), len(self.suite_names))

        loader = ModelLoader(self.device_config)

        # Phase 1: Run base model
        logger.info("Running base model inference...")
        if self._progress:
            self._progress("Loading base model", 0, 4)

        base_model = loader.load_model(self.model_pair.base_model)
        tokenizer = loader.load_tokenizer(self.model_pair.base_model)
        runner = PromptRunner(
            base_model,
            tokenizer,
            self.device_config,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            temperature=self.temperature,
            collect_logits=self.collect_logits,
            top_k_probs=5,
        )
        base_results = runner.run_suite(all_prompts)  # type: ignore[arg-type]

        # Unload base model
        loader.unload_model(base_model)
        del base_model, runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 2: Run trained model
        logger.info("Running trained model inference...")
        if self._progress:
            self._progress("Loading trained model", 1, 4)

        trained_model = loader.load_model(self.model_pair.trained_model)
        tokenizer = loader.load_tokenizer(
            self.model_pair.trained_model,
            fallback_model_id=self.model_pair.base_model,
        )
        runner = PromptRunner(
            trained_model,
            tokenizer,
            self.device_config,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            temperature=self.temperature,
            collect_logits=self.collect_logits,
            top_k_probs=5,
        )
        trained_results = runner.run_suite(all_prompts)  # type: ignore[arg-type]

        # Unload trained model
        loader.unload_model(trained_model)
        del trained_model, runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 3: Analyze differences
        logger.info("Analyzing behavioural differences...")
        if self._progress:
            self._progress("Analyzing behaviour", 2, 4)

        length_analysis = analyze_length_distribution(
            base_results, trained_results,
            significance_level=self.config.significance_level,
            effect_size_threshold=self.config.effect_size_threshold,
        )
        format_analysis = analyze_format_compliance(base_results, trained_results)
        strategy_analysis = analyze_strategy_shift(base_results, trained_results)
        cot_analysis = analyze_chain_of_thought(base_results, trained_results)
        calibration = analyze_calibration(base_results, trained_results)
        diversity = analyze_diversity(
            [r.output_text for r in base_results],
            [r.output_text for r in trained_results],
            base_categories=[r.category for r in base_results],
            trained_categories=[r.category for r in trained_results],
        )
        token_divergence = analyze_token_divergence(base_results, trained_results)

        # Apply Benjamini-Hochberg FDR correction across all p-values
        from afterburn.ci import benjamini_hochberg

        p_values = [length_analysis.p_value]
        if p_values[0] < 1.0:
            corrected = benjamini_hochberg(p_values, alpha=self.config.significance_level)
            length_analysis.corrected_p_value = corrected[0][0]
            # Re-evaluate significance with corrected p-value
            length_analysis.is_significant = (
                corrected[0][1] and abs(length_analysis.cohens_d) > self.config.effect_size_threshold
            )

        if self._progress:
            self._progress("Complete", 4, 4)

        return BehaviourResult(
            base_results=base_results,
            trained_results=trained_results,
            length_analysis=length_analysis,
            format_analysis=format_analysis,
            strategy_analysis=strategy_analysis,
            cot_analysis=cot_analysis,
            calibration=calibration,
            diversity=diversity,
            token_divergence=token_divergence,
        )

    def _load_prompts(self) -> list[object]:
        """Load all configured prompt suites."""
        from afterburn.types import Prompt

        all_prompts: list[Prompt] = []
        for name in self.suite_names:
            suite = PromptSuite.load(name)
            all_prompts.extend(suite.prompts)
        return all_prompts  # type: ignore[return-value]
