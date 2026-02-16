"""Batched inference runner for prompt suites."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable

import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

# Suppress noisy transformers warnings during generation
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

from afterburn.device import DeviceConfig
from afterburn.types import Prompt, PromptResult

logger = logging.getLogger(__name__)


class PromptRunner:
    """Runs prompt suites through models and collects results."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device_config: DeviceConfig,
        max_new_tokens: int = 512,
        batch_size: int = 4,
        temperature: float = 0.0,
        collect_logits: bool = False,
        top_k_probs: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device_config.device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.collect_logits = collect_logits
        self.top_k_probs = top_k_probs

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def run_suite(
        self,
        prompts: list[Prompt],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[PromptResult]:
        """Run all prompts through the model.

        Uses batched inference for efficiency. Collects generated text
        and timing information.
        """
        results: list[PromptResult] = []
        total = len(prompts)

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = prompts[batch_start:batch_end]

            batch_results = self._run_batch(batch)
            results.extend(batch_results)

            if progress_callback:
                progress_callback(batch_end, total)

        return results

    def _run_batch(self, batch: list[Prompt]) -> list[PromptResult]:
        """Run a single batch through the model."""
        texts = [p.text for p in batch]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_lengths = inputs["input_ids"].shape[1]

        # Generate
        start_time = time.perf_counter()

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature

            if self.collect_logits:
                gen_kwargs["output_scores"] = True
                gen_kwargs["return_dict_in_generate"] = True

            gen_output = self.model.generate(**inputs, **gen_kwargs)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        per_sample_ms = elapsed_ms / len(batch)

        # Extract sequences and optional scores
        if self.collect_logits and hasattr(gen_output, "sequences"):
            output_ids = gen_output.sequences
            scores = gen_output.scores  # tuple of (batch_size, vocab_size) per step
        else:
            output_ids = (
                gen_output
                if not hasattr(gen_output, "sequences")
                else gen_output.sequences
            )
            scores = None

        # Decode outputs
        results = []
        for i, prompt in enumerate(batch):
            # Extract only the generated tokens (exclude input)
            generated_ids = output_ids[i][input_lengths:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            output_tokens = len(generated_ids)

            # Extract top-k token probabilities if available
            top_token_probs = None
            if scores is not None:
                top_token_probs = self._extract_top_k_probs(scores, i, output_tokens)

            results.append(
                PromptResult(
                    prompt_id=prompt.id,
                    prompt_text=prompt.text,
                    category=prompt.category,
                    output_text=output_text.strip(),
                    output_tokens=output_tokens,
                    generation_time_ms=per_sample_ms,
                    expected_answer=prompt.expected_answer,
                    top_token_probs=top_token_probs,
                )
            )

        return results

    def _extract_top_k_probs(
        self,
        scores: tuple[object, ...],
        batch_idx: int,
        num_tokens: int,
    ) -> list[dict[str, float]]:
        """Extract top-k token probabilities from generation scores."""
        top_probs_list = []
        k = self.top_k_probs

        for step_idx in range(min(num_tokens, len(scores))):
            logits = scores[step_idx][batch_idx]  # type: ignore[index]
            probs = torch.softmax(logits, dim=-1)
            top_k_vals, top_k_ids = torch.topk(probs, k=min(k, probs.size(-1)))

            step_probs = {}
            for j in range(top_k_vals.size(0)):
                token_str = self.tokenizer.decode([top_k_ids[j].item()])
                step_probs[token_str] = float(top_k_vals[j].item())
            top_probs_list.append(step_probs)

        return top_probs_list

    def run_single(self, prompt: Prompt) -> PromptResult:
        """Run a single prompt (convenience method)."""
        results = self._run_batch([prompt])
        return results[0]
