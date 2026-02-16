"""HuggingFace model loading with compatibility validation."""

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from afterburn.device import DeviceConfig, estimate_model_memory_gb, register_cleanup
from afterburn.exceptions import IncompatibleModelsError, ModelLoadError, ModelNotFoundError
from afterburn.loading.checkpoint import CheckpointInfo, detect_checkpoint

logger = logging.getLogger(__name__)

COMPATIBILITY_ATTRS = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
]


class ModelLoader:
    """Loads HuggingFace models with memory management."""

    def __init__(self, device_config: DeviceConfig):
        self.device_config = device_config

    def load_config(self, model_id: str) -> AutoConfig:
        """Load only the model config (no weights)."""
        try:
            config: AutoConfig = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
            return config
        except OSError as e:
            if "does not appear to have" in str(e) or "not found" in str(e).lower():
                raise ModelNotFoundError(
                    f"Model '{model_id}' not found. Check the model ID or path."
                ) from e
            raise ModelLoadError(f"Failed to load config for '{model_id}': {e}") from e

    def load_tokenizer(self, model_id: str) -> PreTrainedTokenizer:
        """Load tokenizer for a model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)  # type: ignore[no-untyped-call]
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tok: PreTrainedTokenizer = tokenizer
            return tok
        except Exception as e:
            raise ModelLoadError(f"Failed to load tokenizer for '{model_id}': {e}") from e

    def load_model(self, model_id: str) -> PreTrainedModel:
        """Load a full model for inference."""
        # Memory estimation before loading
        try:
            config = self.load_config(model_id)
            num_params = self._estimate_num_parameters(config)
            if num_params > 0:
                estimated_gb = estimate_model_memory_gb(num_params, self.device_config.dtype)
                if estimated_gb > self.device_config.max_memory_gb:
                    logger.warning(
                        "Model '%s' may not fit in memory: estimated %.2f GB, available %.2f GB. "
                        "Loading will proceed but may fail with OOM.",
                        model_id,
                        estimated_gb,
                        self.device_config.max_memory_gb,
                    )
                else:
                    logger.debug(
                        "Memory estimate for '%s': %.2f GB (%.2f GB available)",
                        model_id,
                        estimated_gb,
                        self.device_config.max_memory_gb,
                    )
        except Exception as e:
            logger.debug("Could not estimate memory for '%s': %s", model_id, e)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.device_config.dtype,
                device_map="auto" if self.device_config.is_cuda else None,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
            if not self.device_config.is_cuda:
                model = model.to(self.device_config.device)  # type: ignore[arg-type]
            model.eval()

            # Register cleanup for this model
            def cleanup_model() -> None:
                self._cleanup_model(model)
            register_cleanup(cleanup_model)

            return model
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM loading '%s'. Falling back to CPU.", model_id)
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
            model.eval()

            # Register cleanup for this model
            def cleanup_model() -> None:
                self._cleanup_model(model)
            register_cleanup(cleanup_model)

            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load model '{model_id}': {e}") from e

    def validate_compatibility(self, base_id: str, trained_id: str) -> None:
        """Verify two models have compatible architectures."""
        base_config = self.load_config(base_id)
        trained_config = self.load_config(trained_id)

        mismatches = []
        for attr in COMPATIBILITY_ATTRS:
            base_val = getattr(base_config, attr, None)
            trained_val = getattr(trained_config, attr, None)
            if base_val is not None and trained_val is not None and base_val != trained_val:
                mismatches.append(f"{attr}: base={base_val}, trained={trained_val}")

        if mismatches:
            raise IncompatibleModelsError(
                "Models have incompatible architectures:\n"
                + "\n".join(f"  - {m}" for m in mismatches)
            )

    def get_checkpoint_info(self, model_id: str) -> CheckpointInfo:
        """Get checkpoint info for a model."""
        return detect_checkpoint(model_id)

    def _estimate_num_parameters(self, config: AutoConfig) -> int:
        """Estimate the number of parameters from model config."""
        # Try to get num_parameters if available
        if hasattr(config, "num_parameters"):
            return int(config.num_parameters)

        # Estimate from architecture
        hidden_size = getattr(config, "hidden_size", 0)
        num_layers = getattr(config, "num_hidden_layers", 0)
        intermediate_size = getattr(config, "intermediate_size", 0)
        vocab_size = getattr(config, "vocab_size", 0)

        if all([hidden_size, num_layers, vocab_size]):
            # Rough estimate: embeddings + layers
            if intermediate_size == 0:
                intermediate_size = hidden_size * 4  # common default

            # Embedding parameters
            embed_params = vocab_size * hidden_size

            # Per-layer parameters (simplified):
            # - Self-attention: 4 * hidden_size^2 (Q, K, V, O projections)
            # - FFN: 2 * hidden_size * intermediate_size
            # - LayerNorms: ~4 * hidden_size (small, ignore for rough estimate)
            per_layer = (4 * hidden_size * hidden_size) + (2 * hidden_size * intermediate_size)
            layer_params = num_layers * per_layer

            # Output head
            head_params = vocab_size * hidden_size

            total = embed_params + layer_params + head_params
            logger.debug("Estimated parameters: %d (%.2fB)", total, total / 1e9)
            return total

        logger.debug("Could not estimate parameters from config")
        return 0

    @staticmethod
    def _cleanup_model(model: PreTrainedModel) -> None:
        """Cleanup function for registered models."""
        import contextlib

        with contextlib.suppress(Exception):
            del model

    @staticmethod
    def unload_model(model: PreTrainedModel) -> None:
        """Unload a model and free memory."""
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()
