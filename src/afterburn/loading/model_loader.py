"""HuggingFace model loading with compatibility validation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from afterburn.device import DeviceConfig
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
            return AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        except OSError as e:
            if "does not appear to have" in str(e) or "not found" in str(e).lower():
                raise ModelNotFoundError(
                    f"Model '{model_id}' not found. Check the model ID or path."
                ) from e
            raise ModelLoadError(f"Failed to load config for '{model_id}': {e}") from e

    def load_tokenizer(self, model_id: str) -> PreTrainedTokenizer:
        """Load tokenizer for a model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            raise ModelLoadError(f"Failed to load tokenizer for '{model_id}': {e}") from e

    def load_model(self, model_id: str) -> PreTrainedModel:
        """Load a full model for inference."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.device_config.dtype,
                device_map="auto" if self.device_config.is_cuda else None,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
            if not self.device_config.is_cuda:
                model = model.to(self.device_config.device)
            model.eval()
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
                f"Models have incompatible architectures:\n"
                + "\n".join(f"  - {m}" for m in mismatches)
            )

    def get_checkpoint_info(self, model_id: str) -> CheckpointInfo:
        """Get checkpoint info for a model."""
        return detect_checkpoint(model_id)

    @staticmethod
    def unload_model(model: PreTrainedModel) -> None:
        """Unload a model and free memory."""
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()
