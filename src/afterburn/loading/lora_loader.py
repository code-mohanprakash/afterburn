"""LoRA/QLoRA adapter loading and analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from afterburn.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Parsed LoRA adapter configuration."""

    r: int = 0
    lora_alpha: float = 1.0
    target_modules: list[str] = field(default_factory=list)
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = ""
    base_model_name_or_path: str = ""


@dataclass
class LoRAWeights:
    """Loaded LoRA adapter weights."""

    config: LoRAConfig
    weights: dict[str, torch.Tensor] = field(default_factory=dict)
    affected_layers: list[str] = field(default_factory=list)


def load_lora_adapter(adapter_path: str | Path) -> LoRAWeights:
    """Load a LoRA adapter from a directory.

    Supports PEFT-format adapters with adapter_config.json
    and adapter_model.safetensors/adapter_model.bin.
    """
    path = Path(adapter_path)

    # Load config
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise ModelLoadError(f"No adapter_config.json found in {path}")

    with open(config_path) as f:
        raw_config = json.load(f)

    config = LoRAConfig(
        r=raw_config.get("r", 0),
        lora_alpha=raw_config.get("lora_alpha", 1.0),
        target_modules=raw_config.get("target_modules", []),
        lora_dropout=raw_config.get("lora_dropout", 0.0),
        bias=raw_config.get("bias", "none"),
        task_type=raw_config.get("task_type", ""),
        base_model_name_or_path=raw_config.get("base_model_name_or_path", ""),
    )

    # Load weights
    weights = {}
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    elif bin_path.exists():
        weights = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise ModelLoadError(f"No adapter weights found in {path}")

    # Identify affected layers
    affected = set()
    for key in weights:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:
                affected.add(f"layer_{part}")
                break

    result = LoRAWeights(
        config=config,
        weights=weights,
        affected_layers=sorted(affected),
    )

    logger.info(
        "Loaded LoRA adapter: rank=%d, alpha=%.1f, targets=%s, %d params",
        config.r,
        config.lora_alpha,
        config.target_modules,
        sum(w.numel() for w in weights.values()),
    )

    return result
