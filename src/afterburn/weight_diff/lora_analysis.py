"""LoRA adapter decomposition and impact analysis."""

from __future__ import annotations

import logging
from typing import Any

import torch

from afterburn.loading.lora_loader import LoRAWeights

logger = logging.getLogger(__name__)


def analyze_lora_adapter(lora: LoRAWeights) -> dict[str, Any]:
    """Analyze a LoRA adapter's weight distribution and impact.

    Returns a dict with analysis results including:
    - per_layer_norms: L2 norm of adapter weights per layer
    - total_params: total number of adapter parameters
    - rank: adapter rank
    - alpha: adapter alpha
    - scaling: effective scaling factor (alpha / rank)
    - most_affected_layers: layers with highest adapter weight norms
    """
    per_layer_norms: dict[str, float] = {}
    total_params = 0

    for key, tensor in lora.weights.items():
        total_params += tensor.numel()
        # Group by layer
        layer_name = _extract_layer_name(key)
        norm = torch.linalg.vector_norm(tensor.float()).item()
        per_layer_norms[layer_name] = per_layer_norms.get(layer_name, 0.0) + norm

    # Sort by impact
    sorted_layers = sorted(per_layer_norms.items(), key=lambda x: x[1], reverse=True)

    scaling = lora.config.lora_alpha / max(lora.config.r, 1)

    return {
        "rank": lora.config.r,
        "alpha": lora.config.lora_alpha,
        "scaling": scaling,
        "target_modules": lora.config.target_modules,
        "total_params": total_params,
        "per_layer_norms": per_layer_norms,
        "most_affected_layers": sorted_layers[:10],
        "affected_layer_count": len(per_layer_norms),
    }


def _extract_layer_name(key: str) -> str:
    """Extract a logical layer name from a LoRA parameter key."""
    parts = key.split(".")
    for _i, part in enumerate(parts):
        if part.isdigit():
            return f"layer_{part}"
    return "unknown"
