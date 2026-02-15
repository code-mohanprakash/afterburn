"""LayerNorm shift detection."""

from __future__ import annotations

import logging

import torch

from afterburn.types import LayerNormShift

logger = logging.getLogger(__name__)

# Threshold for significance (relative change in LayerNorm params)
SIGNIFICANCE_THRESHOLD = 0.01


def detect_layernorm_shift(
    base_params: dict[str, torch.Tensor],
    trained_params: dict[str, torch.Tensor],
    layer_name: str,
    layer_index: int,
    threshold: float = SIGNIFICANCE_THRESHOLD,
) -> LayerNormShift | None:
    """Detect significant shifts in LayerNorm parameters.

    LayerNorm shifts are strong indicators of behavioural changes
    because they control the scale and bias of activations flowing
    through the entire layer.

    Looks for LayerNorm/RMSNorm weight (gamma) and bias (beta) parameters.
    """
    gamma_key = _find_norm_weight(base_params)
    beta_key = _find_norm_bias(base_params)

    if gamma_key is None:
        return None

    # Compute gamma (scale) shift
    base_gamma = base_params[gamma_key].float()
    if gamma_key not in trained_params:
        return None
    trained_gamma = trained_params[gamma_key].float()

    gamma_diff = (trained_gamma - base_gamma).abs()
    gamma_shift = gamma_diff.mean().item()
    gamma_relative = gamma_shift / (base_gamma.abs().mean().item() + 1e-10)

    # Compute beta (bias) shift if present
    beta_shift = 0.0
    if beta_key and beta_key in base_params and beta_key in trained_params:
        base_beta = base_params[beta_key].float()
        trained_beta = trained_params[beta_key].float()
        beta_diff = (trained_beta - base_beta).abs()
        beta_shift = beta_diff.mean().item()

    total_shift = gamma_relative + beta_shift
    is_significant = total_shift > threshold

    return LayerNormShift(
        layer_name=layer_name,
        layer_index=layer_index,
        gamma_shift=gamma_shift,
        beta_shift=beta_shift,
        total_shift=total_shift,
        is_significant=is_significant,
    )


def _find_norm_weight(params: dict[str, torch.Tensor]) -> str | None:
    """Find LayerNorm/RMSNorm weight parameter."""
    patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "ln_1.weight",
        "ln_2.weight",
        "layer_norm.weight",
        "norm.weight",
    ]
    for key in params:
        for pattern in patterns:
            if key.endswith(pattern):
                return key
    # Fallback: any key containing 'norm' and 'weight'
    for key in params:
        if "norm" in key.lower() and "weight" in key.lower():
            return key
    return None


def _find_norm_bias(params: dict[str, torch.Tensor]) -> str | None:
    """Find LayerNorm bias parameter (not present in RMSNorm)."""
    for key in params:
        if "norm" in key.lower() and "bias" in key.lower():
            return key
    return None
