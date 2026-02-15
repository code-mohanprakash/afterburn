"""Weight diff engine — orchestrates layer-by-layer weight comparison."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import torch

from afterburn.device import DeviceConfig
from afterburn.loading.layer_iterator import LayerIterator
from afterburn.loading.model_loader import ModelLoader
from afterburn.types import (
    AttentionHeadScore,
    EmbeddingDrift,
    LayerDiff,
    LayerNormShift,
    ModelPair,
    WeightDiffResult,
)
from afterburn.weight_diff.attention import compare_attention_heads
from afterburn.weight_diff.embedding import measure_embedding_drift
from afterburn.weight_diff.layernorm import detect_layernorm_shift
from afterburn.weight_diff.metrics import (
    cosine_similarity,
    frobenius_norm_diff,
    l2_norm_diff,
    relative_change,
    svd_analysis,
)
from afterburn.weight_diff.spectral import spectral_analysis

logger = logging.getLogger(__name__)


class WeightDiffEngine:
    """Orchestrates full weight diff analysis between two models."""

    def __init__(
        self,
        model_pair: ModelPair,
        device_config: DeviceConfig,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ):
        self.model_pair = model_pair
        self.device_config = device_config
        self._progress = progress_callback

    def run(self) -> WeightDiffResult:
        """Execute full weight diff analysis.

        1. Validate model compatibility
        2. Create LayerIterators for both models
        3. Iterate paired layers, computing metrics
        4. Run attention head analysis
        5. Run LayerNorm shift detection
        6. Run embedding drift measurement
        7. Aggregate into WeightDiffResult
        """
        loader = ModelLoader(self.device_config)

        # Step 1: Validate compatibility
        logger.info("Validating model compatibility...")
        loader.validate_compatibility(
            self.model_pair.base_model, self.model_pair.trained_model
        )

        # Step 2: Create iterators
        base_iter = LayerIterator(
            self.model_pair.base_model,
            device=self.device_config.device,
            dtype=self.device_config.dtype,
        )
        trained_iter = LayerIterator(
            self.model_pair.trained_model,
            device=self.device_config.device,
            dtype=self.device_config.dtype,
        )

        # Get model architecture info
        config = loader.load_config(self.model_pair.base_model)
        num_heads = getattr(config, "num_attention_heads", 0)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        hidden_size = getattr(config, "hidden_size", 0)
        head_dim = hidden_size // num_heads if num_heads > 0 else 0

        # Step 3-6: Iterate and analyze
        layer_diffs: list[LayerDiff] = []
        attention_heads: list[AttentionHeadScore] = []
        layernorm_shifts: list[LayerNormShift] = []
        embedding_drift_result: EmbeddingDrift | None = None
        total_params = 0
        changed_params = 0

        total_layers = len(base_iter.layer_names)
        current = 0

        for layer_name, base_params, trained_params in LayerIterator.paired_iterate(
            base_iter, trained_iter
        ):
            current += 1
            if self._progress:
                self._progress(layer_name, current, total_layers)

            logger.debug("Analyzing layer: %s (%d params)", layer_name, len(base_params))

            # Compute layer-level metrics
            layer_index = _extract_layer_index(layer_name)
            layer_diff = self._compute_layer_diff(
                layer_name, layer_index, base_params, trained_params
            )
            if layer_diff:
                layer_diffs.append(layer_diff)
                total_params += layer_diff.param_count
                if layer_diff.relative_change > 0.001:
                    changed_params += layer_diff.param_count

            # Attention head analysis (only for numbered layers)
            if layer_name.startswith("layer_") and num_heads > 0:
                heads = compare_attention_heads(
                    base_params,
                    trained_params,
                    layer_index=layer_index,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_kv_heads=num_kv_heads,
                )
                attention_heads.extend(heads)

            # LayerNorm shift detection
            if layer_name.startswith("layer_") or layer_name == "final_norm":
                shift = detect_layernorm_shift(
                    base_params, trained_params, layer_name, layer_index
                )
                if shift is not None:
                    layernorm_shifts.append(shift)

            # Embedding drift
            if layer_name in ("embedding", "lm_head"):
                if embedding_drift_result is None:
                    # Collect all embedding/lm_head params first
                    all_embed_base: dict[str, torch.Tensor] = {}
                    all_embed_trained: dict[str, torch.Tensor] = {}

                all_embed_base.update(base_params)
                all_embed_trained.update(trained_params)

        # Compute embedding drift from collected params
        if embedding_drift_result is None and all_embed_base:
            embedding_drift_result = measure_embedding_drift(
                all_embed_base, all_embed_trained
            )

        if embedding_drift_result is None:
            embedding_drift_result = EmbeddingDrift(
                input_embedding_l2=0.0,
                input_embedding_cosine=1.0,
                output_embedding_l2=None,
                output_embedding_cosine=None,
                top_drifted_tokens=[],
            )

        return WeightDiffResult(
            layer_diffs=layer_diffs,
            attention_heads=attention_heads,
            layernorm_shifts=layernorm_shifts,
            embedding_drift=embedding_drift_result,
            lora_analysis=None,
            total_param_count=total_params,
            changed_param_count=changed_params,
        )

    def _compute_layer_diff(
        self,
        layer_name: str,
        layer_index: int,
        base_params: dict[str, torch.Tensor],
        trained_params: dict[str, torch.Tensor],
    ) -> LayerDiff | None:
        """Compute aggregate diff metrics for a single layer."""
        # Collect all tensors that exist in both
        common_keys = set(base_params.keys()) & set(trained_params.keys())
        if not common_keys:
            return None

        # Concatenate all params for aggregate metrics
        base_flat = torch.cat([base_params[k].flatten() for k in sorted(common_keys)])
        trained_flat = torch.cat([trained_params[k].flatten() for k in sorted(common_keys)])

        # SVD analysis on the largest 2D weight matrix in this layer
        svd_result = None
        spectral_result = None
        for k in sorted(common_keys):
            if base_params[k].dim() >= 2:
                if svd_result is None:
                    svd_result = svd_analysis(base_params[k], trained_params[k])
                if spectral_result is None:
                    spectral_result = spectral_analysis(trained_params[k])
                if svd_result is not None and spectral_result is not None:
                    break

        return LayerDiff(
            layer_name=layer_name,
            layer_index=layer_index,
            l2_norm=l2_norm_diff(base_flat, trained_flat),
            cosine_similarity=cosine_similarity(base_flat, trained_flat),
            frobenius_norm=frobenius_norm_diff(base_flat, trained_flat),
            relative_change=relative_change(base_flat, trained_flat),
            param_count=base_flat.numel(),
            svd_top_singular_values=tuple(svd_result.top_singular_values) if svd_result else None,
            svd_effective_rank=svd_result.effective_rank if svd_result else None,
            svd_concentration_ratio=svd_result.concentration_ratio if svd_result else None,
            svd_stable_rank=svd_result.stable_rank if svd_result else None,
            spectral_alpha=spectral_result.alpha if spectral_result else None,
            spectral_alpha_quality=spectral_result.alpha_quality if spectral_result else None,
            spectral_stable_rank=spectral_result.stable_rank if spectral_result else None,
        )


def _extract_layer_index(layer_name: str) -> int:
    """Extract numeric index from layer name like 'layer_5'."""
    match = re.search(r"(\d+)", layer_name)
    if match:
        return int(match.group(1))
    return -1
