"""Attention head importance analysis."""

from __future__ import annotations

import logging
import re

import torch

from afterburn.types import AttentionHeadScore

logger = logging.getLogger(__name__)


def compute_head_importance(
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> list[float]:
    """Compute importance score per head based on weight magnitude.

    Takes a Q, K, or V projection weight matrix and computes
    the Frobenius norm for each head's slice.

    Args:
        weight: Shape [num_heads * head_dim, hidden_size] or [hidden_size, num_heads * head_dim]
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        List of importance scores (Frobenius norm) per head.
    """
    w = weight.float()

    # Handle both weight orientations
    expected_head_total = num_heads * head_dim
    if w.shape[0] == expected_head_total:
        # [num_heads * head_dim, hidden_size]
        w = w.reshape(num_heads, head_dim, -1)
    elif w.shape[1] == expected_head_total:
        # [hidden_size, num_heads * head_dim]
        w = w.t().reshape(num_heads, head_dim, -1)
    else:
        logger.warning(
            "Weight shape %s doesn't match expected head dimensions "
            "(num_heads=%d, head_dim=%d). Skipping head analysis.",
            w.shape,
            num_heads,
            head_dim,
        )
        return [0.0] * num_heads

    scores = []
    for h in range(num_heads):
        score = torch.linalg.matrix_norm(w[h], ord="fro").item()
        scores.append(score)

    return scores


def compare_attention_heads(
    base_params: dict[str, torch.Tensor],
    trained_params: dict[str, torch.Tensor],
    layer_index: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
) -> list[AttentionHeadScore]:
    """Compare attention head importance between base and trained models.

    Looks for Q, K, V projection weights in the parameter dicts
    and computes per-head importance deltas.
    """
    # Find Q/K/V projection keys
    q_key = _find_key(base_params, ["q_proj", "query", "q_attn"])
    k_key = _find_key(base_params, ["k_proj", "key", "k_attn"])
    v_key = _find_key(base_params, ["v_proj", "value", "v_attn"])

    if q_key is None:
        logger.debug("No Q projection found for layer %d", layer_index)
        return []

    # Compute Q head importance (always uses num_heads)
    base_q_scores = compute_head_importance(base_params[q_key], num_heads, head_dim)
    trained_q_scores = compute_head_importance(trained_params[q_key], num_heads, head_dim)

    # K and V may use GQA (grouped query attention) with fewer heads
    kv_heads = num_kv_heads or num_heads

    base_k_scores = [0.0] * num_heads
    trained_k_scores = [0.0] * num_heads
    base_v_scores = [0.0] * num_heads
    trained_v_scores = [0.0] * num_heads

    if k_key and k_key in base_params and k_key in trained_params:
        raw_base_k = compute_head_importance(base_params[k_key], kv_heads, head_dim)
        raw_trained_k = compute_head_importance(trained_params[k_key], kv_heads, head_dim)
        # Expand KV scores to match Q heads if GQA
        base_k_scores = _expand_kv_scores(raw_base_k, num_heads)
        trained_k_scores = _expand_kv_scores(raw_trained_k, num_heads)

    if v_key and v_key in base_params and v_key in trained_params:
        raw_base_v = compute_head_importance(base_params[v_key], kv_heads, head_dim)
        raw_trained_v = compute_head_importance(trained_params[v_key], kv_heads, head_dim)
        base_v_scores = _expand_kv_scores(raw_base_v, num_heads)
        trained_v_scores = _expand_kv_scores(raw_trained_v, num_heads)

    # Combine Q + K + V importance per head
    results = []
    for h in range(num_heads):
        base_imp = base_q_scores[h] + base_k_scores[h] + base_v_scores[h]
        trained_imp = trained_q_scores[h] + trained_k_scores[h] + trained_v_scores[h]
        delta = trained_imp - base_imp

        results.append(
            AttentionHeadScore(
                layer_index=layer_index,
                head_index=h,
                base_importance=base_imp,
                trained_importance=trained_imp,
                importance_delta=delta,
            )
        )

    return results


def _find_key(params: dict[str, torch.Tensor], patterns: list[str]) -> str | None:
    """Find a parameter key matching any of the given patterns."""
    for key in params:
        for pattern in patterns:
            if pattern in key and "weight" in key:
                return key
    # Also check without 'weight' suffix
    for key in params:
        for pattern in patterns:
            if pattern in key:
                return key
    return None


def _expand_kv_scores(kv_scores: list[float], num_heads: int) -> list[float]:
    """Expand KV head scores to match Q heads (for GQA)."""
    num_kv = len(kv_scores)
    if num_kv == num_heads:
        return kv_scores
    if num_kv == 0:
        return [0.0] * num_heads

    # GQA: each KV head is shared by num_heads // num_kv Q heads
    group_size = num_heads // num_kv
    expanded = []
    for score in kv_scores:
        expanded.extend([score] * group_size)

    # Handle remainder if num_heads is not evenly divisible
    while len(expanded) < num_heads:
        expanded.append(kv_scores[-1])

    return expanded[:num_heads]
