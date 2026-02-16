"""Embedding layer drift measurement."""

from __future__ import annotations

import logging

import torch

from afterburn.types import EmbeddingDrift
from afterburn.weight_diff.metrics import cosine_similarity, l2_norm_diff

logger = logging.getLogger(__name__)


def measure_embedding_drift(
    base_params: dict[str, torch.Tensor],
    trained_params: dict[str, torch.Tensor],
    top_k: int = 20,
) -> EmbeddingDrift:
    """Measure drift in embedding layers.

    Computes L2 norm and cosine similarity for input embeddings
    (embed_tokens) and optionally output embeddings (lm_head).
    Also identifies the top-k most drifted token embeddings.
    """
    # Find input embedding
    input_key = _find_embedding_key(base_params, is_input=True)
    output_key = _find_embedding_key(base_params, is_input=False)

    if input_key is None:
        logger.warning("No input embedding found")
        return EmbeddingDrift(
            input_embedding_l2=0.0,
            input_embedding_cosine=1.0,
            output_embedding_l2=None,
            output_embedding_cosine=None,
            top_drifted_tokens=[],
        )

    base_embed = base_params[input_key]
    trained_embed = trained_params.get(input_key)

    if trained_embed is None:
        logger.warning("Input embedding key '%s' not found in trained model", input_key)
        return EmbeddingDrift(
            input_embedding_l2=0.0,
            input_embedding_cosine=1.0,
            output_embedding_l2=None,
            output_embedding_cosine=None,
            top_drifted_tokens=[],
        )

    input_l2 = l2_norm_diff(base_embed, trained_embed)
    input_cos = cosine_similarity(base_embed, trained_embed)

    # Per-token drift
    top_drifted = _compute_per_token_drift(base_embed, trained_embed, top_k)

    # Output embedding (lm_head)
    output_l2 = None
    output_cos = None
    if output_key and output_key in base_params and output_key in trained_params:
        output_l2 = l2_norm_diff(base_params[output_key], trained_params[output_key])
        output_cos = cosine_similarity(base_params[output_key], trained_params[output_key])

    return EmbeddingDrift(
        input_embedding_l2=input_l2,
        input_embedding_cosine=input_cos,
        output_embedding_l2=output_l2,
        output_embedding_cosine=output_cos,
        top_drifted_tokens=top_drifted,
    )


def _compute_per_token_drift(
    base: torch.Tensor,
    trained: torch.Tensor,
    top_k: int,
) -> list[tuple[int, float]]:
    """Find the top-k tokens with the largest embedding drift."""
    diff = (trained.float() - base.float())
    # Per-token L2 norm
    per_token_drift = torch.linalg.vector_norm(diff, dim=-1)

    k = min(top_k, per_token_drift.shape[0])
    top_values, top_indices = torch.topk(per_token_drift, k)

    return [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values, strict=False)]


def _find_embedding_key(params: dict[str, torch.Tensor], is_input: bool) -> str | None:
    """Find embedding parameter key."""
    if is_input:
        patterns = [
            "embed_tokens.weight",
            "wte.weight",
            "word_embeddings.weight",
            "embed_in.weight",
        ]
    else:
        patterns = ["lm_head.weight", "embed_out.weight"]

    for key in params:
        for pattern in patterns:
            if key.endswith(pattern):
                return key
    return None
