"""Token probability KL divergence between base and trained model outputs.

Measures how differently two models "think" at the token probability level,
beyond just the surface text. Inspired by LMdiff (Strobelt et al., EMNLP 2021).

When both models have token probability data (collected via collect_logits=True),
this computes per-prompt and aggregate KL divergence between their output
distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from afterburn.types import PromptResult


@dataclass
class TokenDivergenceAnalysis:
    """Token-level probability divergence between base and trained models."""

    # Mean KL divergence across all prompts (bits)
    mean_kl_divergence: float = 0.0
    # Per-prompt KL divergences
    per_prompt_kl: list[float] = field(default_factory=list)
    # Prompts with highest divergence (prompt_id, kl_value)
    top_divergent_prompts: list[tuple[str, float]] = field(default_factory=list)
    # Per-category average KL
    per_category: dict[str, float] = field(default_factory=dict)
    # Whether token probability data was available
    has_token_probs: bool = False
    # Number of prompts analyzed
    num_prompts_analyzed: int = 0


def analyze_token_divergence(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
    top_n: int = 5,
) -> TokenDivergenceAnalysis:
    """Compute KL divergence between base and trained token probability distributions.

    For each prompt that has token probability data in both models, computes
    the average KL divergence across generated tokens. Returns aggregate
    statistics and identifies the most divergent prompts.

    Args:
        base_results: Results from the base model (with top_token_probs).
        trained_results: Results from the trained model (with top_token_probs).
        top_n: Number of top divergent prompts to report.

    Returns:
        TokenDivergenceAnalysis with KL divergence metrics.
    """
    # Match results by prompt_id
    base_by_id = {r.prompt_id: r for r in base_results}
    trained_by_id = {r.prompt_id: r for r in trained_results}

    common_ids = set(base_by_id.keys()) & set(trained_by_id.keys())

    per_prompt_kl: list[tuple[str, str, float]] = []  # (prompt_id, category, kl)

    for pid in sorted(common_ids):
        base_r = base_by_id[pid]
        trained_r = trained_by_id[pid]

        if not base_r.top_token_probs or not trained_r.top_token_probs:
            continue

        kl = _compute_prompt_kl(base_r.top_token_probs, trained_r.top_token_probs)
        if kl is not None:
            per_prompt_kl.append((pid, base_r.category, kl))

    if not per_prompt_kl:
        return TokenDivergenceAnalysis(has_token_probs=False)

    kl_values = [kl for _, _, kl in per_prompt_kl]
    mean_kl = sum(kl_values) / len(kl_values)

    # Top divergent prompts
    sorted_by_kl = sorted(per_prompt_kl, key=lambda x: x[2], reverse=True)
    top_divergent = [(pid, kl) for pid, _, kl in sorted_by_kl[:top_n]]

    # Per-category
    from collections import defaultdict
    cat_kls: dict[str, list[float]] = defaultdict(list)
    for _, cat, kl in per_prompt_kl:
        cat_kls[cat].append(kl)

    per_category = {
        cat: sum(kls) / len(kls) for cat, kls in sorted(cat_kls.items())
    }

    return TokenDivergenceAnalysis(
        mean_kl_divergence=mean_kl,
        per_prompt_kl=kl_values,
        top_divergent_prompts=top_divergent,
        per_category=per_category,
        has_token_probs=True,
        num_prompts_analyzed=len(per_prompt_kl),
    )


def _compute_prompt_kl(
    base_probs: list[dict[str, float]],
    trained_probs: list[dict[str, float]],
) -> float | None:
    """Compute average KL divergence across token positions for one prompt.

    KL(base || trained) = sum(base[t] * log(base[t] / trained[t]))

    Uses the top-k probability dictionaries from both models. Tokens not
    in the other model's top-k get a small epsilon probability.
    """
    if not base_probs or not trained_probs:
        return None

    eps = 1e-10
    kl_sum = 0.0
    num_steps = min(len(base_probs), len(trained_probs))

    if num_steps == 0:
        return None

    for step in range(num_steps):
        base_dist = base_probs[step]
        trained_dist = trained_probs[step]

        if not base_dist:
            continue

        # Get all tokens from both distributions
        all_tokens = set(base_dist.keys()) | set(trained_dist.keys())

        step_kl = 0.0
        for token in all_tokens:
            p = base_dist.get(token, eps)
            q = trained_dist.get(token, eps)
            if p > eps:
                step_kl += p * math.log(max(p, eps) / max(q, eps))

        kl_sum += max(0.0, step_kl)

    return kl_sum / num_steps
