"""Token probability Jensen-Shannon divergence between base and trained models.

Measures how differently two models "think" at the token probability level,
beyond just the surface text. Inspired by LMdiff (Strobelt et al., EMNLP 2021).

Uses JSD instead of KL divergence because:
- JSD is symmetric: JSD(P||Q) = JSD(Q||P)
- JSD is always defined (no division-by-zero when supports differ)
- JSD is bounded: 0 <= JSD <= 1 (with log base 2)

    JSD(P || Q) = (1/2) * KL(P || M) + (1/2) * KL(Q || M)
    where M = (P + Q) / 2
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from afterburn.ci import bootstrap_ci
from afterburn.types import ConfidenceInterval, PromptResult, TokenDivergenceAnalysis

# Re-export so existing imports from this module keep working
__all__ = ["TokenDivergenceAnalysis", "analyze_token_divergence"]


def analyze_token_divergence(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
    top_n: int = 5,
) -> TokenDivergenceAnalysis:
    """Compute JSD between base and trained token probability distributions.

    For each prompt that has token probability data in both models, computes
    the average Jensen-Shannon divergence across generated tokens.

    Args:
        base_results: Results from the base model (with top_token_probs).
        trained_results: Results from the trained model (with top_token_probs).
        top_n: Number of top divergent prompts to report.

    Returns:
        TokenDivergenceAnalysis with JSD metrics.
    """
    # Match results by prompt_id
    base_by_id = {r.prompt_id: r for r in base_results}
    trained_by_id = {r.prompt_id: r for r in trained_results}

    common_ids = set(base_by_id.keys()) & set(trained_by_id.keys())

    per_prompt: list[tuple[str, str, float]] = []  # (prompt_id, category, jsd)

    for pid in sorted(common_ids):
        base_r = base_by_id[pid]
        trained_r = trained_by_id[pid]

        if not base_r.top_token_probs or not trained_r.top_token_probs:
            continue

        jsd = _compute_prompt_jsd(base_r.top_token_probs, trained_r.top_token_probs)
        if jsd is not None:
            per_prompt.append((pid, base_r.category, jsd))

    if not per_prompt:
        return TokenDivergenceAnalysis(has_token_probs=False)

    jsd_values = [jsd for _, _, jsd in per_prompt]
    mean_jsd = sum(jsd_values) / len(jsd_values)

    # Top divergent prompts
    sorted_by_jsd = sorted(per_prompt, key=lambda x: x[2], reverse=True)
    top_divergent = [(pid, jsd) for pid, _, jsd in sorted_by_jsd[:top_n]]

    # Per-category
    cat_jsds: dict[str, list[float]] = defaultdict(list)
    for _, cat, jsd in per_prompt:
        cat_jsds[cat].append(jsd)

    per_category = {
        cat: sum(jsds) / len(jsds) for cat, jsds in sorted(cat_jsds.items())
    }

    # Bootstrap CI for mean JSD
    jsd_arr = np.array(jsd_values)
    ci_lower, ci_upper = bootstrap_ci(jsd_arr, statistic="mean")
    ci_mean_jsd = ConfidenceInterval(lower=ci_lower, upper=ci_upper)

    return TokenDivergenceAnalysis(
        mean_jsd=mean_jsd,
        per_prompt_jsd=jsd_values,
        top_divergent_prompts=top_divergent,
        per_category=per_category,
        has_token_probs=True,
        num_prompts_analyzed=len(per_prompt),
        mean_jsd_ci=ci_mean_jsd,
    )


def _compute_prompt_jsd(
    base_probs: list[dict[str, float]],
    trained_probs: list[dict[str, float]],
) -> float | None:
    """Compute average Jensen-Shannon divergence across token positions.

    JSD(P || Q) = (1/2) * KL(P || M) + (1/2) * KL(Q || M)
    where M = (P + Q) / 2

    JSD is bounded [0, 1] when using log base 2.
    No epsilon smoothing needed — the mixture M guarantees non-zero denominators.
    """
    if not base_probs or not trained_probs:
        return None

    num_steps = min(len(base_probs), len(trained_probs))
    if num_steps == 0:
        return None

    jsd_sum = 0.0
    valid_steps = 0

    for step in range(num_steps):
        base_dist = base_probs[step]
        trained_dist = trained_probs[step]

        if not base_dist and not trained_dist:
            continue

        jsd = _jsd_dicts(base_dist, trained_dist)
        jsd_sum += jsd
        valid_steps += 1

    if valid_steps == 0:
        return None

    return jsd_sum / valid_steps


def _jsd_dicts(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two token probability dicts.

    Handles tokens appearing in only one distribution naturally via the
    mixture M = (P + Q) / 2, which is always > 0 for any token in either dist.
    """
    all_tokens = set(p.keys()) | set(q.keys())

    jsd = 0.0
    for token in all_tokens:
        p_i = p.get(token, 0.0)
        q_i = q.get(token, 0.0)
        m_i = 0.5 * (p_i + q_i)

        if m_i > 0:
            if p_i > 0:
                jsd += 0.5 * p_i * math.log2(p_i / m_i)
            if q_i > 0:
                jsd += 0.5 * q_i * math.log2(q_i / m_i)

    # Clamp to [0, 1] to handle floating point errors
    return max(0.0, min(1.0, jsd))
