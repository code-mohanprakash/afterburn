"""Strategy collapse detection — flags convergence on single solution strategy."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from afterburn.behaviour.reasoning import classify_reasoning_strategy
from afterburn.ci import bootstrap_ci
from afterburn.types import (
    ConfidenceInterval,
    PromptResult,
    ReasoningStrategy,
    StrategyCollapseResult,
)


def detect_strategy_collapse(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
    threshold: float = 0.3,
    dominant_fraction_threshold: float = 0.6,
) -> StrategyCollapseResult:
    """Detect if post-training collapsed solution strategies.

    Example: Model used to solve math with both verbal reasoning
    and code. After RLVR, it ONLY uses code. This may inflate
    benchmarks but reduce genuine capability diversity.
    """
    if not base_results or not trained_results:
        return StrategyCollapseResult(
            score=0.0,
            base_entropy=0.0,
            trained_entropy=0.0,
            entropy_drop=0.0,
            is_flagged=False,
            detail="Insufficient data for strategy collapse analysis.",
        )

    base_strategies = [classify_reasoning_strategy(r.output_text) for r in base_results]
    trained_strategies = [classify_reasoning_strategy(r.output_text) for r in trained_results]

    base_entropy = _compute_entropy(base_strategies)
    trained_entropy = _compute_entropy(trained_strategies)
    entropy_drop = base_entropy - trained_entropy

    # Find dominant strategy in trained model
    trained_counts = Counter(trained_strategies)
    dominant = (
        trained_counts.most_common(1)[0]
        if trained_counts
        else (ReasoningStrategy.UNKNOWN, 0)
    )
    dominant_strategy = dominant[0].value
    dominant_fraction = dominant[1] / len(trained_strategies) if trained_strategies else 0.0

    # Score: based on entropy drop and dominance
    if entropy_drop > 0:
        # Calibrated sigmoid
        raw_signal = entropy_drop / threshold
        score = 100.0 / (1.0 + math.exp(-3.0 * (raw_signal - 1.0)))

        # Boost if one strategy is very dominant (>70%)
        if dominant_fraction > 0.7:
            score = min(score + 15.0, 100.0)
    else:
        score = 0.0  # Entropy increased = more diverse

    is_flagged = entropy_drop > threshold and dominant_fraction > dominant_fraction_threshold

    if is_flagged:
        detail = (
            f"Strategy collapse detected: entropy dropped from {base_entropy:.2f} "
            f"to {trained_entropy:.2f} (Δ={entropy_drop:+.2f}). "
            f"Dominant strategy: {dominant_strategy} ({dominant_fraction:.0%} of outputs)."
        )
    else:
        detail = (
            f"Strategy entropy: {base_entropy:.2f} → {trained_entropy:.2f} "
            f"(Δ={entropy_drop:+.2f}). "
            f"No significant strategy collapse detected."
        )

    # Bootstrap CI for entropy drop
    ci_entropy = None
    if len(trained_strategies) >= 3:
        # Bootstrap: resample trained strategies, compute entropy_drop for each
        rng = np.random.default_rng(42)
        n_boot = 1000
        boot_drops = np.empty(n_boot)
        strat_indices = np.arange(len(trained_strategies))
        for i in range(n_boot):
            sample_idx = rng.choice(strat_indices, size=len(trained_strategies), replace=True)
            boot_strats = [trained_strategies[j] for j in sample_idx]
            boot_drops[i] = base_entropy - _compute_entropy(boot_strats)
        lo = float(np.percentile(boot_drops, 2.5))
        hi = float(np.percentile(boot_drops, 97.5))
        ci_entropy = ConfidenceInterval(lower=round(lo, 6), upper=round(hi, 6))

    return StrategyCollapseResult(
        score=min(score, 100.0),
        base_entropy=base_entropy,
        trained_entropy=trained_entropy,
        entropy_drop=entropy_drop,
        dominant_strategy=dominant_strategy,
        is_flagged=is_flagged,
        detail=detail,
        entropy_drop_ci=ci_entropy,
    )


def _compute_entropy(strategies: list[ReasoningStrategy]) -> float:
    """Compute Shannon entropy of strategy distribution."""
    if not strategies:
        return 0.0
    counts = Counter(strategies)
    total = len(strategies)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)
