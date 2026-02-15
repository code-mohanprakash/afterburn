"""Output diversity analysis using EAD and optional SBERT semantic similarity.

Two complementary diversity metrics:

1. **EAD** (Expectation-Adjusted Distinct N-grams): Lexical diversity.
   Adapted from "Understanding the Effects of RLHF on LLM Generalisation
   and Diversity" (Kirkman et al., ICLR 2024). Corrects for length bias
   in standard distinct-n-gram metrics.

2. **SBERT Semantic Diversity** (optional): Embeds outputs using a sentence
   transformer and computes average pairwise cosine similarity. Lower
   similarity = more diverse. Requires `sentence-transformers` package.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DiversityAnalysis:
    """Output diversity comparison between base and trained models."""

    # EAD scores per n-gram level (1-5)
    base_ead: dict[int, float] = field(default_factory=dict)
    trained_ead: dict[int, float] = field(default_factory=dict)

    # Aggregated diversity score (mean EAD across n=1..5)
    base_diversity_score: float = 0.0
    trained_diversity_score: float = 0.0
    diversity_change: float = 0.0

    # SBERT semantic diversity (1 - avg pairwise cosine similarity)
    # None if sentence-transformers not installed
    base_semantic_diversity: float | None = None
    trained_semantic_diversity: float | None = None
    semantic_diversity_change: float | None = None

    # Per-category breakdown
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)


def analyze_diversity(
    base_outputs: list[str],
    trained_outputs: list[str],
    base_categories: list[str] | None = None,
    trained_categories: list[str] | None = None,
    max_n: int = 5,
) -> DiversityAnalysis:
    """Compare output diversity between base and trained model outputs.

    Args:
        base_outputs: List of text outputs from the base model.
        trained_outputs: List of text outputs from the trained model.
        base_categories: Optional category labels for base outputs.
        trained_categories: Optional category labels for trained outputs.
        max_n: Maximum n-gram level (default 5).

    Returns:
        DiversityAnalysis with EAD scores and diversity comparison.
    """
    base_ead = _compute_ead_all(base_outputs, max_n)
    trained_ead = _compute_ead_all(trained_outputs, max_n)

    base_score = _mean_ead(base_ead)
    trained_score = _mean_ead(trained_ead)

    # Per-category analysis
    per_category: dict[str, dict[str, float]] = {}
    if base_categories and trained_categories:
        cats = set(base_categories) | set(trained_categories)
        for cat in sorted(cats):
            base_cat = [
                base_outputs[i]
                for i, c in enumerate(base_categories)
                if c == cat
            ]
            trained_cat = [
                trained_outputs[i]
                for i, c in enumerate(trained_categories)
                if c == cat
            ]
            if base_cat and trained_cat:
                b_ead = _mean_ead(_compute_ead_all(base_cat, max_n))
                t_ead = _mean_ead(_compute_ead_all(trained_cat, max_n))
                per_category[cat] = {
                    "base_diversity": b_ead,
                    "trained_diversity": t_ead,
                    "diversity_change": t_ead - b_ead,
                }

    # SBERT semantic diversity (optional)
    base_sem = None
    trained_sem = None
    sem_change = None
    if base_outputs and trained_outputs:
        base_sem = compute_semantic_diversity(base_outputs)
        trained_sem = compute_semantic_diversity(trained_outputs)
        if base_sem is not None and trained_sem is not None:
            sem_change = trained_sem - base_sem

    return DiversityAnalysis(
        base_ead=base_ead,
        trained_ead=trained_ead,
        base_diversity_score=base_score,
        trained_diversity_score=trained_score,
        diversity_change=trained_score - base_score,
        base_semantic_diversity=base_sem,
        trained_semantic_diversity=trained_sem,
        semantic_diversity_change=sem_change,
        per_category=per_category,
    )


def _compute_ead_all(outputs: list[str], max_n: int) -> dict[int, float]:
    """Compute EAD for n=1..max_n across all outputs."""
    result = {}
    for n in range(1, max_n + 1):
        result[n] = _compute_ead(outputs, n)
    return result


def _compute_ead(outputs: list[str], n: int) -> float:
    """Compute Expectation-Adjusted Distinct n-grams (EAD).

    Standard distinct-n ratio = |unique n-grams| / |total n-grams|.
    This is biased by length: longer text naturally has lower ratios.

    EAD corrects this by computing the expected distinct ratio for
    uniformly random text of the same length, then normalizing:

        EAD = (observed_distinct_ratio - expected_ratio) / (1 - expected_ratio)

    When there aren't enough n-grams, we fall back to the raw distinct ratio.
    """
    # Tokenize all outputs into words
    all_ngrams: list[tuple[str, ...]] = []
    for text in outputs:
        tokens = text.lower().split()
        ngrams = _extract_ngrams(tokens, n)
        all_ngrams.extend(ngrams)

    total = len(all_ngrams)
    if total == 0:
        return 0.0

    unique = len(set(all_ngrams))
    observed_ratio = unique / total

    # Expected distinct ratio under uniform random sampling
    # Using the coupon collector approximation:
    # E[distinct] = V * (1 - (1 - 1/V)^T) where V = vocabulary size, T = total
    # We estimate V from the observed unique count
    # For the correction, we use the simpler approach from Liu et al. 2022:
    # expected_ratio = 1 - (1 - 1/unique)^total when unique > 0
    if unique <= 1 or total <= 1:
        return observed_ratio

    expected_ratio = 1.0 - (1.0 - 1.0 / unique) ** total
    if expected_ratio >= 1.0 - 1e-10:
        return observed_ratio

    # Normalize: how much more diverse than random expectation
    ead = (observed_ratio - expected_ratio) / (1.0 - expected_ratio)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, ead))


def _extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _mean_ead(ead_dict: dict[int, float]) -> float:
    """Compute mean EAD across all n-gram levels."""
    if not ead_dict:
        return 0.0
    return sum(ead_dict.values()) / len(ead_dict)


def compute_semantic_diversity(outputs: list[str]) -> float | None:
    """Compute semantic diversity using sentence embeddings.

    Diversity = 1 - mean(pairwise cosine similarities).
    Returns None if sentence-transformers is not installed.
    Returns a value in [0, 1] where 1 = maximally diverse.
    """
    if len(outputs) < 2:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.debug("sentence-transformers not installed, skipping semantic diversity")
        return None

    # Use a lightweight model for embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Truncate very long outputs to first 512 chars for embedding
    truncated = [text[:512] for text in outputs]
    embeddings = model.encode(truncated, normalize_embeddings=True)

    # Compute pairwise cosine similarity (already normalized = dot product)
    sim_matrix = np.dot(embeddings, embeddings.T)

    # Extract upper triangle (excluding diagonal)
    n = len(outputs)
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0

    upper_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            upper_sum += sim_matrix[i, j]

    avg_similarity = upper_sum / total_pairs
    diversity = 1.0 - avg_similarity

    return max(0.0, min(1.0, float(diversity)))
