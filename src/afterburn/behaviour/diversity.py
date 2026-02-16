"""Output diversity analysis using EAD and optional SBERT semantic similarity.

Two complementary diversity metrics:

1. **EAD** (Expectation-Adjusted Distinct N-grams): Lexical diversity.
   From "Rethinking and Refining the Distinct Metric" (Liu et al., ACL 2022),
   referenced by Kirkman et al. (ICLR 2024). Corrects for length bias by
   normalizing observed distinct n-grams against uniform random expectation:

       EAD = D / (V * (1 - ((V-1)/V)^T))

   where D = observed distinct n-grams, V = vocabulary size, T = total n-grams.

2. **SBERT Semantic Diversity** (optional): Embeds outputs using a sentence
   transformer and computes average pairwise cosine similarity. Lower
   similarity = more diverse. Requires `sentence-transformers` package.
"""

from __future__ import annotations

import logging

from afterburn.types import DiversityAnalysis

logger = logging.getLogger(__name__)

# Re-export so existing imports from this module keep working
__all__ = ["DiversityAnalysis", "analyze_diversity", "compute_semantic_diversity"]


def analyze_diversity(
    base_outputs: list[str],
    trained_outputs: list[str],
    base_categories: list[str] | None = None,
    trained_categories: list[str] | None = None,
    max_n: int = 5,
    vocab_size: int | None = None,
) -> DiversityAnalysis:
    """Compare output diversity between base and trained model outputs.

    Args:
        base_outputs: List of text outputs from the base model.
        trained_outputs: List of text outputs from the trained model.
        base_categories: Optional category labels for base outputs.
        trained_categories: Optional category labels for trained outputs.
        max_n: Maximum n-gram level (default 5).
        vocab_size: External vocabulary size (e.g. tokenizer vocab). If None,
            estimated from the union of all n-grams in both output sets.

    Returns:
        DiversityAnalysis with EAD scores and diversity comparison.
    """
    # Estimate per-n vocabulary sizes from combined corpus if not provided
    vocab_sizes = _estimate_vocab_sizes(base_outputs, trained_outputs, max_n, vocab_size)

    base_ead = _compute_ead_all(base_outputs, max_n, vocab_sizes)
    trained_ead = _compute_ead_all(trained_outputs, max_n, vocab_sizes)

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
                cat_vocab = _estimate_vocab_sizes(base_cat, trained_cat, max_n, vocab_size)
                b_ead = _mean_ead(_compute_ead_all(base_cat, max_n, cat_vocab))
                t_ead = _mean_ead(_compute_ead_all(trained_cat, max_n, cat_vocab))
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


def _estimate_vocab_sizes(
    base_outputs: list[str],
    trained_outputs: list[str],
    max_n: int,
    fixed_vocab_size: int | None = None,
) -> dict[int, int]:
    """Estimate vocabulary size per n-gram level from combined outputs.

    If fixed_vocab_size is provided, uses that for all n-gram levels.
    Otherwise, uses the union of all distinct n-grams from both output sets.
    """
    vocab_sizes: dict[int, int] = {}
    for n in range(1, max_n + 1):
        if fixed_vocab_size is not None:
            vocab_sizes[n] = fixed_vocab_size
        else:
            all_ngrams: set[tuple[str, ...]] = set()
            for text in base_outputs + trained_outputs:
                tokens = text.lower().split()
                for ngram in _extract_ngrams(tokens, n):
                    all_ngrams.add(ngram)
            vocab_sizes[n] = max(len(all_ngrams), 1)
    return vocab_sizes


def _compute_ead_all(
    outputs: list[str], max_n: int, vocab_sizes: dict[int, int]
) -> dict[int, float]:
    """Compute EAD for n=1..max_n across all outputs."""
    result = {}
    for n in range(1, max_n + 1):
        result[n] = _compute_ead(outputs, n, vocab_sizes.get(n, 1))
    return result


def _compute_ead(outputs: list[str], n: int, vocab_size: int) -> float:
    """Compute Expectation-Adjusted Distinct n-grams (EAD).

    From Liu et al. "Rethinking and Refining the Distinct Metric" (ACL 2022).

        EAD = D / (V * (1 - ((V-1)/V)^T))

    where D = observed distinct n-grams, V = vocabulary size, T = total n-grams.
    EAD ≈ 1.0 means diversity matches uniform random expectation.
    EAD < 1.0 means less diverse (repetitive). EAD > 1.0 is possible but rare.
    """
    all_ngrams: list[tuple[str, ...]] = []
    for text in outputs:
        tokens = text.lower().split()
        ngrams = _extract_ngrams(tokens, n)
        all_ngrams.extend(ngrams)

    total_ngrams = len(all_ngrams)
    if total_ngrams == 0:
        return 0.0

    distinct_count = len(set(all_ngrams))
    vocab = max(vocab_size, 1)

    # Expected distinct n-grams under uniform random sampling from vocab V
    # E[D] = V * (1 - ((V-1)/V)^T)
    expected_distinct = vocab * (1.0 - ((vocab - 1) / vocab) ** total_ngrams)

    if expected_distinct <= 0:
        return 0.0

    return distinct_count / expected_distinct


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


_sbert_model = None


def _get_sbert_model() -> object:
    """Lazy singleton for SentenceTransformer model."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def compute_semantic_diversity(outputs: list[str]) -> float | None:
    """Compute semantic diversity using sentence embeddings.

    Diversity = 1 - mean(pairwise cosine similarities).
    Returns None if sentence-transformers is not installed.
    Returns a value in [0, 1] where 1 = maximally diverse.
    """
    if len(outputs) < 2:
        return 0.0

    try:
        import numpy as np
        model = _get_sbert_model()
    except ImportError:
        logger.debug("sentence-transformers not installed, skipping semantic diversity")
        return None

    # Truncate very long outputs to first 512 chars for embedding
    truncated = [text[:512] for text in outputs]
    embeddings = model.encode(truncated, normalize_embeddings=True)  # type: ignore[attr-defined]

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
