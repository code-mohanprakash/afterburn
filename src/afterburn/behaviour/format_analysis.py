"""Format compliance scoring — detects formatting pattern changes."""

from __future__ import annotations

import math
import re
from collections import Counter

from afterburn.types import FormatAnalysis, PromptResult

# Common output format patterns
FORMAT_PATTERNS: dict[str, str] = {
    "boxed_answer": r"\\boxed\{.*?\}",
    "final_answer_marker": r"(?:The answer is|Therefore|Thus|Hence|So the answer is)[:\s]",
    "hash_format": r"####\s*",
    "json_output": r"\{[\s\S]*\"answer\"[\s\S]*\}",
    "code_block": r"```[\s\S]*?```",
    "numbered_steps": r"(?:Step \d|^\d+\.)",
    "bullet_points": r"^[\s]*[-*]\s",
    "latex_math": r"\$.*?\$",
    # New patterns
    "markdown_header": r"^#{1,6}\s+",
    "table_format": r"\|.*\|.*\|",
    "xml_tags": r"<[^>]+>.*?</[^>]+>",
    "thinking_tags": r"<think>[\s\S]*?</think>",
    "verification_marker": (
        r"(?:Let me verify|Let's verify|Let me check|Let's check|Verification)[:\s]"
    ),
}


def analyze_format_compliance(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
) -> FormatAnalysis:
    """Analyze format compliance changes between base and trained outputs.

    Detects if the trained model uses specific formatting patterns
    more or less than the base model.

    Args:
        base_results: Results from the base model
        trained_results: Results from the trained model

    Returns:
        FormatAnalysis containing:
        - patterns_detected: Per-pattern usage rates
        - base/trained_format_rate: Average format pattern usage
        - format_increase: Change in format usage
        - per_category: Format rates broken down by category
        - base/trained_format_diversity: Shannon entropy of pattern usage
    """
    if not base_results or not trained_results:
        return FormatAnalysis()

    patterns_detected: dict[str, dict[str, float]] = {}

    total_base_format = 0.0
    total_trained_format = 0.0
    pattern_count = 0

    # Track pattern usage for diversity calculation
    base_pattern_usage: Counter[str] = Counter()
    trained_pattern_usage: Counter[str] = Counter()

    # Overall pattern analysis
    for pattern_name, pattern_regex in FORMAT_PATTERNS.items():
        base_matches = _count_matches(base_results, pattern_regex)
        trained_matches = _count_matches(trained_results, pattern_regex)

        base_rate = base_matches / len(base_results)
        trained_rate = trained_matches / len(trained_results)

        if base_rate > 0 or trained_rate > 0:
            patterns_detected[pattern_name] = {
                "base_rate": base_rate,
                "trained_rate": trained_rate,
                "change": trained_rate - base_rate,
                "ratio": trained_rate / max(base_rate, 0.01),
            }
            total_base_format += base_rate
            total_trained_format += trained_rate
            pattern_count += 1

        # Track for diversity
        if base_matches > 0:
            base_pattern_usage[pattern_name] = base_matches
        if trained_matches > 0:
            trained_pattern_usage[pattern_name] = trained_matches

    avg_base_rate = total_base_format / max(pattern_count, 1)
    avg_trained_rate = total_trained_format / max(pattern_count, 1)
    format_increase = avg_trained_rate - avg_base_rate

    # Per-category analysis
    per_category = _compute_per_category_analysis(base_results, trained_results)

    # Format diversity: Shannon entropy over pattern frequency distribution
    # Higher entropy = more evenly distributed pattern usage = more diverse
    base_diversity = _shannon_entropy(base_pattern_usage)
    trained_diversity = _shannon_entropy(trained_pattern_usage)

    return FormatAnalysis(
        patterns_detected=patterns_detected,
        base_format_rate=avg_base_rate,
        trained_format_rate=avg_trained_rate,
        format_increase=format_increase,
        per_category=per_category,
        base_format_diversity=base_diversity,
        trained_format_diversity=trained_diversity,
    )


def _count_matches(results: list[PromptResult], pattern: str) -> int:
    """Count how many results contain a match for the given pattern."""
    count = 0
    regex = re.compile(pattern, re.MULTILINE)
    for r in results:
        if regex.search(r.output_text):
            count += 1
    return count


def _compute_per_category_analysis(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
) -> dict[str, dict[str, float]]:
    """Compute format rates per category.

    Returns:
        Dictionary mapping category to format metrics:
        {
            "category_name": {
                "base_format_rate": float,
                "trained_format_rate": float,
                "format_increase": float,
            }
        }
    """
    # Group results by category
    base_by_category: dict[str, list[PromptResult]] = {}
    trained_by_category: dict[str, list[PromptResult]] = {}

    for result in base_results:
        base_by_category.setdefault(result.category, []).append(result)

    for result in trained_results:
        trained_by_category.setdefault(result.category, []).append(result)

    # Get all categories (union of both sets)
    all_categories = set(base_by_category.keys()) | set(trained_by_category.keys())

    per_category: dict[str, dict[str, float]] = {}

    for category in all_categories:
        base_cat_results = base_by_category.get(category, [])
        trained_cat_results = trained_by_category.get(category, [])

        if not base_cat_results or not trained_cat_results:
            # Skip categories that don't exist in both sets
            continue

        # Count pattern matches for this category
        base_total_matches = 0
        trained_total_matches = 0
        pattern_count = 0

        for pattern_regex in FORMAT_PATTERNS.values():
            base_matches = _count_matches(base_cat_results, pattern_regex)
            trained_matches = _count_matches(trained_cat_results, pattern_regex)

            base_rate = base_matches / len(base_cat_results)
            trained_rate = trained_matches / len(trained_cat_results)

            if base_rate > 0 or trained_rate > 0:
                base_total_matches += base_matches
                trained_total_matches += trained_matches
                pattern_count += 1

        if pattern_count > 0:
            base_format_rate = (base_total_matches / len(base_cat_results)) / pattern_count
            trained_format_rate = (trained_total_matches / len(trained_cat_results)) / pattern_count
        else:
            base_format_rate = 0.0
            trained_format_rate = 0.0

        per_category[category] = {
            "base_format_rate": base_format_rate,
            "trained_format_rate": trained_format_rate,
            "format_increase": trained_format_rate - base_format_rate,
        }

    return per_category


def _shannon_entropy(counter: Counter[str]) -> float:
    """Compute Shannon entropy of a frequency distribution.

    H = -sum(p_i * log2(p_i)) where p_i = count_i / total

    Returns 0.0 for empty counters. Maximum entropy = log2(num_categories).
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy
