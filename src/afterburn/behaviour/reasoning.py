"""Reasoning strategy classification and shift analysis."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from afterburn.types import PromptResult, ReasoningStrategy, StrategyShiftAnalysis


@dataclass
class ClassificationResult:
    """Detailed result of reasoning strategy classification."""

    strategy: ReasoningStrategy
    confidence: float
    scores: dict[str, float]


# ─── Pattern definitions ────────────────────────────────────────────

_TOOL_PATTERNS = [
    r"(?:```(?:bash|shell|sh)\n|^\$\s)",
    r"\b(?:curl|wget|pip install|npm|apt-get|brew)\b",
    r"(?:API|endpoint|request|response)\s+(?:call|GET|POST|PUT|DELETE)",
    r"\b(?:function_call|tool_use)\b",
]

_CODE_PATTERNS = [
    r"```(?:python|java|cpp|javascript|typescript|code|sql|bash|rust|go|c\b)",
    r"```\n",
    r"\bdef\s+\w+\s*\(",
    r"\bimport\s+\w+",
    r"\bclass\s+\w+",
    r"\bfor\s+\w+\s+in\s+",
    r"\bwhile\s+.*:",
    r"\breturn\s+",
    r"\bif\s+.*:\s*$",
    r"\bprint\s*\(",
]

_STEP_PATTERNS = [
    r"Step\s+\d+[:\.]",
    r"^\s*\d+\.\s+\S",
    r"(?:First|Second|Third|Fourth|Fifth),?\s",
    r"(?:Firstly|Secondly|Thirdly),?\s",
    r"(?:Next|Then|After that|Finally|Lastly),?\s",
]

_COT_PATTERNS = [
    r"(?:Let me think|Let's think|Let's break|Let me break)",
    r"(?:I need to|We need to|We can|I can)\s+(?:first|consider|think|analyze)",
    r"(?:So,?\s+|Therefore,?\s+|This means|This implies)",
    r"(?:Wait,?\s+|Actually,?\s+|Hmm|On second thought)",
    r"(?:Let's verify|Let me check|double.check|verify this)",
]


def _count_matches(text: str, patterns: list[str], flags: int = 0) -> int:
    return sum(len(re.findall(p, text, flags)) for p in patterns)


# ─── Public API ─────────────────────────────────────────────────────


def classify_reasoning_strategy(output: str) -> ReasoningStrategy:
    """Classify the reasoning strategy used in a model output.

    Priority-based: tool_use > code_assisted > step_by_step > cot > direct.
    Each strategy must meet a minimum match threshold to qualify.
    """
    return classify_reasoning_strategy_detailed(output).strategy


def classify_reasoning_strategy_detailed(output: str) -> ClassificationResult:
    """Classify with match counts for each strategy."""
    if not output.strip():
        return ClassificationResult(
            strategy=ReasoningStrategy.UNKNOWN,
            confidence=1.0,
            scores={s.value: 0.0 for s in ReasoningStrategy},
        )

    # Count matches for each strategy
    tool = _count_matches(output, _TOOL_PATTERNS, re.MULTILINE | re.IGNORECASE)
    code = _count_matches(output, _CODE_PATTERNS, re.MULTILINE)
    step = _count_matches(output, _STEP_PATTERNS, re.MULTILINE | re.IGNORECASE)
    cot = _count_matches(output, _COT_PATTERNS, re.IGNORECASE)
    word_count = len(output.split())

    scores = {
        "tool_use": float(tool),
        "code_assisted": float(code),
        "step_by_step": float(step),
        "chain_of_thought": float(cot),
        "direct_answer": 0.0,
        "unknown": 0.0,
    }

    # Priority-based classification with minimum thresholds
    if tool >= 2:
        strategy = ReasoningStrategy.TOOL_USE
    elif code >= 3:
        strategy = ReasoningStrategy.CODE_ASSISTED
    elif step >= 2:
        strategy = ReasoningStrategy.STEP_BY_STEP
    elif cot >= 2:
        strategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    elif word_count < 50:
        strategy = ReasoningStrategy.DIRECT_ANSWER
        scores["direct_answer"] = 1.0
    else:
        # Long text with no clear signals → chain of thought by default
        strategy = ReasoningStrategy.CHAIN_OF_THOUGHT

    # Confidence: how dominant is the top signal vs others?
    top = scores.get(strategy.value, 1.0)
    others = [v for k, v in scores.items() if k != strategy.value and v > 0]
    if top > 0 and others:
        confidence = 1.0 - (max(others) / (top + max(others)))
    else:
        confidence = 1.0

    return ClassificationResult(
        strategy=strategy,
        confidence=max(0.0, min(confidence, 1.0)),
        scores=scores,
    )


# ─── Strategy shift analysis ───────────────────────────────────────


def analyze_strategy_shift(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
) -> StrategyShiftAnalysis:
    """Detect if post-training shifted reasoning strategies."""
    if not base_results or not trained_results:
        return StrategyShiftAnalysis()

    base_strategies = [classify_reasoning_strategy(r.output_text) for r in base_results]
    trained_strategies = [classify_reasoning_strategy(r.output_text) for r in trained_results]

    base_dist = _compute_distribution(base_strategies)
    trained_dist = _compute_distribution(trained_strategies)

    base_entropy = _compute_entropy(base_strategies)
    trained_entropy = _compute_entropy(trained_strategies)

    dominant_shift = _find_dominant_shift(base_dist, trained_dist)

    return StrategyShiftAnalysis(
        base_distribution=base_dist,
        trained_distribution=trained_dist,
        dominant_shift=dominant_shift,
        base_entropy=base_entropy,
        trained_entropy=trained_entropy,
        entropy_change=trained_entropy - base_entropy,
    )


def _compute_distribution(strategies: list[ReasoningStrategy]) -> dict[str, float]:
    if not strategies:
        return {}
    counts = Counter(s.value for s in strategies)
    total = len(strategies)
    return {k: v / total for k, v in sorted(counts.items())}


def _compute_entropy(strategies: list[ReasoningStrategy]) -> float:
    if not strategies:
        return 0.0
    counts = Counter(strategies)
    total = len(strategies)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _find_dominant_shift(
    base_dist: dict[str, float],
    trained_dist: dict[str, float],
) -> str:
    all_strategies = set(base_dist.keys()) | set(trained_dist.keys())

    max_increase_strategy = ""
    max_increase = 0.0
    max_decrease_strategy = ""
    max_decrease = 0.0

    for s in all_strategies:
        diff = trained_dist.get(s, 0.0) - base_dist.get(s, 0.0)
        if diff > max_increase:
            max_increase = diff
            max_increase_strategy = s
        if diff < max_decrease:
            max_decrease = diff
            max_decrease_strategy = s

    if max_increase > 0.1:
        if max_decrease < -0.1:
            return f"{max_decrease_strategy} → {max_increase_strategy}"
        return f"increased {max_increase_strategy}"

    if max_decrease < -0.1:
        return f"decreased {max_decrease_strategy}"

    return ""
