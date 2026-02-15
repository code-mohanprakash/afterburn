"""Chain-of-thought pattern analysis."""

from __future__ import annotations

import re

import numpy as np

from afterburn.types import ChainOfThoughtAnalysis, PromptResult


def analyze_chain_of_thought(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
) -> ChainOfThoughtAnalysis:
    """Analyze chain-of-thought patterns in model outputs.

    Measures:
    - Average number of reasoning steps
    - Average reasoning depth (nesting/complexity)
    - Changes in step count between models
    - Self-correction rates (detecting when model corrects itself)
    - Verification rates (detecting when model double-checks work)
    """
    if not base_results or not trained_results:
        return ChainOfThoughtAnalysis()

    # Count reasoning steps
    base_steps = [_count_reasoning_steps(r.output_text) for r in base_results]
    trained_steps = [_count_reasoning_steps(r.output_text) for r in trained_results]

    # Estimate reasoning depth
    base_depths = [_estimate_reasoning_depth(r.output_text) for r in base_results]
    trained_depths = [_estimate_reasoning_depth(r.output_text) for r in trained_results]

    # Detect self-corrections
    base_corrections = [_detect_self_correction(r.output_text) for r in base_results]
    trained_corrections = [_detect_self_correction(r.output_text) for r in trained_results]

    # Detect verifications
    base_verifications = [_detect_verification(r.output_text) for r in base_results]
    trained_verifications = [_detect_verification(r.output_text) for r in trained_results]

    # Calculate averages
    base_avg_steps = float(np.mean(base_steps)) if base_steps else 0.0
    trained_avg_steps = float(np.mean(trained_steps)) if trained_steps else 0.0

    base_avg_depth = float(np.mean(base_depths)) if base_depths else 0.0
    trained_avg_depth = float(np.mean(trained_depths)) if trained_depths else 0.0

    # Calculate changes
    step_count_change = trained_avg_steps - base_avg_steps
    depth_change = trained_avg_depth - base_avg_depth

    # Calculate rates (proportion of responses with self-correction/verification)
    base_self_correction_rate = float(np.mean(base_corrections)) if base_corrections else 0.0
    trained_self_correction_rate = float(np.mean(trained_corrections)) if trained_corrections else 0.0

    base_verification_rate = float(np.mean(base_verifications)) if base_verifications else 0.0
    trained_verification_rate = float(np.mean(trained_verifications)) if trained_verifications else 0.0

    return ChainOfThoughtAnalysis(
        base_avg_steps=base_avg_steps,
        trained_avg_steps=trained_avg_steps,
        base_avg_depth=base_avg_depth,
        trained_avg_depth=trained_avg_depth,
        step_count_change=step_count_change,
        depth_change=depth_change,
        base_self_correction_rate=base_self_correction_rate,
        trained_self_correction_rate=trained_self_correction_rate,
        base_verification_rate=base_verification_rate,
        trained_verification_rate=trained_verification_rate,
    )


def _count_reasoning_steps(text: str) -> int:
    """Count the number of explicit reasoning steps in text.

    Uses weighted pattern matching to avoid double-counting and provide
    more accurate step detection.
    """
    if not text.strip():
        return 0

    # Define patterns with weights
    # Higher weight = more likely to be a distinct reasoning step
    weighted_patterns = [
        (r"Step\s+\d+", 3.0),  # Explicit step numbering - highest weight
        (r"^\d+\.\s+\S", 2.5),  # Numbered list items at line start
        (r"^-\s+\S", 1.5),  # Bullet points at line start
        (r"(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)(?:ly)?[,:]?\s+", 2.0),
        (r"(?:Then|Next|After that|Finally|Lastly)[,:]?\s+", 1.5),
        (r"(?:Therefore|Thus|Hence|Consequently)[,:]?\s+", 1.2),
        (r"(?:Now|At this point|Moving forward)[,:]?\s+", 1.0),
    ]

    # Track which parts of the text we've already counted
    counted_positions = set()
    total_weighted_steps = 0.0

    for pattern, weight in weighted_patterns:
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            start, end = match.span()
            # Check if this position overlaps with already counted positions
            if not any(start <= pos < end for pos in counted_positions):
                total_weighted_steps += weight
                # Mark this position as counted
                for pos in range(start, end):
                    counted_positions.add(pos)

    # If we found explicit steps, use the weighted count
    if total_weighted_steps > 0:
        # Normalize weights: divide by average weight (roughly 1.8) to get step count
        return int(round(total_weighted_steps / 1.8))

    # Fallback: count paragraphs as implicit steps
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return len(paragraphs)

    # Ultimate fallback: count sentences
    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([s for s in sentences if s.strip() and len(s.strip()) > 10])
    return max(1, sentence_count // 3)  # Rough heuristic: 3 sentences = 1 step


def _estimate_reasoning_depth(text: str) -> float:
    """Estimate reasoning depth/complexity.

    Higher values indicate more complex, multi-layered reasoning.
    Measures:
    - Sentence count (longer reasoning = deeper)
    - Conditional/causal markers (if/then, because, therefore)
    - Sub-problem decomposition markers
    """
    if not text.strip():
        return 0.0

    # Sentence count (rough approximation)
    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Causal/logical connectors
    logical_markers = len(re.findall(
        r"\b(?:because|therefore|thus|hence|since|so|if|then|implies|means that|"
        r"it follows|consequently|as a result)\b",
        text,
        re.IGNORECASE,
    ))

    # Sub-problem markers
    subproblem_markers = len(re.findall(
        r"\b(?:first we need|let's consider|breaking this down|"
        r"sub-problem|case \d|scenario)\b",
        text,
        re.IGNORECASE,
    ))

    # Normalize to a 0-10 scale
    depth = min(
        (sentence_count * 0.3 + logical_markers * 1.5 + subproblem_markers * 2.0),
        10.0,
    )

    return depth


def _detect_self_correction(text: str) -> bool:
    """Detect if the text contains self-correction patterns.

    Returns True if the model corrects itself during reasoning.
    Looks for phrases indicating the model reconsidered or fixed a mistake.
    """
    if not text.strip():
        return False

    # Self-correction patterns (case-insensitive)
    correction_patterns = [
        r"\bWait,\s",
        r"\bActually,\s",
        r"\bI made a mistake\b",
        r"\bLet me reconsider\b",
        r"\bcorrection[:\s]",
        r"\bon second thought\b",
        r"\bNo,?\s+(?:that's|that is)\s+(?:wrong|incorrect)\b",
        r"\bI was wrong\b",
        r"\bLet me correct\b",
        r"\bUpon (?:reflection|reconsideration)\b",
        r"\bOops,?\s",
        r"\bMy mistake\b",
        r"\bI should have\b",
        r"\bRevising\s+(?:my|the)\s+(?:answer|approach)\b",
    ]

    for pattern in correction_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def _detect_verification(text: str) -> bool:
    """Detect if the text contains verification/double-checking patterns.

    Returns True if the model explicitly verifies or checks its work.
    Looks for phrases indicating the model is confirming or validating results.
    """
    if not text.strip():
        return False

    # Verification patterns (case-insensitive)
    verification_patterns = [
        r"\bLet's verify\b",
        r"\bLet me check\b",
        r"\bTo confirm\b",
        r"\bdouble-check\b",
        r"\bwe can verify\b",
        r"\bVerifying\b",
        r"\bChecking\b",
        r"\bTo make sure\b",
        r"\bLet's confirm\b",
        r"\bDouble checking\b",
        r"\bSanity check\b",
        r"\bValidating\b",
        r"\bLet's test\b",
        r"\bTo ensure\b",
        r"\bCross-check\b",
        r"\bReviewing\b",
    ]

    for pattern in verification_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False
