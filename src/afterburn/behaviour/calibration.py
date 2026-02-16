"""Confidence calibration comparison with token probability analysis."""

from __future__ import annotations

import numpy as np

from afterburn.types import CalibrationAnalysis, CalibrationBin, PromptResult

NUM_BINS = 10


def analyze_calibration(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
) -> CalibrationAnalysis:
    """Compare confidence calibration between base and trained models.

    If token probabilities are available (Phase 2), computes proper ECE
    using binned calibration. Otherwise, falls back to hedging-based
    heuristic estimation.
    """
    base_has_probs = any(r.top_token_probs for r in base_results)
    trained_has_probs = any(r.top_token_probs for r in trained_results)

    if base_has_probs and trained_has_probs:
        return _calibration_from_token_probs(base_results, trained_results)

    # No token probabilities available — return empty calibration
    return CalibrationAnalysis(has_token_probs=False)


def _calibration_from_token_probs(
    base_results: list[PromptResult],
    trained_results: list[PromptResult],
) -> CalibrationAnalysis:
    """Compute calibration using actual token probabilities.

    For each output, the model's confidence is the mean of the max
    token probability at each generation step. Accuracy is determined
    by whether the expected answer appears in the output.
    """
    base_confs, base_accs = _extract_confidence_accuracy(base_results)
    trained_confs, trained_accs = _extract_confidence_accuracy(trained_results)

    base_bins = _compute_calibration_bins(base_confs, base_accs)
    trained_bins = _compute_calibration_bins(trained_confs, trained_accs)

    base_ece = _compute_ece(base_bins, len(base_confs))
    trained_ece = _compute_ece(trained_bins, len(trained_confs))

    base_overconf = _overconfidence_rate(base_bins)
    trained_overconf = _overconfidence_rate(trained_bins)

    return CalibrationAnalysis(
        base_ece=base_ece,
        trained_ece=trained_ece,
        calibration_change=trained_ece - base_ece,
        base_bins=base_bins,
        trained_bins=trained_bins,
        base_overconfidence_rate=base_overconf,
        trained_overconfidence_rate=trained_overconf,
        has_token_probs=True,
    )


def _extract_confidence_accuracy(
    results: list[PromptResult],
) -> tuple[list[float], list[float]]:
    """Extract per-sample confidence and accuracy from results."""
    confidences = []
    accuracies = []

    for r in results:
        if not r.top_token_probs:
            continue

        # Confidence: mean of max token probability at each step
        max_probs = []
        for step_probs in r.top_token_probs:
            if step_probs:
                max_probs.append(max(step_probs.values()))
        if not max_probs:
            continue

        confidence = float(np.mean(max_probs))
        confidences.append(confidence)

        # Accuracy: check if expected answer is in output
        if r.expected_answer:
            is_correct = r.expected_answer.lower().strip() in r.output_text.lower()
            accuracies.append(1.0 if is_correct else 0.0)
        else:
            # Without expected answer, use a neutral value
            accuracies.append(0.5)

    return confidences, accuracies


def _compute_calibration_bins(
    confidences: list[float],
    accuracies: list[float],
) -> list[CalibrationBin]:
    """Bin confidences and compute per-bin accuracy."""
    if not confidences:
        return []

    bins: list[CalibrationBin] = []
    bin_width = 1.0 / NUM_BINS

    for i in range(NUM_BINS):
        lower = i * bin_width
        upper = (i + 1) * bin_width

        indices = [
            j for j, c in enumerate(confidences)
            if lower <= c < upper or (i == NUM_BINS - 1 and c == upper)
        ]

        if not indices:
            continue

        bin_confs = [confidences[j] for j in indices]
        bin_accs = [accuracies[j] for j in indices]

        bins.append(CalibrationBin(
            bin_lower=lower,
            bin_upper=upper,
            avg_confidence=float(np.mean(bin_confs)),
            avg_accuracy=float(np.mean(bin_accs)),
            count=len(indices),
        ))

    return bins


def _compute_ece(bins: list[CalibrationBin], total_samples: int) -> float:
    """Compute Expected Calibration Error from bins."""
    if not bins or total_samples == 0:
        return 0.0

    ece = 0.0
    for b in bins:
        weight = b.count / total_samples
        ece += weight * abs(b.avg_accuracy - b.avg_confidence)

    return ece


def _overconfidence_rate(bins: list[CalibrationBin]) -> float:
    """Fraction of bins where confidence exceeds accuracy."""
    if not bins:
        return 0.0
    overconf = sum(1 for b in bins if b.avg_confidence > b.avg_accuracy + 0.05)
    return overconf / len(bins)


