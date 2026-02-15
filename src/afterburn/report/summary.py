"""Executive summary auto-generation."""

from __future__ import annotations

from afterburn.types import DiagnosticReport, RiskLevel


def generate_summary(report: DiagnosticReport) -> str:
    """Generate a plain-English executive summary of the diagnostic report."""
    parts: list[str] = []

    parts.append(
        f"Diagnostic comparison of {report.model_pair.base_model} (base) "
        f"vs {report.model_pair.trained_model} (trained), "
        f"using {report.model_pair.method.value.upper()} training method."
    )

    # Weight diff summary
    if report.weight_diff:
        wd = report.weight_diff
        top = wd.top_changed_layers
        if top:
            most_changed = top[0]
            parts.append(
                f"Weight analysis shows the most significant changes in "
                f"{most_changed.layer_name} (relative change: {most_changed.relative_change:.4f}). "
                f"{wd.changed_param_count:,} of {wd.total_param_count:,} parameters "
                f"show measurable change."
            )

            concentration = wd.change_concentration
            if concentration > 0.5:
                parts.append(
                    f"Changes are highly concentrated — the top 5 layers account for "
                    f"{concentration:.0%} of total weight change."
                )

        # LayerNorm shifts
        significant_norms = [s for s in wd.layernorm_shifts if s.is_significant]
        if significant_norms:
            parts.append(
                f"{len(significant_norms)} layer(s) show significant LayerNorm shifts, "
                f"indicating meaningful activation distribution changes."
            )

    # Behaviour summary
    if report.behaviour:
        parts.append(report.behaviour.summary)

    # Reward hack summary
    if report.reward_hack:
        rh = report.reward_hack
        level = rh.risk_level

        if level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            parts.append(
                f"WARNING: Reward hacking risk is {level.value.upper()} "
                f"({rh.composite_score:.0f}/100). "
                f"Review the flagged items carefully before deploying this model."
            )
        elif level == RiskLevel.MODERATE:
            parts.append(
                f"Reward hacking risk is MODERATE ({rh.composite_score:.0f}/100). "
                f"Some patterns warrant investigation."
            )
        else:
            parts.append(
                f"Reward hacking risk is LOW ({rh.composite_score:.0f}/100). "
                f"No concerning patterns detected."
            )

    return " ".join(parts)


def generate_recommendations(report: DiagnosticReport) -> list[str]:
    """Generate actionable recommendations based on findings."""
    recommendations: list[str] = []

    if report.weight_diff:
        wd = report.weight_diff
        if wd.change_concentration > 0.7:
            recommendations.append(
                "Weight changes are highly concentrated in a few layers. "
                "Consider whether this indicates overfitting to specific patterns. "
                "Try reducing the learning rate or increasing regularization."
            )

        significant_norms = [s for s in wd.layernorm_shifts if s.is_significant]
        if len(significant_norms) > wd.total_param_count * 0.5 / max(len(wd.layer_diffs), 1):
            recommendations.append(
                "Many LayerNorm parameters show significant shifts. "
                "This can indicate aggressive training — consider reducing epochs or learning rate."
            )

    if report.behaviour:
        bh = report.behaviour
        if bh.length_analysis.is_significant and bh.length_analysis.cohens_d > 0.8:
            recommendations.append(
                "Output lengths changed dramatically. Check if your reward function "
                "has a length bias — consider adding a length penalty."
            )

        if bh.format_analysis.format_increase > 0.2:
            recommendations.append(
                "Significant increase in format pattern usage. If using a format-based "
                "verifier, ensure it validates correctness, not just format compliance."
            )

        if bh.strategy_analysis.entropy_change < -0.3:
            recommendations.append(
                "Reasoning strategy diversity decreased. The model may be converging "
                "on a single approach. Try diversifying training examples or using "
                "multiple verifiers."
            )

    if report.reward_hack:
        rh = report.reward_hack
        if rh.length_bias.is_flagged:
            recommendations.append(
                "Length bias detected. Add a length normalization term to your "
                "reward function or filter training samples by output length."
            )

        if rh.format_gaming.is_flagged:
            recommendations.append(
                "Format gaming detected. Strengthen your verifier to check "
                "semantic correctness, not just output format patterns."
            )

        if rh.strategy_collapse.is_flagged:
            recommendations.append(
                f"Strategy collapsed to primarily '{rh.strategy_collapse.dominant_strategy}'. "
                f"Consider training with diverse solution strategies or penalizing "
                f"strategy monotony."
            )

        if rh.sycophancy.is_flagged:
            recommendations.append(
                "Sycophancy increase detected. Review alignment data for over-representation "
                "of agreeable responses. Include more examples of respectful disagreement."
            )

    if not recommendations:
        recommendations.append(
            "No specific concerns identified. The training run appears well-calibrated. "
            "Consider running with more prompts for higher confidence in results."
        )

    return recommendations
