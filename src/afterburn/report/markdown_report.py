"""Markdown report generation."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from afterburn.exceptions import ReportGenerationError
from afterburn.types import DiagnosticReport
from afterburn.version import __version__

logger = logging.getLogger(__name__)


class MarkdownReport:
    """Generates Markdown diagnostic reports."""

    def __init__(self, report: DiagnosticReport):
        self.report = report

    def generate(self, output_path: Path) -> Path:
        """Generate Markdown report file."""
        try:
            md = self._build_markdown()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md)
            logger.info("Markdown report saved to %s", output_path)
            return output_path
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate Markdown report: {e}") from e

    def _build_markdown(self) -> str:
        r = self.report
        lines: list[str] = []

        lines.append("# Afterburn Diagnostic Report\n")
        lines.append(f"**Base Model:** `{r.model_pair.base_model}`  ")
        lines.append(f"**Trained Model:** `{r.model_pair.trained_model}`  ")
        lines.append(f"**Method:** {r.model_pair.method.value.upper()}  ")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        if r.reward_hack:
            lines.append(
                f"**Risk Level:** {r.reward_hack.risk_level.value.upper()} "
                f"({r.hack_score:.0f}/100)\n"
            )
        lines.append(f"{r.summary}\n")

        # Weight Diff
        if r.weight_diff:
            wd = r.weight_diff
            lines.append("## Weight Diff Analysis\n")
            lines.append(f"- **Total Parameters:** {wd.total_param_count:,}")
            lines.append(f"- **Changed Parameters:** {wd.changed_param_count:,}")
            lines.append(f"- **Change Concentration (Top 5):** {wd.change_concentration:.1%}\n")

            lines.append("### Top Changed Layers\n")
            lines.append("| Layer | L2 Norm | Cosine Sim | Frobenius | Rel. Change | Params |")
            lines.append("|-------|---------|------------|-----------|-------------|--------|")
            for layer in wd.top_changed_layers:
                lines.append(
                    f"| `{layer.layer_name}` | {layer.l2_norm:.4f} | "
                    f"{layer.cosine_similarity:.6f} | {layer.frobenius_norm:.4f} | "
                    f"{layer.relative_change:.6f} | {layer.param_count:,} |"
                )
            lines.append("")

            # LayerNorm shifts
            significant_norms = [s for s in wd.layernorm_shifts if s.is_significant]
            if significant_norms:
                lines.append("### Significant LayerNorm Shifts\n")
                lines.append("| Layer | Gamma Shift | Beta Shift | Total Shift |")
                lines.append("|-------|-------------|------------|-------------|")
                for shift in significant_norms:
                    lines.append(
                        f"| `{shift.layer_name}` | {shift.gamma_shift:.6f} | "
                        f"{shift.beta_shift:.6f} | {shift.total_shift:.6f} |"
                    )
                lines.append("")

            # Embedding drift
            ed = wd.embedding_drift
            lines.append("### Embedding Drift\n")
            lines.append(f"- **Input Embedding L2:** {ed.input_embedding_l2:.4f}")
            lines.append(f"- **Input Embedding Cosine:** {ed.input_embedding_cosine:.6f}")
            if ed.output_embedding_l2 is not None:
                lines.append(f"- **Output Embedding L2:** {ed.output_embedding_l2:.4f}")
            lines.append("")

        # Behaviour
        if r.behaviour:
            bh = r.behaviour
            lines.append("## Behavioural Analysis\n")
            lines.append(f"{bh.summary}\n")

            la = bh.length_analysis
            lines.append("### Output Length\n")
            lines.append(f"- **Base Mean:** {la.base_mean:.1f} tokens")
            lines.append(f"- **Trained Mean:** {la.trained_mean:.1f} tokens")
            lines.append(f"- **Cohen's d:** {la.cohens_d:.2f}")
            lines.append(f"- **p-value:** {la.p_value:.4f}")
            lines.append(f"- **Significant:** {'Yes' if la.is_significant else 'No'}\n")

            if bh.strategy_analysis.dominant_shift:
                lines.append("### Strategy Shift\n")
                lines.append(f"- **Dominant shift:** {bh.strategy_analysis.dominant_shift}")
                lines.append(
                    f"- **Entropy:** {bh.strategy_analysis.base_entropy:.2f} → "
                    f"{bh.strategy_analysis.trained_entropy:.2f}\n"
                )

            lines.append("### Chain-of-Thought\n")
            lines.append(f"- **Base Avg Steps:** {bh.cot_analysis.base_avg_steps:.1f}")
            lines.append(f"- **Trained Avg Steps:** {bh.cot_analysis.trained_avg_steps:.1f}")
            lines.append(f"- **Step Count Change:** {bh.cot_analysis.step_count_change:+.1f}\n")

        # Reward Hacking
        if r.reward_hack:
            rh = r.reward_hack
            lines.append("## Reward Hacking Assessment\n")
            lines.append(
                f"**Composite Score:** {rh.composite_score:.0f}/100 "
                f"({rh.risk_level.value.upper()})\n"
            )

            lines.append("| Check | Score | Flagged | Detail |")
            lines.append("|-------|-------|---------|--------|")
            checks = [
                (
                    "Length Bias",
                    rh.length_bias.score,
                    rh.length_bias.is_flagged,
                    rh.length_bias.detail,
                ),
                (
                    "Format Gaming",
                    rh.format_gaming.score,
                    rh.format_gaming.is_flagged,
                    rh.format_gaming.detail,
                ),
                (
                    "Strategy Collapse",
                    rh.strategy_collapse.score,
                    rh.strategy_collapse.is_flagged,
                    rh.strategy_collapse.detail,
                ),
                (
                    "Sycophancy",
                    rh.sycophancy.score,
                    rh.sycophancy.is_flagged,
                    rh.sycophancy.detail,
                ),
            ]
            for name, score, flagged, detail in checks:
                flag_str = "⚠ Yes" if flagged else "No"
                lines.append(f"| {name} | {score:.1f} | {flag_str} | {detail} |")
            lines.append("")

            if rh.flags:
                lines.append("### Flags\n")
                for flag in rh.flags:
                    lines.append(f"- ⚠ {flag}")
                lines.append("")

        # Recommendations
        if r.recommendations:
            lines.append("## Recommendations\n")
            for rec in r.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Generated by Afterburn v{__version__}*")

        return "\n".join(lines)
