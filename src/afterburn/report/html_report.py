"""HTML report generation using Jinja2 and Plotly."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from afterburn.exceptions import ReportGenerationError
from afterburn.report.visualizations import (
    create_attention_head_chart,
    create_layer_heatmap,
    create_length_distribution_chart,
    create_reward_hack_breakdown,
    create_risk_score_gauge,
    create_strategy_shift_chart,
)

# Phase 2 visualizations - import with fallback for compatibility
try:
    from afterburn.report.visualizations import (
        create_calibration_curve,
        create_cot_depth_chart,
        create_embedding_drift_chart,
        create_format_pattern_radar,
        create_per_category_length_chart,
        create_sycophancy_chart,
    )
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False
from afterburn.types import DiagnosticReport
from afterburn.version import __version__

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


class HTMLReport:
    """Generates interactive HTML diagnostic reports."""

    def __init__(self, report: DiagnosticReport):
        self.report = report

    def generate(self, output_path: Path) -> Path:
        """Generate HTML report file."""
        try:
            env = Environment(
                loader=FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=True,
            )
            template = env.get_template("report.html.j2")

            # Load CSS
            css_path = TEMPLATE_DIR / "styles.css"
            styles = css_path.read_text() if css_path.exists() else ""

            # Generate charts
            context = self._build_context(styles)

            html = template.render(**context)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)

            logger.info("HTML report saved to %s", output_path)
            return output_path

        except Exception as e:
            raise ReportGenerationError(f"Failed to generate HTML report: {e}") from e

    def _build_context(self, styles: str) -> dict[str, object]:
        """Build the template context with all charts and data."""
        context = {
            "report": self.report,
            "styles": styles,
            "version": __version__,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer_heatmap": "",
            "attention_chart": "",
            "length_chart": "",
            "strategy_chart": "",
            "risk_gauge": "",
            "hack_breakdown": "",
            # Phase 2 chart keys
            "calibration_chart": "",
            "cot_chart": "",
            "category_length_chart": "",
            "format_radar": "",
            "embedding_chart": "",
            "sycophancy_chart": "",
        }

        # Weight diff charts
        if self.report.weight_diff:
            context["layer_heatmap"] = create_layer_heatmap(
                self.report.weight_diff.layer_diffs
            )
            context["attention_chart"] = create_attention_head_chart(
                self.report.weight_diff.attention_heads
            )

            # Phase 2: Embedding drift chart
            if PHASE2_AVAILABLE and self.report.weight_diff.embedding_drift:
                context["embedding_chart"] = create_embedding_drift_chart(
                    self.report.weight_diff.embedding_drift
                )

        # Behaviour charts
        if self.report.behaviour:
            context["length_chart"] = create_length_distribution_chart(
                self.report.behaviour.length_analysis
            )
            context["strategy_chart"] = create_strategy_shift_chart(
                self.report.behaviour.strategy_analysis
            )

            # Phase 2: Additional behaviour charts
            if PHASE2_AVAILABLE:
                # Calibration chart
                if (self.report.behaviour.calibration and
                    self.report.behaviour.calibration.base_bins):
                    context["calibration_chart"] = create_calibration_curve(
                        self.report.behaviour.calibration.base_bins,
                        self.report.behaviour.calibration.trained_bins
                    )

                # Chain-of-thought depth chart
                if self.report.behaviour.cot_analysis:
                    context["cot_chart"] = create_cot_depth_chart(
                        self.report.behaviour.cot_analysis
                    )

                # Per-category length chart
                if self.report.behaviour.length_analysis.per_category:
                    context["category_length_chart"] = create_per_category_length_chart(
                        self.report.behaviour.length_analysis.per_category
                    )

                # Format pattern radar
                if self.report.behaviour.format_analysis.patterns_detected:
                    context["format_radar"] = create_format_pattern_radar(
                        self.report.behaviour.format_analysis.patterns_detected
                    )

        # Reward hack charts
        if self.report.reward_hack:
            rh = self.report.reward_hack
            context["risk_gauge"] = create_risk_score_gauge(
                rh.composite_score, rh.risk_level
            )
            context["hack_breakdown"] = create_reward_hack_breakdown(
                rh.length_bias.score,
                rh.format_gaming.score,
                rh.strategy_collapse.score,
                rh.sycophancy.score,
            )

            # Phase 2: Sycophancy chart
            if PHASE2_AVAILABLE and rh.sycophancy:
                context["sycophancy_chart"] = create_sycophancy_chart(
                    rh.sycophancy
                )

        return context
