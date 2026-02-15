"""JSON structured report output."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from afterburn.exceptions import ReportGenerationError
from afterburn.types import DiagnosticReport
from afterburn.version import __version__

logger = logging.getLogger(__name__)


class JSONReport:
    """Generates JSON structured diagnostic reports."""

    def __init__(self, report: DiagnosticReport):
        self.report = report

    def generate(self, output_path: Path) -> Path:
        """Generate JSON report file."""
        try:
            data = self._build_json()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=_json_serializer)

            logger.info("JSON report saved to %s", output_path)
            return output_path

        except Exception as e:
            raise ReportGenerationError(f"Failed to generate JSON report: {e}") from e

    def _build_json(self) -> dict[str, Any]:
        """Build JSON-serializable dict from report."""
        r = self.report

        data: dict[str, Any] = {
            "afterburn_version": __version__,
            "generated_at": datetime.now().isoformat(),
            "model_pair": {
                "base_model": r.model_pair.base_model,
                "trained_model": r.model_pair.trained_model,
                "method": r.model_pair.method.value,
            },
            "summary": r.summary,
            "hack_score": r.hack_score,
            "recommendations": r.recommendations,
        }

        if r.weight_diff:
            wd = r.weight_diff
            data["weight_diff"] = {
                "total_param_count": wd.total_param_count,
                "changed_param_count": wd.changed_param_count,
                "change_concentration": wd.change_concentration,
                "top_changed_layers": [
                    {
                        "layer_name": l.layer_name,
                        "layer_index": l.layer_index,
                        "l2_norm": l.l2_norm,
                        "cosine_similarity": l.cosine_similarity,
                        "frobenius_norm": l.frobenius_norm,
                        "relative_change": l.relative_change,
                        "param_count": l.param_count,
                    }
                    for l in wd.top_changed_layers
                ],
                "layernorm_shifts": [
                    {
                        "layer_name": s.layer_name,
                        "gamma_shift": s.gamma_shift,
                        "beta_shift": s.beta_shift,
                        "total_shift": s.total_shift,
                        "is_significant": s.is_significant,
                    }
                    for s in wd.layernorm_shifts
                ],
                "embedding_drift": {
                    "input_l2": wd.embedding_drift.input_embedding_l2,
                    "input_cosine": wd.embedding_drift.input_embedding_cosine,
                    "output_l2": wd.embedding_drift.output_embedding_l2,
                    "output_cosine": wd.embedding_drift.output_embedding_cosine,
                },
            }

        if r.behaviour:
            bh = r.behaviour
            data["behaviour"] = {
                "summary": bh.summary,
                "length_analysis": {
                    "base_mean": bh.length_analysis.base_mean,
                    "trained_mean": bh.length_analysis.trained_mean,
                    "mean_diff": bh.length_analysis.mean_diff,
                    "cohens_d": bh.length_analysis.cohens_d,
                    "p_value": bh.length_analysis.p_value,
                    "is_significant": bh.length_analysis.is_significant,
                },
                "strategy_analysis": {
                    "base_distribution": bh.strategy_analysis.base_distribution,
                    "trained_distribution": bh.strategy_analysis.trained_distribution,
                    "dominant_shift": bh.strategy_analysis.dominant_shift,
                    "entropy_change": bh.strategy_analysis.entropy_change,
                },
                "format_analysis": {
                    "base_format_rate": bh.format_analysis.base_format_rate,
                    "trained_format_rate": bh.format_analysis.trained_format_rate,
                    "format_increase": bh.format_analysis.format_increase,
                    "patterns": bh.format_analysis.patterns_detected,
                },
                "cot_analysis": {
                    "base_avg_steps": bh.cot_analysis.base_avg_steps,
                    "trained_avg_steps": bh.cot_analysis.trained_avg_steps,
                    "step_count_change": bh.cot_analysis.step_count_change,
                },
                "prompt_count": len(bh.base_results),
            }

        if r.reward_hack:
            rh = r.reward_hack
            data["reward_hack"] = {
                "composite_score": rh.composite_score,
                "risk_level": rh.risk_level.value,
                "flags": rh.flags,
                "length_bias": {
                    "score": rh.length_bias.score,
                    "cohens_d": rh.length_bias.cohens_d,
                    "p_value": rh.length_bias.p_value,
                    "mean_length_ratio": rh.length_bias.mean_length_ratio,
                    "is_flagged": rh.length_bias.is_flagged,
                },
                "format_gaming": {
                    "score": rh.format_gaming.score,
                    "is_flagged": rh.format_gaming.is_flagged,
                    "patterns": rh.format_gaming.patterns,
                },
                "strategy_collapse": {
                    "score": rh.strategy_collapse.score,
                    "base_entropy": rh.strategy_collapse.base_entropy,
                    "trained_entropy": rh.strategy_collapse.trained_entropy,
                    "entropy_drop": rh.strategy_collapse.entropy_drop,
                    "dominant_strategy": rh.strategy_collapse.dominant_strategy,
                    "is_flagged": rh.strategy_collapse.is_flagged,
                },
                "sycophancy": {
                    "score": rh.sycophancy.score,
                    "base_agreement_rate": rh.sycophancy.base_agreement_rate,
                    "trained_agreement_rate": rh.sycophancy.trained_agreement_rate,
                    "agreement_increase": rh.sycophancy.agreement_increase,
                    "is_flagged": rh.sycophancy.is_flagged,
                },
            }

        return data


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default."""
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
