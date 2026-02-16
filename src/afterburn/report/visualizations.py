"""Plotly chart builders for diagnostic reports."""

from __future__ import annotations

import json

from afterburn.types import (
    AttentionHeadScore,
    CalibrationBin,
    ChainOfThoughtAnalysis,
    EmbeddingDrift,
    LayerDiff,
    LengthAnalysis,
    RiskLevel,
    StrategyShiftAnalysis,
    SycophancyResult,
)


def create_layer_heatmap(layer_diffs: list[LayerDiff]) -> str:
    """Create a Plotly heatmap of per-layer weight changes.

    Returns HTML/JS string for embedding in the report.
    """
    if not layer_diffs:
        return "<p>No layer diff data available.</p>"

    # Filter to numbered layers only for the heatmap
    numbered = [d for d in layer_diffs if d.layer_name.startswith("layer_")]
    if not numbered:
        return "<p>No transformer layer data available.</p>"

    # Sort by index
    numbered.sort(key=lambda d: d.layer_index)

    labels = [d.layer_name for d in numbered]
    metrics = ["L2 Norm", "Cosine Sim", "Frobenius", "Rel. Change"]

    z = [
        [d.l2_norm for d in numbered],
        [d.cosine_similarity for d in numbered],
        [d.frobenius_norm for d in numbered],
        [d.relative_change for d in numbered],
    ]

    data = {
        "data": [
            {
                "type": "heatmap",
                "z": z,
                "x": labels,
                "y": metrics,
                "colorscale": "YlOrRd",
                "hoverongaps": False,
            }
        ],
        "layout": {
            "title": "Per-Layer Weight Difference",
            "xaxis": {"title": "Layer", "tickangle": -45},
            "yaxis": {"title": "Metric"},
            "height": 350,
            "margin": {"b": 100},
        },
    }

    div_id = "layer-heatmap"
    return _plotly_div(div_id, data)


def create_attention_head_chart(heads: list[AttentionHeadScore]) -> str:
    """Create attention head importance delta chart."""
    if not heads:
        return "<p>No attention head data available.</p>"

    # Group by layer and show top deltas
    sorted_heads = sorted(heads, key=lambda h: abs(h.importance_delta), reverse=True)[:30]

    labels = [f"L{h.layer_index}H{h.head_index}" for h in sorted_heads]
    deltas = [h.importance_delta for h in sorted_heads]
    colors = ["#d32f2f" if d > 0 else "#1976d2" for d in deltas]

    data = {
        "data": [
            {
                "type": "bar",
                "x": labels,
                "y": deltas,
                "marker": {"color": colors},
                "name": "Importance Delta",
            }
        ],
        "layout": {
            "title": "Top 30 Most Changed Attention Heads",
            "xaxis": {"title": "Layer.Head", "tickangle": -45},
            "yaxis": {"title": "Importance Delta"},
            "height": 350,
            "margin": {"b": 100},
        },
    }

    return _plotly_div("attention-heads", data)


def create_length_distribution_chart(length_analysis: LengthAnalysis) -> str:
    """Create side-by-side box plot for output lengths."""
    data = {
        "data": [
            {
                "type": "bar",
                "x": ["Base Model", "Trained Model"],
                "y": [length_analysis.base_mean, length_analysis.trained_mean],
                "error_y": {
                    "type": "data",
                    "array": [length_analysis.base_std, length_analysis.trained_std],
                    "visible": True,
                },
                "marker": {"color": ["#1976d2", "#d32f2f"]},
                "name": "Mean Length",
            }
        ],
        "layout": {
            "title": f"Output Length Comparison (Cohen's d = {length_analysis.cohens_d:.2f})",
            "yaxis": {"title": "Tokens"},
            "height": 300,
            "showlegend": False,
        },
    }

    return _plotly_div("length-dist", data)


def create_strategy_shift_chart(analysis: StrategyShiftAnalysis) -> str:
    """Create grouped bar chart for strategy distributions."""
    if not analysis.base_distribution and not analysis.trained_distribution:
        return "<p>No strategy shift data available.</p>"

    all_strategies = sorted(
        set(analysis.base_distribution.keys()) | set(analysis.trained_distribution.keys())
    )

    base_values = [analysis.base_distribution.get(s, 0.0) for s in all_strategies]
    trained_values = [analysis.trained_distribution.get(s, 0.0) for s in all_strategies]

    # Clean up strategy names for display
    display_names = [s.replace("_", " ").title() for s in all_strategies]

    data = {
        "data": [
            {
                "type": "bar",
                "x": display_names,
                "y": [v * 100 for v in base_values],
                "name": "Base Model",
                "marker": {"color": "#1976d2"},
            },
            {
                "type": "bar",
                "x": display_names,
                "y": [v * 100 for v in trained_values],
                "name": "Trained Model",
                "marker": {"color": "#d32f2f"},
            },
        ],
        "layout": {
            "title": "Reasoning Strategy Distribution",
            "xaxis": {"title": "Strategy"},
            "yaxis": {"title": "Percentage (%)"},
            "barmode": "group",
            "height": 350,
        },
    }

    return _plotly_div("strategy-shift", data)


def create_risk_score_gauge(score: float, risk_level: RiskLevel) -> str:
    """Create a gauge chart for the composite risk score."""
    color_map = {
        RiskLevel.LOW: "#4caf50",
        RiskLevel.MODERATE: "#ff9800",
        RiskLevel.HIGH: "#f44336",
        RiskLevel.CRITICAL: "#b71c1c",
    }

    data = {
        "data": [
            {
                "type": "indicator",
                "mode": "gauge+number",
                "value": score,
                "title": {"text": "Reward Hacking Risk"},
                "gauge": {
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color_map.get(risk_level, "#666")},
                    "steps": [
                        {"range": [0, 25], "color": "#e8f5e9"},
                        {"range": [25, 50], "color": "#fff3e0"},
                        {"range": [50, 75], "color": "#ffebee"},
                        {"range": [75, 100], "color": "#fce4ec"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            }
        ],
        "layout": {
            "height": 250,
            "margin": {"t": 50, "b": 20, "l": 30, "r": 30},
        },
    }

    return _plotly_div("risk-gauge", data)


def create_reward_hack_breakdown(
    length_score: float,
    format_score: float,
    collapse_score: float,
    sycophancy_score: float,
) -> str:
    """Create a horizontal bar chart of individual risk scores."""
    labels = ["Length Bias", "Format Gaming", "Strategy Collapse", "Sycophancy"]
    scores = [length_score, format_score, collapse_score, sycophancy_score]
    colors = [
        "#f44336" if s > 50 else "#ff9800" if s > 25 else "#4caf50" for s in scores
    ]

    data = {
        "data": [
            {
                "type": "bar",
                "y": labels,
                "x": scores,
                "orientation": "h",
                "marker": {"color": colors},
            }
        ],
        "layout": {
            "title": "Risk Score Breakdown",
            "xaxis": {"title": "Score (0-100)", "range": [0, 100]},
            "height": 250,
            "margin": {"l": 140},
        },
    }

    return _plotly_div("hack-breakdown", data)


def create_calibration_curve(
    base_bins: list[CalibrationBin], trained_bins: list[CalibrationBin]
) -> str:
    """Create a reliability diagram showing calibration bins for base vs trained.

    X-axis: avg_confidence, Y-axis: avg_accuracy.
    Includes perfect calibration diagonal line.
    """
    if not base_bins and not trained_bins:
        return "<p>No calibration data available.</p>"

    # Extract data for base model
    base_x = [b.avg_confidence for b in base_bins] if base_bins else []
    base_y = [b.avg_accuracy for b in base_bins] if base_bins else []

    # Extract data for trained model
    trained_x = [b.avg_confidence for b in trained_bins] if trained_bins else []
    trained_y = [b.avg_accuracy for b in trained_bins] if trained_bins else []

    # Perfect calibration line (diagonal)
    perfect_line = [0, 1]

    traces = [
        {
            "type": "scatter",
            "x": perfect_line,
            "y": perfect_line,
            "mode": "lines",
            "name": "Perfect Calibration",
            "line": {"color": "#999", "dash": "dash", "width": 2},
        }
    ]

    if base_x:
        traces.append({
            "type": "scatter",
            "x": base_x,
            "y": base_y,
            "mode": "markers+lines",
            "name": "Base Model",
            "marker": {"color": "#1976d2", "size": 8},
            "line": {"color": "#1976d2"},
        })

    if trained_x:
        traces.append({
            "type": "scatter",
            "x": trained_x,
            "y": trained_y,
            "mode": "markers+lines",
            "name": "Trained Model",
            "marker": {"color": "#d32f2f", "size": 8},
            "line": {"color": "#d32f2f"},
        })

    data = {
        "data": traces,
        "layout": {
            "title": "Calibration Curve (Reliability Diagram)",
            "xaxis": {"title": "Average Confidence", "range": [0, 1]},
            "yaxis": {"title": "Average Accuracy", "range": [0, 1]},
            "height": 400,
            "showlegend": True,
        },
    }

    return _plotly_div("calibration-curve", data)


def create_cot_depth_chart(cot_analysis: ChainOfThoughtAnalysis) -> str:
    """Create grouped bar chart comparing base vs trained CoT metrics.

    Compares: avg_steps, avg_depth, self_correction_rate, verification_rate.
    """
    if not cot_analysis:
        return "<p>No chain-of-thought analysis data available.</p>"

    metrics = ["Avg Steps", "Avg Depth", "Self-Correction Rate", "Verification Rate"]

    # Convert rates to percentages for better visualization
    base_values = [
        cot_analysis.base_avg_steps,
        cot_analysis.base_avg_depth,
        cot_analysis.base_self_correction_rate * 100,
        cot_analysis.base_verification_rate * 100,
    ]

    trained_values = [
        cot_analysis.trained_avg_steps,
        cot_analysis.trained_avg_depth,
        cot_analysis.trained_self_correction_rate * 100,
        cot_analysis.trained_verification_rate * 100,
    ]

    data = {
        "data": [
            {
                "type": "bar",
                "x": metrics,
                "y": base_values,
                "name": "Base Model",
                "marker": {"color": "#1976d2"},
            },
            {
                "type": "bar",
                "x": metrics,
                "y": trained_values,
                "name": "Trained Model",
                "marker": {"color": "#d32f2f"},
            },
        ],
        "layout": {
            "title": "Chain-of-Thought Depth Analysis",
            "xaxis": {"title": "Metric"},
            "yaxis": {"title": "Value (rates in %)"},
            "barmode": "group",
            "height": 350,
        },
    }

    return _plotly_div("cot-depth", data)


def create_per_category_length_chart(
    per_category: dict[str, dict[str, float]]
) -> str:
    """Create grouped bar chart showing base_mean vs trained_mean per category.

    Takes dict[category, dict[str, float]] where inner dict has base_mean, trained_mean.
    """
    if not per_category:
        return "<p>No per-category length data available.</p>"

    categories = sorted(per_category.keys())
    base_means = [per_category[cat].get("base_mean", 0.0) for cat in categories]
    trained_means = [per_category[cat].get("trained_mean", 0.0) for cat in categories]

    # Clean up category names for display
    display_names = [cat.replace("_", " ").title() for cat in categories]

    data = {
        "data": [
            {
                "type": "bar",
                "x": display_names,
                "y": base_means,
                "name": "Base Model",
                "marker": {"color": "#1976d2"},
            },
            {
                "type": "bar",
                "x": display_names,
                "y": trained_means,
                "name": "Trained Model",
                "marker": {"color": "#d32f2f"},
            },
        ],
        "layout": {
            "title": "Per-Category Length Analysis",
            "xaxis": {"title": "Category", "tickangle": -45},
            "yaxis": {"title": "Mean Output Length (tokens)"},
            "barmode": "group",
            "height": 350,
            "margin": {"b": 100},
        },
    }

    return _plotly_div("per-category-length", data)


def create_format_pattern_radar(
    patterns_detected: dict[str, dict[str, float]]
) -> str:
    """Create radar/polar chart showing base_rate vs trained_rate for each format pattern.

    Takes dict[pattern_name, dict[str, float]] where inner dict has base_rate, trained_rate.
    """
    if not patterns_detected:
        return "<p>No format pattern data available.</p>"

    patterns = sorted(patterns_detected.keys())
    base_rates = [patterns_detected[p].get("base_rate", 0.0) * 100 for p in patterns]
    trained_rates = [patterns_detected[p].get("trained_rate", 0.0) * 100 for p in patterns]

    # Close the polygon by repeating first value
    base_rates_closed = base_rates + [base_rates[0]] if base_rates else []
    trained_rates_closed = trained_rates + [trained_rates[0]] if trained_rates else []

    # Clean up pattern names for display
    display_names = [p.replace("_", " ").title() for p in patterns]
    display_names_closed = display_names + [display_names[0]] if display_names else []

    data = {
        "data": [
            {
                "type": "scatterpolar",
                "r": base_rates_closed,
                "theta": display_names_closed,
                "fill": "toself",
                "name": "Base Model",
                "marker": {"color": "#1976d2"},
                "line": {"color": "#1976d2"},
            },
            {
                "type": "scatterpolar",
                "r": trained_rates_closed,
                "theta": display_names_closed,
                "fill": "toself",
                "name": "Trained Model",
                "marker": {"color": "#d32f2f"},
                "line": {"color": "#d32f2f"},
            },
        ],
        "layout": {
            "title": "Format Pattern Detection Rates",
            "polar": {
                "radialaxis": {
                    "visible": True,
                    "range": [0, 100],
                    "ticksuffix": "%",
                }
            },
            "height": 400,
            "showlegend": True,
        },
    }

    return _plotly_div("format-pattern-radar", data)


def create_embedding_drift_chart(embedding_drift: EmbeddingDrift) -> str:
    """Create bar chart showing input/output embedding L2 and cosine metrics."""
    if not embedding_drift:
        return "<p>No embedding drift data available.</p>"

    metrics = ["Input L2", "Input Cosine", "Output L2", "Output Cosine"]
    values = [
        embedding_drift.input_embedding_l2,
        embedding_drift.input_embedding_cosine,
        embedding_drift.output_embedding_l2 or 0.0,
        embedding_drift.output_embedding_cosine or 0.0,
    ]

    # Use different colors for L2 (distance, lower is better)
    # vs Cosine (similarity, higher is better)
    colors = ["#f44336", "#4caf50", "#f44336", "#4caf50"]

    data = {
        "data": [
            {
                "type": "bar",
                "x": metrics,
                "y": values,
                "marker": {"color": colors},
            }
        ],
        "layout": {
            "title": "Embedding Drift Analysis",
            "xaxis": {"title": "Metric"},
            "yaxis": {"title": "Value"},
            "height": 300,
            "showlegend": False,
        },
    }

    return _plotly_div("embedding-drift", data)


def create_sycophancy_chart(sycophancy: SycophancyResult) -> str:
    """Create grouped bar chart showing agreement rate and pushback rate for base vs trained."""
    if not sycophancy:
        return "<p>No sycophancy analysis data available.</p>"

    metrics = ["Agreement Rate", "Pushback Rate"]

    base_values = [
        sycophancy.base_agreement_rate * 100,
        sycophancy.base_pushback_rate * 100,
    ]

    trained_values = [
        sycophancy.trained_agreement_rate * 100,
        sycophancy.trained_pushback_rate * 100,
    ]

    data = {
        "data": [
            {
                "type": "bar",
                "x": metrics,
                "y": base_values,
                "name": "Base Model",
                "marker": {"color": "#1976d2"},
            },
            {
                "type": "bar",
                "x": metrics,
                "y": trained_values,
                "name": "Trained Model",
                "marker": {"color": "#d32f2f"},
            },
        ],
        "layout": {
            "title": "Sycophancy Analysis",
            "xaxis": {"title": "Metric"},
            "yaxis": {"title": "Percentage (%)", "range": [0, 100]},
            "barmode": "group",
            "height": 300,
        },
    }

    return _plotly_div("sycophancy", data)


def _plotly_div(div_id: str, data: object) -> str:
    """Wrap Plotly data in an HTML div with inline JS."""
    json_str = json.dumps(data["data"])  # type: ignore[index]
    layout_str = json.dumps(data["layout"])  # type: ignore[index]
    return (
        f'<div id="{div_id}"></div>\n'
        f"<script>Plotly.newPlot('{div_id}', {json_str}, {layout_str}, "
        f"{{responsive: true}});</script>"
    )
