"""Tests for Phase 2 Plotly visualization functions."""


from afterburn.report.visualizations import (
    create_calibration_curve,
    create_cot_depth_chart,
    create_embedding_drift_chart,
    create_format_pattern_radar,
    create_per_category_length_chart,
    create_sycophancy_chart,
)
from afterburn.types import (
    CalibrationBin,
    ChainOfThoughtAnalysis,
    EmbeddingDrift,
    SycophancyResult,
)


class TestCalibrationCurve:
    def test_renders_with_bins(self):
        base_bins = [
            CalibrationBin(0.0, 0.5, 0.3, 0.35, 10),
            CalibrationBin(0.5, 1.0, 0.8, 0.75, 10),
        ]
        trained_bins = [
            CalibrationBin(0.0, 0.5, 0.25, 0.4, 10),
            CalibrationBin(0.5, 1.0, 0.85, 0.6, 10),
        ]
        html = create_calibration_curve(base_bins, trained_bins)
        assert "calibration-curve" in html
        assert "Plotly.newPlot" in html
        assert "Perfect Calibration" in html

    def test_base_only(self):
        base_bins = [CalibrationBin(0.0, 0.5, 0.3, 0.35, 10)]
        html = create_calibration_curve(base_bins, [])
        assert "calibration-curve" in html

    def test_empty_returns_placeholder(self):
        html = create_calibration_curve([], [])
        assert "No calibration data" in html


class TestCotDepthChart:
    def test_renders_chart(self):
        cot = ChainOfThoughtAnalysis(
            base_avg_steps=3.0,
            trained_avg_steps=5.0,
            base_avg_depth=2.0,
            trained_avg_depth=4.0,
            step_count_change=2.0,
            depth_change=2.0,
            base_self_correction_rate=0.1,
            trained_self_correction_rate=0.3,
            base_verification_rate=0.2,
            trained_verification_rate=0.5,
        )
        html = create_cot_depth_chart(cot)
        assert "cot-depth" in html
        assert "Plotly.newPlot" in html


class TestPerCategoryLengthChart:
    def test_renders_with_data(self):
        per_category = {
            "math": {"base_mean": 50.0, "trained_mean": 100.0},
            "code": {"base_mean": 60.0, "trained_mean": 80.0},
        }
        html = create_per_category_length_chart(per_category)
        assert "per-category-length" in html
        assert "Plotly.newPlot" in html

    def test_empty_returns_placeholder(self):
        html = create_per_category_length_chart({})
        assert "No per-category" in html


class TestFormatPatternRadar:
    def test_renders_with_patterns(self):
        patterns = {
            "code_block": {"base_rate": 0.1, "trained_rate": 0.5},
            "boxed_answer": {"base_rate": 0.0, "trained_rate": 0.3},
            "numbered_steps": {"base_rate": 0.4, "trained_rate": 0.2},
        }
        html = create_format_pattern_radar(patterns)
        assert "format-pattern-radar" in html
        assert "Plotly.newPlot" in html
        assert "scatterpolar" in html

    def test_empty_returns_placeholder(self):
        html = create_format_pattern_radar({})
        assert "No format pattern" in html


class TestEmbeddingDriftChart:
    def test_renders_with_data(self):
        drift = EmbeddingDrift(
            input_embedding_l2=0.5,
            input_embedding_cosine=0.95,
            output_embedding_l2=0.3,
            output_embedding_cosine=0.98,
            top_drifted_tokens=[(42, 0.1), (100, 0.05)],
        )
        html = create_embedding_drift_chart(drift)
        assert "embedding-drift" in html
        assert "Plotly.newPlot" in html

    def test_none_output_embedding(self):
        drift = EmbeddingDrift(
            input_embedding_l2=0.5,
            input_embedding_cosine=0.95,
            output_embedding_l2=None,
            output_embedding_cosine=None,
            top_drifted_tokens=[],
        )
        html = create_embedding_drift_chart(drift)
        assert "embedding-drift" in html


class TestSycophancyChart:
    def test_renders_with_data(self):
        syco = SycophancyResult(
            score=45.0,
            base_agreement_rate=0.2,
            trained_agreement_rate=0.6,
            agreement_increase=0.4,
            is_flagged=True,
            base_pushback_rate=0.7,
            trained_pushback_rate=0.3,
            persuasion_resistance_drop=0.4,
        )
        html = create_sycophancy_chart(syco)
        assert "sycophancy" in html
        assert "Plotly.newPlot" in html
