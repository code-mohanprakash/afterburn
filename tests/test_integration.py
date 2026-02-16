"""End-to-end integration tests for Afterburn.

Tests the full pipeline data flow without requiring real model downloads.
Uses synthetic data to verify cross-module integration.
"""

import pytest
import torch

from afterburn.types import (
    BehaviourResult,
    ChainOfThoughtAnalysis,
    DiagnosticReport,
    FormatAnalysis,
    LengthAnalysis,
    ModelPair,
    PromptResult,
    RewardHackResult,
    StrategyShiftAnalysis,
)

# ─── Fixtures ────────────────────────────────────────────────────────


def _make_prompt_results(category: str, n: int, prefix: str = "base") -> list[PromptResult]:
    """Generate synthetic prompt results."""
    results = []
    for i in range(n):
        results.append(PromptResult(
            prompt_id=f"{prefix}_{category}_{i}",
            prompt_text=f"Test prompt {i} for {category}",
            category=category,
            output_text=f"This is a {'long ' * (20 if prefix == 'trained' else 10)}response to question {i}.",
            output_tokens=50 if prefix == "base" else 80,
            generation_time_ms=100.0,
            expected_answer=f"answer_{i}",
        ))
    return results


def _make_behaviour_result() -> BehaviourResult:
    """Create a synthetic BehaviourResult for testing."""
    base_results = (
        _make_prompt_results("math", 5, "base")
        + _make_prompt_results("code", 5, "base")
        + _make_prompt_results("safety", 5, "base")
    )
    trained_results = (
        _make_prompt_results("math", 5, "trained")
        + _make_prompt_results("code", 5, "trained")
        + _make_prompt_results("safety", 5, "trained")
    )

    return BehaviourResult(
        base_results=base_results,
        trained_results=trained_results,
        length_analysis=LengthAnalysis(
            base_mean=50.0, base_median=48.0, base_std=10.0,
            trained_mean=80.0, trained_median=78.0, trained_std=12.0,
            mean_diff=30.0, p_value=0.001, cohens_d=2.5,
            is_significant=True,
        ),
        format_analysis=FormatAnalysis(),
        strategy_analysis=StrategyShiftAnalysis(),
        cot_analysis=ChainOfThoughtAnalysis(),
    )


# ─── Integration Tests ───────────────────────────────────────────────


class TestRewardHackIntegration:
    """Test that RewardHackDetector works with BehaviourResult."""

    def test_detector_accepts_behaviour_result(self):
        from afterburn.reward_hack.detector import RewardHackDetector

        behaviour = _make_behaviour_result()
        detector = RewardHackDetector(behaviour)
        result = detector.run()

        assert isinstance(result, RewardHackResult)
        assert 0 <= result.composite_score <= 100
        assert result.risk_level is not None

    def test_detector_with_training_method(self):
        from afterburn.reward_hack.detector import RewardHackDetector
        from afterburn.types import TrainingMethod

        behaviour = _make_behaviour_result()
        detector = RewardHackDetector(behaviour, method=TrainingMethod.DPO)
        result = detector.run()

        assert isinstance(result, RewardHackResult)

    def test_all_sub_detectors_run(self):
        from afterburn.reward_hack.detector import RewardHackDetector

        behaviour = _make_behaviour_result()
        result = RewardHackDetector(behaviour).run()

        assert result.length_bias is not None
        assert result.format_gaming is not None
        assert result.strategy_collapse is not None
        assert result.sycophancy is not None

    def test_flags_are_strings(self):
        from afterburn.reward_hack.detector import RewardHackDetector

        behaviour = _make_behaviour_result()
        result = RewardHackDetector(behaviour).run()

        assert all(isinstance(f, str) for f in result.flags)


class TestReportIntegration:
    """Test report generation from DiagnosticReport."""

    def test_json_report_roundtrip(self, tmp_path):
        import json

        from afterburn.report.json_report import JSONReport

        report = DiagnosticReport(
            model_pair=ModelPair(base_model="test/base", trained_model="test/trained"),
            summary="Test summary",
            hack_score=42.0,
        )

        output_path = tmp_path / "report.json"
        JSONReport(report).generate(output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["summary"] == "Test summary"

    def test_markdown_report(self, tmp_path):
        from afterburn.report.markdown_report import MarkdownReport

        report = DiagnosticReport(
            model_pair=ModelPair(base_model="test/base", trained_model="test/trained"),
            summary="Test summary",
        )

        output_path = tmp_path / "report.md"
        MarkdownReport(report).generate(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "Afterburn" in content or "afterburn" in content.lower()

    def test_html_report(self, tmp_path):
        from afterburn.report.html_report import HTMLReport

        report = DiagnosticReport(
            model_pair=ModelPair(base_model="test/base", trained_model="test/trained"),
            summary="Test summary",
        )

        output_path = tmp_path / "report.html"
        HTMLReport(report).generate(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "<html" in content.lower()


class TestSummaryIntegration:
    """Test summary and recommendations generation."""

    def test_summary_with_behaviour_only(self):
        from afterburn.report.summary import generate_recommendations, generate_summary

        report = DiagnosticReport(
            model_pair=ModelPair(base_model="test/base", trained_model="test/trained"),
            behaviour=_make_behaviour_result(),
        )

        summary = generate_summary(report)
        assert isinstance(summary, str)
        assert len(summary) > 0

        recs = generate_recommendations(report)
        assert isinstance(recs, list)

    def test_summary_with_reward_hack(self):
        from afterburn.report.summary import generate_summary
        from afterburn.reward_hack.detector import RewardHackDetector

        behaviour = _make_behaviour_result()
        reward_hack = RewardHackDetector(behaviour).run()

        report = DiagnosticReport(
            model_pair=ModelPair(base_model="test/base", trained_model="test/trained"),
            behaviour=behaviour,
            reward_hack=reward_hack,
            hack_score=reward_hack.composite_score,
        )

        summary = generate_summary(report)
        assert isinstance(summary, str)


class TestCrossModuleDataFlow:
    """Verify data flows correctly between modules."""

    def test_behaviour_result_summary_property(self):
        behaviour = _make_behaviour_result()
        assert isinstance(behaviour.summary, str)

    def test_reward_hack_result_summary_property(self):
        from afterburn.reward_hack.detector import RewardHackDetector

        behaviour = _make_behaviour_result()
        result = RewardHackDetector(behaviour).run()
        assert isinstance(result.summary, str)
        assert "risk" in result.summary.lower() or "reward" in result.summary.lower()

    def test_diagnostic_report_save_detect_format(self):
        """Test that format detection works for different extensions."""
        from pathlib import Path

        from afterburn.types import ReportFormat, _detect_format

        assert _detect_format(Path("out.html")) == ReportFormat.HTML
        assert _detect_format(Path("out.json")) == ReportFormat.JSON
        assert _detect_format(Path("out.md")) == ReportFormat.MARKDOWN
        assert _detect_format(Path("out.pdf")) == ReportFormat.PDF
        assert _detect_format(Path("out.txt")) == ReportFormat.HTML  # default


class TestWeightDiffMetricsIntegration:
    """Test weight diff metric functions work together."""

    def test_svd_with_behavioral_vectors(self):
        from afterburn.weight_diff.metrics import compute_direction_coherence, svd_analysis

        base = torch.randn(64, 64)
        trained = base + 0.1 * torch.randn(64, 64)

        result = svd_analysis(base, trained, return_vectors=True)
        assert result is not None
        assert result.top_right_vectors is not None

        # Coherence with single layer should be 0
        coherence = compute_direction_coherence({"layer_0": result.top_right_vectors})
        assert coherence == 0.0

        # Add second layer
        result2 = svd_analysis(base, trained + 0.05 * torch.randn(64, 64), return_vectors=True)
        vectors = {
            "layer_0": result.top_right_vectors,
            "layer_1": result2.top_right_vectors,
        }
        coherence = compute_direction_coherence(vectors)
        assert 0.0 <= coherence <= 1.0

    def test_spectral_and_mp_on_same_matrix(self):
        from afterburn.weight_diff.spectral import marchenko_pastur_fit, spectral_analysis

        mat = torch.randn(128, 128)

        spectral = spectral_analysis(mat)
        mp = marchenko_pastur_fit(mat)

        assert spectral is not None
        assert mp is not None
        assert spectral.stable_rank > 0
        assert mp.bulk_fraction > 0


class TestLoRAAnalysis:
    """Test LoRA adapter analysis."""

    def test_analyze_lora_adapter(self, tmp_path):
        import json

        from safetensors.torch import save_file

        from afterburn.loading.lora_loader import load_lora_adapter
        from afterburn.weight_diff.lora_analysis import analyze_lora_adapter

        # Create fake adapter
        config = {
            "r": 8,
            "lora_alpha": 16.0,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        # Create fake weights
        weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 64),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 8),
            "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.randn(8, 64),
            "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight": torch.randn(64, 8),
        }
        save_file(weights, str(tmp_path / "adapter_model.safetensors"))

        # Load and analyze
        lora = load_lora_adapter(tmp_path)
        result = analyze_lora_adapter(lora)

        assert result["rank"] == 8
        assert result["alpha"] == 16.0
        assert result["scaling"] == 2.0
        assert result["total_params"] > 0
        assert len(result["per_layer_norms"]) > 0
        assert len(result["most_affected_layers"]) > 0

    def test_lora_missing_config_raises(self, tmp_path):
        from afterburn.exceptions import ModelLoadError
        from afterburn.loading.lora_loader import load_lora_adapter

        with pytest.raises(ModelLoadError):
            load_lora_adapter(tmp_path)


class TestEngineLoRAIntegration:
    """Test that WeightDiffEngine detects and analyzes LoRA adapters."""

    def test_engine_lora_detection_logic(self, tiny_model_pair, tmp_path):
        """Test that engine's LoRA detection logic works correctly."""
        import json
        from pathlib import Path

        from safetensors.torch import save_file

        from afterburn.loading.lora_loader import load_lora_adapter
        from afterburn.weight_diff.lora_analysis import analyze_lora_adapter

        # This test verifies the LoRA detection code path in engine.py
        # without running the full engine (which requires compatible models)

        base_dir, trained_dir = tiny_model_pair

        # Test 1: Detect that trained model has no adapter
        trained_path = Path(trained_dir)
        adapter_config = trained_path / "adapter_config.json"
        assert not adapter_config.exists()

        # Test 2: Create a model with LoRA adapter and verify detection
        lora_dir = tmp_path / "model_with_lora"
        lora_dir.mkdir()

        # Copy model config
        import shutil
        shutil.copy(Path(base_dir) / "config.json", lora_dir / "config.json")

        # Create adapter config
        config = {
            "r": 8,
            "lora_alpha": 16.0,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        (lora_dir / "adapter_config.json").write_text(json.dumps(config))

        # Create fake adapter weights
        weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 64),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 8),
        }
        save_file(weights, str(lora_dir / "adapter_model.safetensors"))

        # Verify adapter is detected
        adapter_config_path = lora_dir / "adapter_config.json"
        assert adapter_config_path.exists()

        # Simulate the engine's LoRA detection code
        lora_result = None
        if adapter_config_path.exists():
            try:
                lora_weights = load_lora_adapter(lora_dir)
                lora_result = analyze_lora_adapter(lora_weights)
            except Exception as e:
                pytest.fail(f"LoRA analysis failed: {e}")

        # Verify results
        assert lora_result is not None
        assert lora_result["rank"] == 8
        assert lora_result["alpha"] == 16.0
        assert lora_result["scaling"] == 2.0

    def test_engine_without_lora_adapter(self, tiny_model_pair):
        """Test that engine works normally when no LoRA adapter is present."""
        from afterburn.device import DeviceConfig
        from afterburn.weight_diff.engine import WeightDiffEngine

        base_dir, trained_dir = tiny_model_pair

        # Run engine with regular models
        model_pair = ModelPair(base_model=base_dir, trained_model=trained_dir)
        device_config = DeviceConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_memory_gb=1.0
        )
        engine = WeightDiffEngine(model_pair, device_config)

        result = engine.run()

        # Verify LoRA analysis is None
        assert result.lora_analysis is None
        # But other analysis should still work
        assert len(result.layer_diffs) > 0
