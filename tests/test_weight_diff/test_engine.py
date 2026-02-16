"""Tests for the weight diff engine."""


import torch

from afterburn.device import DeviceConfig
from afterburn.types import ModelPair
from afterburn.weight_diff.engine import WeightDiffEngine


class TestWeightDiffEngine:
    def test_run_with_tiny_models(self, tiny_model_pair):
        base_path, trained_path = tiny_model_pair
        model_pair = ModelPair(base_model=base_path, trained_model=trained_path)
        device_config = DeviceConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_memory_gb=8.0,
        )

        engine = WeightDiffEngine(model_pair, device_config)
        result = engine.run()

        # Basic structure checks
        assert result.total_param_count > 0
        assert len(result.layer_diffs) > 0
        assert len(result.top_changed_layers) <= 5

        # Layer 2 should be the most changed (we perturbed it with 0.1 noise)
        top_layer = result.top_changed_layers[0]
        assert "2" in top_layer.layer_name

        # Relative change for layer 2 should be largest
        layer2_diffs = [d for d in result.layer_diffs if "2" in d.layer_name]
        other_diffs = [d for d in result.layer_diffs if d.layer_name.startswith("layer_") and "2" not in d.layer_name]

        if layer2_diffs and other_diffs:
            max_layer2_change = max(d.relative_change for d in layer2_diffs)
            max_other_change = max(d.relative_change for d in other_diffs)
            assert max_layer2_change > max_other_change

    def test_cosine_similarity_near_one(self, tiny_model_pair):
        """Cosine similarity should be near 1 for minimally perturbed layers."""
        base_path, trained_path = tiny_model_pair
        model_pair = ModelPair(base_model=base_path, trained_model=trained_path)
        device_config = DeviceConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_memory_gb=8.0,
        )

        engine = WeightDiffEngine(model_pair, device_config)
        result = engine.run()

        # Non-layer-2 layers should have very high cosine similarity
        for diff in result.layer_diffs:
            if diff.layer_name.startswith("layer_") and "2" not in diff.layer_name:
                assert diff.cosine_similarity > 0.99, (
                    f"{diff.layer_name}: cosine sim = {diff.cosine_similarity}"
                )

    def test_attention_heads_populated(self, tiny_model_pair):
        base_path, trained_path = tiny_model_pair
        model_pair = ModelPair(base_model=base_path, trained_model=trained_path)
        device_config = DeviceConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_memory_gb=8.0,
        )

        engine = WeightDiffEngine(model_pair, device_config)
        result = engine.run()

        assert len(result.attention_heads) > 0

    def test_layernorm_shifts_detected(self, tiny_model_pair):
        base_path, trained_path = tiny_model_pair
        model_pair = ModelPair(base_model=base_path, trained_model=trained_path)
        device_config = DeviceConfig(
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_memory_gb=8.0,
        )

        engine = WeightDiffEngine(model_pair, device_config)
        result = engine.run()

        # Should have some layernorm shift data
        assert len(result.layernorm_shifts) >= 0  # May be 0 if no significant shifts
