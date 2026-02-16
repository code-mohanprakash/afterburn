"""Tests for layer-by-layer model weight iteration."""

import re
from unittest.mock import MagicMock, patch

import pytest
import torch

from afterburn.loading.layer_iterator import LAYER_INDEX_PATTERN, LayerIterator


class TestLayerIndexPattern:
    """Tests for LAYER_INDEX_PATTERN regex."""

    def test_matches_model_layers_pattern(self):
        """LAYER_INDEX_PATTERN matches 'model.layers.0.self_attn.q_proj.weight'."""
        param_name = "model.layers.0.self_attn.q_proj.weight"
        match = LAYER_INDEX_PATTERN.search(param_name)

        assert match is not None
        assert match.group(1) == "0"

    def test_matches_transformer_h_pattern(self):
        """LAYER_INDEX_PATTERN matches 'transformer.h.5.mlp.weight'."""
        param_name = "transformer.h.5.mlp.weight"
        match = LAYER_INDEX_PATTERN.search(param_name)

        assert match is not None
        assert match.group(1) == "5"

    def test_matches_gpt_neox_layers_pattern(self):
        """LAYER_INDEX_PATTERN matches 'gpt_neox.layers.12.attention.weight'."""
        param_name = "gpt_neox.layers.12.attention.weight"
        match = LAYER_INDEX_PATTERN.search(param_name)

        assert match is not None
        assert match.group(1) == "12"

    def test_does_not_match_embedding(self):
        """LAYER_INDEX_PATTERN does not match embedding keys."""
        param_name = "model.embed_tokens.weight"
        match = LAYER_INDEX_PATTERN.search(param_name)

        assert match is None

    def test_does_not_match_lm_head(self):
        """LAYER_INDEX_PATTERN does not match lm_head keys."""
        param_name = "lm_head.weight"
        match = LAYER_INDEX_PATTERN.search(param_name)

        assert match is None


class TestLayerIterator:
    """Tests for LayerIterator class."""

    @pytest.fixture
    def mock_checkpoint(self):
        """Create a mock CheckpointInfo."""
        checkpoint = MagicMock()
        checkpoint.format = "safetensors"
        checkpoint.weight_files = []
        checkpoint.num_hidden_layers = 4
        checkpoint.local_path = MagicMock()
        return checkpoint

    def test_build_layer_map_groups_parameters_correctly(self, mock_checkpoint):
        """_build_layer_map() groups parameters correctly."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            # Mock the parameter keys
            param_keys = [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.1.self_attn.q_proj.weight",
                "model.layers.1.mlp.gate_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ]

            iterator = LayerIterator("test/model")
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                layer_map = iterator._build_layer_map()

            assert "embedding" in layer_map
            assert "model.embed_tokens.weight" in layer_map["embedding"]

            assert "layer_0" in layer_map
            assert "model.layers.0.self_attn.q_proj.weight" in layer_map["layer_0"]
            assert "model.layers.0.self_attn.k_proj.weight" in layer_map["layer_0"]

            assert "layer_1" in layer_map
            assert "model.layers.1.self_attn.q_proj.weight" in layer_map["layer_1"]
            assert "model.layers.1.mlp.gate_proj.weight" in layer_map["layer_1"]

            assert "final_norm" in layer_map
            assert "model.norm.weight" in layer_map["final_norm"]

            assert "lm_head" in layer_map
            assert "lm_head.weight" in layer_map["lm_head"]

    def test_embedding_keys_grouped_under_embedding(self, mock_checkpoint):
        """Embedding keys are grouped under 'embedding'."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            param_keys = [
                "model.embed_tokens.weight",
                "transformer.wte.weight",
                "transformer.wpe.weight",
                "gpt_neox.embed_in.weight",
            ]

            iterator = LayerIterator("test/model")
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                layer_map = iterator._build_layer_map()

            assert "embedding" in layer_map
            assert len(layer_map["embedding"]) == 4
            assert all(key in layer_map["embedding"] for key in param_keys)

    def test_lm_head_keys_grouped_under_lm_head(self, mock_checkpoint):
        """lm_head keys are grouped under 'lm_head'."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            param_keys = [
                "lm_head.weight",
                "model.output.weight",
            ]

            iterator = LayerIterator("test/model")
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                layer_map = iterator._build_layer_map()

            assert "lm_head" in layer_map
            assert "lm_head.weight" in layer_map["lm_head"]
            assert "model.output.weight" in layer_map["lm_head"]

    def test_final_norm_keys_grouped_under_final_norm(self, mock_checkpoint):
        """Final norm keys are grouped under 'final_norm'."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            param_keys = [
                "model.norm.weight",
                "transformer.ln_f.weight",
                "gpt_neox.final_layer_norm.weight",
            ]

            iterator = LayerIterator("test/model")
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                layer_map = iterator._build_layer_map()

            assert "final_norm" in layer_map
            assert len(layer_map["final_norm"]) == 3

    def test_layer_names_returns_sorted_names(self, mock_checkpoint):
        """layer_names returns sorted layer names."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            param_keys = [
                "model.layers.2.weight",
                "model.layers.0.weight",
                "model.layers.1.weight",
                "model.embed_tokens.weight",
            ]

            iterator = LayerIterator("test/model")
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                iterator._layer_map = iterator._build_layer_map()

            names = iterator.layer_names

            assert names == sorted(names)
            assert "embedding" in names
            assert "layer_0" in names
            assert "layer_1" in names
            assert "layer_2" in names

    def test_num_layers_returns_correct_count(self, mock_checkpoint):
        """num_layers returns correct count from checkpoint."""
        mock_checkpoint.num_hidden_layers = 32

        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            iterator = LayerIterator("test/model")

            assert iterator.num_layers == 32

    def test_get_layer_with_invalid_name_raises_key_error(self, mock_checkpoint):
        """get_layer() with invalid name raises KeyError."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            param_keys = ["model.layers.0.weight"]

            iterator = LayerIterator("test/model")
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                iterator._layer_map = iterator._build_layer_map()

            with pytest.raises(KeyError, match="Layer 'invalid_layer' not found"):
                iterator.get_layer("invalid_layer")

    def test_get_layer_loads_correct_parameters(self, mock_checkpoint):
        """get_layer() loads the correct parameters."""
        mock_checkpoint.format = "safetensors"

        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            param_keys = [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ]

            iterator = LayerIterator("test/model", device=torch.device("cpu"))
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                iterator._layer_map = iterator._build_layer_map()

            mock_tensor = torch.randn(10, 10)
            with patch.object(iterator, "_load_params") as mock_load:
                mock_load.return_value = {
                    "model.layers.0.self_attn.q_proj.weight": mock_tensor,
                    "model.layers.0.self_attn.k_proj.weight": mock_tensor,
                }

                result = iterator.get_layer("layer_0")

                assert len(result) == 2
                assert "model.layers.0.self_attn.q_proj.weight" in result
                assert "model.layers.0.self_attn.k_proj.weight" in result
                mock_load.assert_called_once()

    def test_per_layer_norm_grouped_with_layer(self, mock_checkpoint):
        """Per-layer norm keys are grouped with their layer, not final_norm."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            param_keys = [
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.1.input_layernorm.weight",
                "model.norm.weight",  # This is final norm
            ]

            iterator = LayerIterator("test/model")
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                layer_map = iterator._build_layer_map()

            # Per-layer norms should be in their respective layers
            assert "model.layers.0.input_layernorm.weight" in layer_map["layer_0"]
            assert "model.layers.0.post_attention_layernorm.weight" in layer_map["layer_0"]
            assert "model.layers.1.input_layernorm.weight" in layer_map["layer_1"]

            # Final norm should be separate
            assert "final_norm" in layer_map
            assert "model.norm.weight" in layer_map["final_norm"]

    def test_iterate_layers_yields_in_correct_order(self, mock_checkpoint):
        """iterate_layers() yields layers in logical order."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            param_keys = [
                "lm_head.weight",
                "model.norm.weight",
                "model.layers.1.weight",
                "model.layers.0.weight",
                "model.embed_tokens.weight",
            ]

            iterator = LayerIterator("test/model", device=torch.device("cpu"))
            with patch.object(iterator, "_get_all_parameter_keys", return_value=param_keys):
                iterator._layer_map = iterator._build_layer_map()

            with patch.object(iterator, "get_layer") as mock_get:
                mock_get.return_value = {}

                layer_order = []
                for layer_name, _ in iterator.iterate_layers():
                    layer_order.append(layer_name)

                # Check order: embedding, layer_0, layer_1, final_norm, lm_head
                assert layer_order == ["embedding", "layer_0", "layer_1", "final_norm", "lm_head"]

    def test_paired_iterate_yields_common_layers(self, mock_checkpoint):
        """paired_iterate() yields only layers common to both models."""
        with patch("afterburn.loading.layer_iterator.detect_checkpoint") as mock_detect:
            mock_detect.return_value = mock_checkpoint

            # Base has layer_0, layer_1, embedding
            base_keys = [
                "model.embed_tokens.weight",
                "model.layers.0.weight",
                "model.layers.1.weight",
            ]

            # Trained has layer_1, layer_2, embedding
            trained_keys = [
                "model.embed_tokens.weight",
                "model.layers.1.weight",
                "model.layers.2.weight",
            ]

            base_iter = LayerIterator("base/model")
            trained_iter = LayerIterator("trained/model")

            with patch.object(base_iter, "_get_all_parameter_keys", return_value=base_keys):
                with patch.object(trained_iter, "_get_all_parameter_keys", return_value=trained_keys):
                    base_iter._layer_map = base_iter._build_layer_map()
                    trained_iter._layer_map = trained_iter._build_layer_map()

            with patch.object(base_iter, "get_layer") as mock_base_get:
                with patch.object(trained_iter, "get_layer") as mock_trained_get:
                    mock_base_get.return_value = {}
                    mock_trained_get.return_value = {}

                    common_layers = []
                    for layer_name, _, _ in LayerIterator.paired_iterate(base_iter, trained_iter):
                        common_layers.append(layer_name)

                    # Only embedding and layer_1 are common
                    assert "embedding" in common_layers
                    assert "layer_1" in common_layers
                    assert "layer_0" not in common_layers  # Only in base
                    assert "layer_2" not in common_layers  # Only in trained
