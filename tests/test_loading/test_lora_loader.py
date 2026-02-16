"""Tests for LoRA adapter loading and analysis."""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from afterburn.exceptions import ModelLoadError
from afterburn.loading.lora_loader import LoRAConfig, LoRAWeights, load_lora_adapter


class TestLoadLoraAdapter:
    """Tests for load_lora_adapter() function."""

    def test_path_without_adapter_config_raises_error(self, tmp_path: Path):
        """Path without adapter_config.json raises ModelLoadError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with pytest.raises(ModelLoadError, match="No adapter_config.json found"):
            load_lora_adapter(adapter_dir)

    def test_valid_adapter_directory_with_safetensors_loads_correctly(self, tmp_path: Path):
        """Valid adapter directory with safetensors loads correctly."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        # Create adapter_config.json
        config = {
            "r": 8,
            "lora_alpha": 16.0,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "meta-llama/Llama-2-7b",
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        # Create a fake safetensors file
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake safetensors data")

        # Mock safetensors loading
        mock_weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 4096),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(4096, 8),
            "base_model.model.layers.1.self_attn.v_proj.lora_A.weight": torch.randn(8, 4096),
            "base_model.model.layers.1.self_attn.v_proj.lora_B.weight": torch.randn(4096, 8),
        }

        with patch("safetensors.safe_open") as mock_safe_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            mock_file.keys.return_value = mock_weights.keys()
            mock_file.get_tensor.side_effect = lambda k: mock_weights[k]
            mock_safe_open.return_value = mock_file

            result = load_lora_adapter(adapter_dir)

        assert isinstance(result, LoRAWeights)
        assert result.config.r == 8
        assert result.config.lora_alpha == 16.0
        assert result.config.target_modules == ["q_proj", "v_proj"]
        assert result.config.lora_dropout == 0.05
        assert result.config.bias == "none"
        assert result.config.task_type == "CAUSAL_LM"
        assert result.config.base_model_name_or_path == "meta-llama/Llama-2-7b"
        assert len(result.weights) == 4
        assert "layer_0" in result.affected_layers
        assert "layer_1" in result.affected_layers

    def test_valid_adapter_directory_with_bin_loads_correctly(self, tmp_path: Path):
        """Valid adapter directory with .bin loads correctly."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        # Create adapter_config.json
        config = {
            "r": 16,
            "lora_alpha": 32.0,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        # Create a fake bin file
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake pytorch data")

        # Mock torch.load
        mock_weights = {
            "base_model.model.layers.5.self_attn.q_proj.lora_A.weight": torch.randn(16, 4096),
            "base_model.model.layers.5.self_attn.q_proj.lora_B.weight": torch.randn(4096, 16),
        }

        with patch("afterburn.loading.lora_loader.torch.load") as mock_load:
            mock_load.return_value = mock_weights

            result = load_lora_adapter(adapter_dir)

        assert isinstance(result, LoRAWeights)
        assert result.config.r == 16
        assert result.config.lora_alpha == 32.0
        assert result.config.target_modules == ["q_proj", "k_proj", "v_proj"]
        assert len(result.weights) == 2
        assert "layer_5" in result.affected_layers

    def test_correctly_parses_lora_config(self, tmp_path: Path):
        """Correctly parses LoRA config with all fields."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {
            "r": 64,
            "lora_alpha": 128.0,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "lora_only",
            "task_type": "SEQ_CLS",
            "base_model_name_or_path": "gpt2",
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake")

        mock_weights = {"dummy.weight": torch.randn(10, 10)}

        with patch("afterburn.loading.lora_loader.torch.load") as mock_load:
            mock_load.return_value = mock_weights

            result = load_lora_adapter(adapter_dir)

            # Verify all config fields were parsed correctly
            assert result.config.r == 64
            assert result.config.lora_alpha == 128.0
            assert result.config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]
            assert result.config.lora_dropout == 0.1
            assert result.config.bias == "lora_only"
            assert result.config.task_type == "SEQ_CLS"
            assert result.config.base_model_name_or_path == "gpt2"

    def test_identifies_affected_layers_from_weight_keys(self, tmp_path: Path):
        """Correctly identifies affected layers from weight keys."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {"r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"]}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake")

        mock_weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 4096),
            "base_model.model.layers.3.self_attn.q_proj.lora_A.weight": torch.randn(8, 4096),
            "base_model.model.layers.7.self_attn.q_proj.lora_A.weight": torch.randn(8, 4096),
            "base_model.model.layers.15.self_attn.q_proj.lora_A.weight": torch.randn(8, 4096),
        }

        with patch("safetensors.safe_open") as mock_safe_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            mock_file.keys.return_value = mock_weights.keys()
            mock_file.get_tensor.side_effect = lambda k: mock_weights[k]
            mock_safe_open.return_value = mock_file

            result = load_lora_adapter(adapter_dir)

        assert result.affected_layers == ["layer_0", "layer_15", "layer_3", "layer_7"]  # Sorted

    def test_no_adapter_weights_found_raises_error(self, tmp_path: Path):
        """No adapter weights found raises ModelLoadError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {"r": 8, "lora_alpha": 16.0}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        # No adapter_model.safetensors or adapter_model.bin

        with pytest.raises(ModelLoadError, match="No adapter weights found"):
            load_lora_adapter(adapter_dir)

    def test_handles_minimal_config(self, tmp_path: Path):
        """Handles minimal config with default values."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        # Minimal config
        config = {}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake")

        mock_weights = {
            "base_model.model.layers.0.mlp.weight": torch.randn(10, 10),
        }

        with patch("afterburn.loading.lora_loader.torch.load") as mock_load:
            mock_load.return_value = mock_weights

            result = load_lora_adapter(adapter_dir)

        # Check defaults
        assert result.config.r == 0
        assert result.config.lora_alpha == 1.0
        assert result.config.target_modules == []
        assert result.config.lora_dropout == 0.0
        assert result.config.bias == "none"
        assert result.config.task_type == ""
        assert result.config.base_model_name_or_path == ""

    def test_accepts_string_path(self, tmp_path: Path):
        """Accepts string path as well as Path object."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {"r": 8}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake")

        mock_weights = {"layer.0.weight": torch.randn(10, 10)}

        with patch("afterburn.loading.lora_loader.torch.load") as mock_load:
            mock_load.return_value = mock_weights

            # Pass as string
            result = load_lora_adapter(str(adapter_dir))

        assert result.config.r == 8

    def test_prefers_safetensors_over_bin(self, tmp_path: Path):
        """When both safetensors and bin exist, prefers safetensors."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {"r": 8}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake safetensors")
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake bin")

        mock_weights = {"layer.0.weight": torch.randn(10, 10)}

        with patch("safetensors.safe_open") as mock_safe_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            mock_file.keys.return_value = mock_weights.keys()
            mock_file.get_tensor.side_effect = lambda k: mock_weights[k]
            mock_safe_open.return_value = mock_file

            result = load_lora_adapter(adapter_dir)

            # Verify safetensors was used
            mock_safe_open.assert_called_once()

    def test_computes_total_parameter_count(self, tmp_path: Path):
        """Computes total parameter count across all LoRA weights."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        config = {"r": 8}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake")

        # Create tensors with known sizes
        mock_weights = {
            "lora_A": torch.randn(8, 100),  # 800 params
            "lora_B": torch.randn(100, 8),  # 800 params
        }

        with patch("afterburn.loading.lora_loader.torch.load") as mock_load:
            mock_load.return_value = mock_weights

            result = load_lora_adapter(adapter_dir)

            # Total should be 800 + 800 = 1600
            total_params = sum(w.numel() for w in result.weights.values())
            assert total_params == 1600
