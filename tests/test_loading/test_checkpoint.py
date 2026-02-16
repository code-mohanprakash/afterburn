"""Tests for checkpoint detection and validation."""

import json
from pathlib import Path

import pytest

from afterburn.exceptions import ModelLoadError, ModelNotFoundError
from afterburn.loading.checkpoint import CheckpointInfo, detect_checkpoint


class TestDetectCheckpoint:
    """Tests for detect_checkpoint() function."""

    def test_local_path_with_safetensors(self, tmp_path: Path):
        """Local path with safetensors files returns CheckpointInfo with format='safetensors'."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create fake safetensors files
        (model_dir / "model.safetensors").write_text("fake weights")
        (model_dir / "model-00001-of-00002.safetensors").write_text("fake weights")

        # Create config.json
        config = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        result = detect_checkpoint(str(model_dir))

        assert isinstance(result, CheckpointInfo)
        assert result.format == "safetensors"
        assert len(result.weight_files) == 2
        assert result.model_type == "llama"
        assert result.num_hidden_layers == 32
        assert result.hidden_size == 4096
        assert result.num_attention_heads == 32
        assert result.num_key_value_heads == 8
        assert result.intermediate_size == 11008
        assert result.vocab_size == 32000
        assert result.config_path == model_dir / "config.json"
        assert not result.has_lora

    def test_local_path_with_pytorch_bin(self, tmp_path: Path):
        """Local path with .bin files returns CheckpointInfo with format='pytorch'."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create fake bin files
        (model_dir / "pytorch_model.bin").write_text("fake weights")
        (model_dir / "pytorch_model-00001-of-00002.bin").write_text("fake weights")

        # Create config.json
        config = {
            "model_type": "gpt2",
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "vocab_size": 50257,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        result = detect_checkpoint(str(model_dir))

        assert isinstance(result, CheckpointInfo)
        assert result.format == "pytorch"
        assert len(result.weight_files) == 2
        assert result.model_type == "gpt2"
        assert result.num_hidden_layers == 12
        assert result.hidden_size == 768
        assert result.vocab_size == 50257

    def test_local_path_prefers_safetensors(self, tmp_path: Path):
        """When both safetensors and bin files exist, prefers safetensors."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create both types
        (model_dir / "model.safetensors").write_text("fake weights")
        (model_dir / "pytorch_model.bin").write_text("fake weights")

        config = {"model_type": "llama", "num_hidden_layers": 32}
        (model_dir / "config.json").write_text(json.dumps(config))

        result = detect_checkpoint(str(model_dir))

        assert result.format == "safetensors"
        assert len(result.weight_files) == 1
        assert result.weight_files[0].name == "model.safetensors"

    def test_path_does_not_exist_raises_error(self, tmp_path: Path):
        """Absolute path that doesn't exist raises ModelNotFoundError."""
        # Use an absolute path that doesn't exist
        nonexistent = tmp_path / "nonexistent"
        # Convert to absolute path
        absolute_nonexistent = nonexistent.resolve()

        with pytest.raises(ModelNotFoundError, match="Path does not exist"):
            detect_checkpoint(str(absolute_nonexistent))

    def test_path_is_file_not_directory_raises_error(self, tmp_path: Path):
        """Path that exists but is a file not directory raises ModelLoadError."""
        file_path = tmp_path / "model.bin"
        file_path.write_text("not a directory")

        with pytest.raises(ModelLoadError, match="Expected a directory, got a file"):
            detect_checkpoint(str(file_path))

    def test_directory_with_no_weight_files_raises_error(self, tmp_path: Path):
        """Directory with no weight files raises ModelLoadError."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        with pytest.raises(ModelLoadError, match="No model weights found"):
            detect_checkpoint(str(model_dir))

    def test_directory_without_config_json_raises_error(self, tmp_path: Path):
        """Directory without config.json raises ModelLoadError."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("fake weights")

        with pytest.raises(ModelLoadError, match="No config.json found"):
            detect_checkpoint(str(model_dir))

    def test_parses_config_json_correctly(self, tmp_path: Path):
        """Parses config.json and extracts all relevant fields."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        (model_dir / "model.safetensors").write_text("fake weights")

        config = {
            "model_type": "mistral",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "other_field": "ignored",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        result = detect_checkpoint(str(model_dir))

        assert result.model_type == "mistral"
        assert result.num_hidden_layers == 32
        assert result.hidden_size == 4096
        assert result.num_attention_heads == 32
        assert result.num_key_value_heads == 8
        assert result.intermediate_size == 14336
        assert result.vocab_size == 32000

    def test_detects_lora_adapters(self, tmp_path: Path):
        """Detects LoRA adapters when adapter_config.json is present."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        (model_dir / "model.safetensors").write_text("fake weights")
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "adapter_config.json").write_text(json.dumps({"r": 8}))

        result = detect_checkpoint(str(model_dir))

        assert result.has_lora is True

    def test_detects_lora_via_adapter_model_safetensors(self, tmp_path: Path):
        """Detects LoRA when adapter_model.safetensors is present."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        (model_dir / "model.safetensors").write_text("fake weights")
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "adapter_model.safetensors").write_text("fake adapter")

        result = detect_checkpoint(str(model_dir))

        assert result.has_lora is True

    def test_filters_out_training_args_bin(self, tmp_path: Path):
        """Filters out training_args.bin, optimizer.bin from weight files."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        (model_dir / "pytorch_model.bin").write_text("fake weights")
        (model_dir / "training_args.bin").write_text("training args")
        (model_dir / "optimizer.bin").write_text("optimizer state")
        (model_dir / "scheduler.bin").write_text("scheduler state")
        (model_dir / "config.json").write_text(json.dumps({"model_type": "gpt2"}))

        result = detect_checkpoint(str(model_dir))

        assert result.format == "pytorch"
        assert len(result.weight_files) == 1
        assert result.weight_files[0].name == "pytorch_model.bin"

    def test_detects_tokenizer_json(self, tmp_path: Path):
        """Detects tokenizer.json and sets tokenizer_path."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        (model_dir / "model.safetensors").write_text("fake weights")
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "tokenizer.json").write_text("{}")

        result = detect_checkpoint(str(model_dir))

        assert result.tokenizer_path == model_dir / "tokenizer.json"

    def test_detects_tokenizer_model(self, tmp_path: Path):
        """Detects tokenizer.model and sets tokenizer_path."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        (model_dir / "model.safetensors").write_text("fake weights")
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "tokenizer.model").write_bytes(b"fake sentencepiece")

        result = detect_checkpoint(str(model_dir))

        assert result.tokenizer_path == model_dir / "tokenizer.model"

    def test_handles_missing_config_fields(self, tmp_path: Path):
        """Handles config.json with missing optional fields gracefully."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        (model_dir / "model.safetensors").write_text("fake weights")
        # Minimal config with only model_type
        (model_dir / "config.json").write_text(json.dumps({"model_type": "custom"}))

        result = detect_checkpoint(str(model_dir))

        assert result.model_type == "custom"
        assert result.num_hidden_layers is None
        assert result.hidden_size is None
        assert result.num_attention_heads is None
        assert result.num_key_value_heads is None
        assert result.intermediate_size is None
        assert result.vocab_size is None
