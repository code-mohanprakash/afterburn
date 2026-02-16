"""Tests for model loader with compatibility validation."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from afterburn.device import DeviceConfig
from afterburn.exceptions import IncompatibleModelsError, ModelLoadError, ModelNotFoundError
from afterburn.loading.model_loader import ModelLoader


@pytest.fixture
def device_config():
    """Create a CPU device config for testing."""
    return DeviceConfig(
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_memory_gb=8.0,
        gpu_name=None,
    )


class TestModelLoader:
    """Tests for ModelLoader class."""

    def test_load_config_with_nonexistent_model_raises_error(self, device_config):
        """load_config() with non-existent model raises ModelNotFoundError."""
        loader = ModelLoader(device_config)

        with patch("afterburn.loading.model_loader.AutoConfig.from_pretrained") as mock_config:
            mock_config.side_effect = OSError("does not appear to have a file named config.json")

            with pytest.raises(ModelNotFoundError, match="not found"):
                loader.load_config("nonexistent/model")

    def test_load_config_with_os_error_raises_model_load_error(self, device_config):
        """load_config() with OSError raises ModelLoadError."""
        loader = ModelLoader(device_config)

        with patch("afterburn.loading.model_loader.AutoConfig.from_pretrained") as mock_config:
            mock_config.side_effect = OSError("Network error")

            with pytest.raises(ModelLoadError, match="Failed to load config"):
                loader.load_config("some/model")

    def test_load_config_success(self, device_config):
        """load_config() successfully returns config."""
        loader = ModelLoader(device_config)

        mock_cfg = MagicMock()
        mock_cfg.model_type = "llama"
        mock_cfg.hidden_size = 4096

        with patch("afterburn.loading.model_loader.AutoConfig.from_pretrained") as mock_config:
            mock_config.return_value = mock_cfg

            result = loader.load_config("meta-llama/Llama-2-7b")

            assert result == mock_cfg
            mock_config.assert_called_once_with("meta-llama/Llama-2-7b", trust_remote_code=False)

    def test_load_tokenizer_with_failing_load_raises_error(self, device_config):
        """load_tokenizer() with failing load raises ModelLoadError."""
        loader = ModelLoader(device_config)

        with patch("afterburn.loading.model_loader.AutoTokenizer.from_pretrained") as mock_tok:
            mock_tok.side_effect = ValueError("Failed to load tokenizer")

            with pytest.raises(ModelLoadError, match="Failed to load tokenizer"):
                loader.load_tokenizer("some/model")

    def test_load_tokenizer_sets_pad_token_if_none(self, device_config):
        """load_tokenizer() sets pad_token to eos_token if pad_token is None."""
        loader = ModelLoader(device_config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"

        with patch("afterburn.loading.model_loader.AutoTokenizer.from_pretrained") as mock_tok:
            mock_tok.return_value = mock_tokenizer

            result = loader.load_tokenizer("meta-llama/Llama-2-7b")

            assert result.pad_token == "</s>"

    def test_load_tokenizer_preserves_existing_pad_token(self, device_config):
        """load_tokenizer() preserves existing pad_token if present."""
        loader = ModelLoader(device_config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"

        with patch("afterburn.loading.model_loader.AutoTokenizer.from_pretrained") as mock_tok:
            mock_tok.return_value = mock_tokenizer

            result = loader.load_tokenizer("meta-llama/Llama-2-7b")

            assert result.pad_token == "<pad>"

    def test_validate_compatibility_with_matching_configs_no_error(self, device_config):
        """validate_compatibility() with matching configs raises no error."""
        loader = ModelLoader(device_config)

        base_config = MagicMock()
        base_config.hidden_size = 4096
        base_config.num_hidden_layers = 32
        base_config.num_attention_heads = 32
        base_config.intermediate_size = 11008
        base_config.vocab_size = 32000

        trained_config = MagicMock()
        trained_config.hidden_size = 4096
        trained_config.num_hidden_layers = 32
        trained_config.num_attention_heads = 32
        trained_config.intermediate_size = 11008
        trained_config.vocab_size = 32000

        with patch.object(loader, "load_config") as mock_load:
            mock_load.side_effect = [base_config, trained_config]

            # Should not raise
            loader.validate_compatibility("base/model", "trained/model")

    def test_validate_compatibility_with_mismatched_hidden_size_raises_error(self, device_config):
        """validate_compatibility() with mismatched hidden_size raises IncompatibleModelsError."""
        loader = ModelLoader(device_config)

        base_config = MagicMock()
        base_config.hidden_size = 4096
        base_config.num_hidden_layers = 32
        base_config.num_attention_heads = 32
        base_config.intermediate_size = 11008
        base_config.vocab_size = 32000

        trained_config = MagicMock()
        trained_config.hidden_size = 2048  # Different!
        trained_config.num_hidden_layers = 32
        trained_config.num_attention_heads = 32
        trained_config.intermediate_size = 11008
        trained_config.vocab_size = 32000

        with patch.object(loader, "load_config") as mock_load:
            mock_load.side_effect = [base_config, trained_config]

            with pytest.raises(IncompatibleModelsError, match="hidden_size"):
                loader.validate_compatibility("base/model", "trained/model")

    def test_validate_compatibility_with_mismatched_vocab_size_raises_error(self, device_config):
        """validate_compatibility() with mismatched vocab_size raises IncompatibleModelsError."""
        loader = ModelLoader(device_config)

        base_config = MagicMock()
        base_config.hidden_size = 4096
        base_config.num_hidden_layers = 32
        base_config.num_attention_heads = 32
        base_config.intermediate_size = 11008
        base_config.vocab_size = 32000

        trained_config = MagicMock()
        trained_config.hidden_size = 4096
        trained_config.num_hidden_layers = 32
        trained_config.num_attention_heads = 32
        trained_config.intermediate_size = 11008
        trained_config.vocab_size = 50257  # Different!

        with patch.object(loader, "load_config") as mock_load:
            mock_load.side_effect = [base_config, trained_config]

            with pytest.raises(IncompatibleModelsError, match="vocab_size"):
                loader.validate_compatibility("base/model", "trained/model")

    def test_validate_compatibility_with_multiple_mismatches(self, device_config):
        """validate_compatibility() reports all mismatches."""
        loader = ModelLoader(device_config)

        base_config = MagicMock()
        base_config.hidden_size = 4096
        base_config.num_hidden_layers = 32
        base_config.num_attention_heads = 32
        base_config.intermediate_size = 11008
        base_config.vocab_size = 32000

        trained_config = MagicMock()
        trained_config.hidden_size = 2048  # Different
        trained_config.num_hidden_layers = 24  # Different
        trained_config.num_attention_heads = 32
        trained_config.intermediate_size = 11008
        trained_config.vocab_size = 32000

        with patch.object(loader, "load_config") as mock_load:
            mock_load.side_effect = [base_config, trained_config]

            with pytest.raises(IncompatibleModelsError) as exc_info:
                loader.validate_compatibility("base/model", "trained/model")

            error_msg = str(exc_info.value)
            assert "hidden_size" in error_msg
            assert "num_hidden_layers" in error_msg

    def test_validate_compatibility_ignores_none_values(self, device_config):
        """validate_compatibility() ignores None values in configs."""
        loader = ModelLoader(device_config)

        base_config = MagicMock()
        base_config.hidden_size = 4096
        base_config.num_hidden_layers = None  # None in base
        base_config.num_attention_heads = 32
        base_config.intermediate_size = 11008
        base_config.vocab_size = 32000

        trained_config = MagicMock()
        trained_config.hidden_size = 4096
        trained_config.num_hidden_layers = 32  # Non-None in trained
        trained_config.num_attention_heads = 32
        trained_config.intermediate_size = 11008
        trained_config.vocab_size = 32000

        with patch.object(loader, "load_config") as mock_load:
            mock_load.side_effect = [base_config, trained_config]

            # Should not raise because base has None
            loader.validate_compatibility("base/model", "trained/model")

    def test_unload_model_works_without_error(self, device_config):
        """unload_model() works without error."""
        loader = ModelLoader(device_config)
        mock_model = MagicMock()

        # Should not raise
        loader.unload_model(mock_model)

    def test_unload_model_clears_cuda_cache_if_available(self, device_config):
        """unload_model() clears CUDA cache if CUDA is available."""
        loader = ModelLoader(device_config)
        mock_model = MagicMock()

        with patch("afterburn.loading.model_loader.torch.cuda.is_available") as mock_cuda:
            with patch("afterburn.loading.model_loader.torch.cuda.empty_cache") as mock_cache:
                mock_cuda.return_value = True

                loader.unload_model(mock_model)

                mock_cache.assert_called_once()

    def test_get_checkpoint_info_calls_detect_checkpoint(self, device_config):
        """get_checkpoint_info() calls detect_checkpoint."""
        loader = ModelLoader(device_config)

        with patch("afterburn.loading.model_loader.detect_checkpoint") as mock_detect:
            mock_info = MagicMock()
            mock_detect.return_value = mock_info

            result = loader.get_checkpoint_info("meta-llama/Llama-2-7b")

            assert result == mock_info
            mock_detect.assert_called_once_with("meta-llama/Llama-2-7b")
