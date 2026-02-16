"""Tests for resource management, retry logic, and cleanup."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from afterburn.device import check_disk_space, estimate_model_memory_gb, register_cleanup
from afterburn.exceptions import ModelLoadError, ModelNotFoundError, OutOfMemoryError
from afterburn.loading.checkpoint import _download_and_inspect


class TestDiskSpaceCheck:
    """Tests for disk space checking."""

    def test_sufficient_disk_space(self, tmp_path: Path):
        """Test that check passes when sufficient space is available."""
        # Should not raise - we have enough space in tmp
        check_disk_space(tmp_path, required_gb=0.001)

    def test_insufficient_disk_space(self, tmp_path: Path):
        """Test that OutOfMemoryError is raised when insufficient space."""
        # Request an impossibly large amount of space
        with pytest.raises(OutOfMemoryError, match="Insufficient disk space"):
            check_disk_space(tmp_path, required_gb=999999999.0)

    def test_disk_space_nonexistent_path(self):
        """Test that OutOfMemoryError is raised for invalid path."""
        invalid_path = Path("/nonexistent/path/that/should/not/exist")
        with pytest.raises(OutOfMemoryError, match="Failed to check disk space"):
            check_disk_space(invalid_path, required_gb=1.0)


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_float32(self):
        """Test memory estimation for float32."""
        num_params = 1_000_000_000  # 1B parameters
        estimated = estimate_model_memory_gb(num_params, torch.float32)
        # 1B params * 4 bytes = 4,000,000,000 bytes / 1024^3 = ~3.73 GB
        expected = (num_params * 4) / (1024**3)
        assert abs(estimated - expected) < 0.01

    def test_estimate_float16(self):
        """Test memory estimation for float16."""
        num_params = 1_000_000_000  # 1B parameters
        estimated = estimate_model_memory_gb(num_params, torch.float16)
        # 1B params * 2 bytes = 2,000,000,000 bytes / 1024^3 = ~1.86 GB
        expected = (num_params * 2) / (1024**3)
        assert abs(estimated - expected) < 0.01

    def test_estimate_bfloat16(self):
        """Test memory estimation for bfloat16."""
        num_params = 7_000_000_000  # 7B parameters
        estimated = estimate_model_memory_gb(num_params, torch.bfloat16)
        # 7B params * 2 bytes = 14,000,000,000 bytes / 1024^3 = ~13.04 GB
        expected = (num_params * 2) / (1024**3)
        assert abs(estimated - expected) < 0.01

    def test_estimate_int8(self):
        """Test memory estimation for int8 quantized models."""
        num_params = 1_000_000_000  # 1B parameters
        estimated = estimate_model_memory_gb(num_params, torch.int8)
        # 1B params * 1 byte = 1,000,000,000 bytes / 1024^3 = ~0.93 GB
        expected = (num_params * 1) / (1024**3)
        assert abs(estimated - expected) < 0.01


class TestCleanupRegistry:
    """Tests for cleanup callback registry."""

    def test_register_cleanup_callback(self):
        """Test that cleanup callbacks can be registered and executed."""
        callback_executed = []

        def cleanup_callback():
            callback_executed.append(True)

        register_cleanup(cleanup_callback)

        # Import the cleanup handler
        from afterburn.device import _cleanup_handler

        _cleanup_handler()

        assert len(callback_executed) > 0

    def test_cleanup_handles_exceptions(self):
        """Test that cleanup continues even if a callback raises."""
        from afterburn.device import _cleanup_callbacks, _cleanup_handler

        # Clear existing callbacks
        initial_count = len(_cleanup_callbacks)

        def failing_callback():
            raise RuntimeError("Intentional failure")

        def success_callback():
            success_callback.called = True

        success_callback.called = False

        register_cleanup(failing_callback)
        register_cleanup(success_callback)

        # Should not raise even though first callback fails
        _cleanup_handler()

        # Second callback should still execute
        assert success_callback.called


class TestRetryLogic:
    """Tests for retry logic in downloads."""

    @patch("afterburn.loading.checkpoint.snapshot_download")
    def test_successful_download_first_try(self, mock_download, tmp_path: Path):
        """Test successful download on first attempt."""
        # Create a minimal model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "gpt2", "num_hidden_layers": 12}')
        (model_dir / "model.safetensors").write_text("fake weights")

        mock_download.return_value = str(model_dir)

        result = _download_and_inspect("test/model")

        assert result.local_path == model_dir
        assert mock_download.call_count == 1

    @patch("afterburn.loading.checkpoint.time.sleep")
    @patch("afterburn.loading.checkpoint.snapshot_download")
    def test_retry_on_connection_error(self, mock_download, mock_sleep, tmp_path: Path):
        """Test that ConnectionError triggers retry."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "gpt2", "num_hidden_layers": 12}')
        (model_dir / "model.safetensors").write_text("fake weights")

        # Fail twice, then succeed
        mock_download.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            str(model_dir),
        ]

        result = _download_and_inspect("test/model")

        assert result.local_path == model_dir
        assert mock_download.call_count == 3
        # Should have slept twice (1s, 2s)
        assert mock_sleep.call_count == 2

    @patch("afterburn.loading.checkpoint.time.sleep")
    @patch("afterburn.loading.checkpoint.snapshot_download")
    def test_retry_on_timeout_error(self, mock_download, mock_sleep):
        """Test that TimeoutError triggers retry."""
        mock_download.side_effect = [
            TimeoutError("Request timed out"),
            TimeoutError("Request timed out"),
            TimeoutError("Request timed out"),
        ]

        with pytest.raises(ModelLoadError, match="Failed to download.*after 3 attempts"):
            _download_and_inspect("test/model")

        assert mock_download.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("afterburn.loading.checkpoint.snapshot_download")
    def test_no_retry_on_repository_not_found(self, mock_download):
        """Test that RepositoryNotFoundError does not trigger retry."""
        mock_download.side_effect = RepositoryNotFoundError("Not found")

        with pytest.raises(ModelNotFoundError, match="not found"):
            _download_and_inspect("test/nonexistent")

        # Should only try once
        assert mock_download.call_count == 1

    @patch("afterburn.loading.checkpoint.time.sleep")
    @patch("afterburn.loading.checkpoint.snapshot_download")
    def test_retry_on_server_error(self, mock_download, mock_sleep, tmp_path: Path):
        """Test that 5xx errors trigger retry."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "gpt2", "num_hidden_layers": 12}')
        (model_dir / "model.safetensors").write_text("fake weights")

        # Create mock response with 503 error
        mock_response = Mock()
        mock_response.status_code = 503
        error_503 = HfHubHTTPError("Service unavailable")
        error_503.response = mock_response

        mock_download.side_effect = [
            error_503,
            str(model_dir),
        ]

        result = _download_and_inspect("test/model")

        assert result.local_path == model_dir
        assert mock_download.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("afterburn.loading.checkpoint.snapshot_download")
    def test_no_retry_on_client_error(self, mock_download):
        """Test that 4xx errors do not trigger retry."""
        mock_response = Mock()
        mock_response.status_code = 404
        error_404 = HfHubHTTPError("Not found")
        error_404.response = mock_response

        mock_download.side_effect = error_404

        with pytest.raises(ModelLoadError, match="Failed to download"):
            _download_and_inspect("test/model")

        # Should only try once
        assert mock_download.call_count == 1

    @patch("afterburn.loading.checkpoint.time.sleep")
    @patch("afterburn.loading.checkpoint.snapshot_download")
    def test_exponential_backoff_timing(self, mock_download, mock_sleep):
        """Test that backoff follows exponential pattern (1s, 2s, 4s)."""
        mock_download.side_effect = [
            ConnectionError("Error 1"),
            ConnectionError("Error 2"),
            ConnectionError("Error 3"),
        ]

        with pytest.raises(ModelLoadError):
            _download_and_inspect("test/model")

        # Check that sleep was called with correct durations
        assert mock_sleep.call_count == 2
        calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert calls == [1, 2]
