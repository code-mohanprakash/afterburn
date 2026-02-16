"""Tests for path validation and sanitization."""

from __future__ import annotations

from pathlib import Path

import pytest

from afterburn.diagnoser import _validate_model_id
from afterburn.exceptions import PathValidationError
from afterburn.loading.checkpoint import _validate_path


def test_valid_huggingface_ids():
    """Test that valid HuggingFace model IDs are accepted."""
    valid_ids = [
        "gpt2",
        "meta-llama/Llama-3.1-8B",
        "my-org/my-model",
        "org/model-name_v2.0",
        "google/gemma-2b",
        "mistralai/Mistral-7B-v0.1",
    ]
    for model_id in valid_ids:
        _validate_model_id(model_id)  # Should not raise


def test_invalid_huggingface_ids():
    """Test that invalid HuggingFace model IDs are rejected."""
    invalid_ids = [
        "../etc/passwd",
        "model/../../../etc/passwd",
        "model; rm -rf /",
        "model`whoami`",
        "model$(whoami)",
        "model@hostname",
    ]
    for model_id in invalid_ids:
        with pytest.raises(PathValidationError):
            _validate_model_id(model_id)


def test_directory_traversal_in_model_id():
    """Test that directory traversal patterns are blocked."""
    with pytest.raises(PathValidationError, match="directory traversal"):
        _validate_model_id("model/../../../etc/passwd")

    with pytest.raises(PathValidationError, match="directory traversal"):
        _validate_model_id("../model")


def test_valid_local_path(tmp_path):
    """Test that valid local paths are accepted."""
    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()

    # Should not raise
    _validate_model_id(str(model_dir))


def test_local_path_with_traversal(tmp_path):
    """Test that local paths with directory traversal are blocked."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Try to use directory traversal in the path
    traversal_path = str(tmp_path / "model" / ".." / ".." / "etc" / "passwd")

    with pytest.raises(PathValidationError, match="directory traversal"):
        _validate_model_id(traversal_path)


def test_validate_path_with_absolute_path(tmp_path):
    """Test that absolute paths are validated correctly."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Should not raise
    _validate_path(model_dir)


def test_validate_path_with_traversal():
    """Test that paths with directory traversal patterns are rejected."""
    dangerous_paths = [
        Path("/tmp/model/../../../etc/passwd"),
        Path("../../../etc/passwd"),
        Path("/var/data/../../etc/passwd"),
    ]

    for path in dangerous_paths:
        with pytest.raises(PathValidationError, match="directory traversal"):
            _validate_path(path)


def test_validate_path_normalizes_symlinks(tmp_path):
    """Test that symlinks are resolved to their target."""
    # Create a real directory
    real_dir = tmp_path / "real_model"
    real_dir.mkdir()

    # Create a symlink to it
    symlink_dir = tmp_path / "symlink_model"
    symlink_dir.symlink_to(real_dir)

    # Should not raise - symlinks are allowed as long as they resolve safely
    _validate_path(symlink_dir)


def test_nonexistent_huggingface_id():
    """Test that nonexistent HuggingFace IDs are accepted at validation stage."""
    # Validation should pass - the actual existence check happens during download
    _validate_model_id("nonexistent-org/nonexistent-model")


def test_model_id_with_special_characters():
    """Test model IDs with special but valid characters."""
    valid_ids = [
        "bert-base-uncased",
        "model_name_123",
        "org/model.v2.0",
        "company-ai/llama-3.1-8b-instruct",
    ]
    for model_id in valid_ids:
        _validate_model_id(model_id)  # Should not raise


def test_empty_model_id():
    """Test that empty model IDs are rejected."""
    with pytest.raises(PathValidationError):
        _validate_model_id("")


def test_model_id_with_null_bytes():
    """Test that model IDs with null bytes are rejected."""
    with pytest.raises(PathValidationError):
        _validate_model_id("model\x00name")


def test_model_id_with_newlines():
    """Test that model IDs with newlines are rejected."""
    with pytest.raises(PathValidationError):
        _validate_model_id("model\nname")


def test_relative_path_vs_model_id(tmp_path):
    """Test that relative paths that don't exist are treated as model IDs."""
    # This doesn't exist as a path, so it's treated as a HuggingFace ID
    _validate_model_id("relative/path/to/model")  # Should not raise


def test_absolute_nonexistent_path():
    """Test absolute paths that don't exist."""
    # Absolute path that doesn't exist - will be treated as invalid
    # The function checks if path.exists() first
    nonexistent = "/absolutely/nonexistent/path/to/model/12345"
    # This will be validated as a HuggingFace ID since it doesn't exist
    _validate_model_id(nonexistent)  # Should validate as model ID format
