"""Checkpoint format detection and validation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from huggingface_hub.utils import (  # type: ignore[attr-defined]
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from afterburn.exceptions import ModelLoadError, ModelNotFoundError, PathValidationError

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a model checkpoint."""

    local_path: Path
    format: str  # "safetensors" or "pytorch"
    weight_files: list[Path] = field(default_factory=list)
    config_path: Path | None = None
    tokenizer_path: Path | None = None
    has_lora: bool = False
    model_type: str | None = None
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None


def _validate_path(path: Path) -> None:
    """Validate that a path is safe and doesn't use directory traversal.

    Args:
        path: Path to validate.

    Raises:
        PathValidationError: If the path is unsafe or uses directory traversal.
    """
    try:
        # Resolve to absolute path to catch symlinks and .. traversal
        resolved = path.resolve()

        # Check for suspicious patterns in the original path
        path_str = str(path)
        suspicious_patterns = ["/../", "/..", "\\..\\", "\\.."]

        for pattern in suspicious_patterns:
            if pattern in path_str:
                raise PathValidationError(
                    f"Path contains directory traversal pattern: {path}"
                )

        # Ensure the resolved path is an absolute path
        if not resolved.is_absolute():
            raise PathValidationError(f"Path must be absolute: {path}")

    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Invalid path: {path}. Error: {e}") from e


def detect_checkpoint(model_id: str) -> CheckpointInfo:
    """Detect checkpoint format and resolve model path.

    Handles both local paths and HuggingFace Hub model IDs.
    Downloads from Hub if needed.
    """
    local_path = Path(model_id)

    if local_path.is_dir():
        _validate_path(local_path)
        return _inspect_local(local_path)

    # If it looks like a local path (exists or is absolute) but is not a directory, fail
    if local_path.exists():
        raise ModelLoadError(f"Expected a directory, got a file: {local_path}")

    # If it's an absolute path that doesn't exist, fail early
    # (Don't try to download absolute paths from HuggingFace Hub)
    if local_path.is_absolute():
        raise ModelNotFoundError(f"Path does not exist: {local_path}")

    # Try HuggingFace Hub
    return _download_and_inspect(model_id)


def _download_and_inspect(model_id: str) -> CheckpointInfo:
    """Download a model from HuggingFace Hub and inspect it."""
    max_retries = 3
    backoff_seconds = [1, 2, 4]  # Exponential backoff

    for attempt in range(max_retries):
        try:
            local_dir = snapshot_download(
                model_id,
                allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt"],
                ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "optimizer*", "training_args*"],
            )
            return _inspect_local(Path(local_dir))

        except RepositoryNotFoundError as e:
            # Don't retry on permanent errors
            raise ModelNotFoundError(
                f"Model '{model_id}' not found on HuggingFace Hub or as a local path."
            ) from e

        except HfHubHTTPError as e:
            # Check HfHubHTTPError BEFORE RequestException (since it inherits from it)
            # Retry on 5xx errors (server issues), but not 4xx (client errors)
            should_retry = False
            if hasattr(e, "response") and e.response is not None:
                status_code = e.response.status_code
                if 500 <= status_code < 600 and attempt < max_retries - 1:
                    should_retry = True
                    wait_time = backoff_seconds[attempt]
                    logger.warning(
                        "Server error %d downloading '%s' (attempt %d/%d). Retrying in %ds...",
                        status_code,
                        model_id,
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)

            if not should_retry:
                # Don't retry on client errors or final attempt
                raise ModelLoadError(f"Failed to download '{model_id}': {e}") from e

        except (
            ConnectionError,
            TimeoutError,
            requests.exceptions.RequestException,
        ) as e:
            # Retry on transient network errors
            if attempt < max_retries - 1:
                wait_time = backoff_seconds[attempt]
                logger.warning(
                    "Download failed for '%s' (attempt %d/%d): %s. Retrying in %ds...",
                    model_id,
                    attempt + 1,
                    max_retries,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                raise ModelLoadError(
                    f"Failed to download '{model_id}' after {max_retries} attempts: {e}"
                ) from e

    # This should never be reached, but for safety
    raise ModelLoadError(f"Failed to download '{model_id}' after {max_retries} attempts")


def _inspect_local(path: Path) -> CheckpointInfo:
    """Inspect a local model directory."""
    if not path.exists():
        raise ModelNotFoundError(f"Path does not exist: {path}")

    if not path.is_dir():
        raise ModelLoadError(f"Expected a directory, got a file: {path}")

    # Find weight files
    safetensor_files = sorted(path.glob("*.safetensors"))
    pytorch_files = sorted(path.glob("*.bin"))

    # Filter out non-model bin files
    pytorch_files = [
        f
        for f in pytorch_files
        if f.name not in ("training_args.bin", "optimizer.bin", "scheduler.bin")
    ]

    if safetensor_files:
        fmt = "safetensors"
        weight_files = safetensor_files
    elif pytorch_files:
        fmt = "pytorch"
        weight_files = pytorch_files
    else:
        raise ModelLoadError(
            f"No model weights found in {path}. "
            "Expected .safetensors or .bin files."
        )

    # Check for config
    config_path = path / "config.json"
    if not config_path.exists():
        raise ModelLoadError(f"No config.json found in {path}")

    # Parse config for model info
    with open(config_path) as f:
        config_data = json.load(f)

    # Check for tokenizer
    tokenizer_path = None
    for tok_file in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]:
        if (path / tok_file).exists():
            tokenizer_path = path / tok_file
            break

    # Check for LoRA adapter
    has_lora = (
        (path / "adapter_config.json").exists()
        or (path / "adapter_model.safetensors").exists()
    )

    info = CheckpointInfo(
        local_path=path,
        format=fmt,
        weight_files=weight_files,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        has_lora=has_lora,
        model_type=config_data.get("model_type"),
        num_hidden_layers=config_data.get("num_hidden_layers"),
        hidden_size=config_data.get("hidden_size"),
        num_attention_heads=config_data.get("num_attention_heads"),
        num_key_value_heads=config_data.get("num_key_value_heads"),
        intermediate_size=config_data.get("intermediate_size"),
        vocab_size=config_data.get("vocab_size"),
    )

    logger.info(
        "Detected %s checkpoint at %s: %d weight files, model_type=%s, %d layers",
        fmt,
        path,
        len(weight_files),
        info.model_type,
        info.num_hidden_layers or 0,
    )

    return info
