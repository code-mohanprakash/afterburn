"""Checkpoint format detection and validation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from afterburn.exceptions import ModelLoadError, ModelNotFoundError

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


def detect_checkpoint(model_id: str) -> CheckpointInfo:
    """Detect checkpoint format and resolve model path.

    Handles both local paths and HuggingFace Hub model IDs.
    Downloads from Hub if needed.
    """
    local_path = Path(model_id)

    if local_path.is_dir():
        return _inspect_local(local_path)

    # Try HuggingFace Hub
    return _download_and_inspect(model_id)


def _download_and_inspect(model_id: str) -> CheckpointInfo:
    """Download a model from HuggingFace Hub and inspect it."""
    try:
        local_dir = snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt"],
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "optimizer*", "training_args*"],
        )
    except RepositoryNotFoundError:
        raise ModelNotFoundError(
            f"Model '{model_id}' not found on HuggingFace Hub or as a local path."
        )
    except HfHubHTTPError as e:
        raise ModelLoadError(f"Failed to download '{model_id}': {e}") from e

    return _inspect_local(Path(local_dir))


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
    has_lora = (path / "adapter_config.json").exists() or (path / "adapter_model.safetensors").exists()

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
