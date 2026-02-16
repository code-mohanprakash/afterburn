"""GPU/CPU auto-detection and memory management."""

from __future__ import annotations

import atexit
import logging
import platform
import shutil
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from afterburn.exceptions import OutOfMemoryError

logger = logging.getLogger(__name__)

# Cleanup registry for graceful shutdown
_cleanup_callbacks: list[Callable[[], None]] = []


def register_cleanup(callback: Callable[[], None]) -> None:
    """Register a cleanup callback for graceful shutdown."""
    _cleanup_callbacks.append(callback)


def _cleanup_handler(signum: object = None, frame: object = None) -> None:
    """Run all cleanup callbacks."""
    import contextlib

    for cb in _cleanup_callbacks:
        with contextlib.suppress(Exception):
            cb()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Register at module load
atexit.register(_cleanup_handler)


def _sigint_handler(s: object, f: object) -> None:
    _cleanup_handler(s, f)


signal.signal(signal.SIGINT, _sigint_handler)


def check_disk_space(path: Path, required_gb: float) -> None:
    """Check if sufficient disk space is available.

    Raises OutOfMemoryError if insufficient space.
    """
    try:
        usage = shutil.disk_usage(path)
        available_gb = usage.free / (1024**3)

        if available_gb < required_gb:
            raise OutOfMemoryError(
                f"Insufficient disk space at {path}: "
                f"required {required_gb:.2f} GB, available {available_gb:.2f} GB"
            )

        logger.debug(
            "Disk space check passed: %.2f GB available (%.2f GB required)",
            available_gb,
            required_gb,
        )
    except OSError as e:
        raise OutOfMemoryError(f"Failed to check disk space at {path}: {e}") from e


@dataclass
class DeviceConfig:
    """Configuration for compute device."""

    device: torch.device
    dtype: torch.dtype
    max_memory_gb: float
    gpu_name: str | None = None

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"


def auto_detect_device(
    prefer_gpu: bool = True,
    max_memory_fraction: float = 0.85,
    force_device: str | None = None,
) -> DeviceConfig:
    """Auto-detect the best available device and memory limits.

    Args:
        prefer_gpu: Whether to prefer GPU over CPU.
        max_memory_fraction: Fraction of available GPU memory to use.
        force_device: Force a specific device ("cuda", "mps", "cpu").

    Returns:
        DeviceConfig with the best available device settings.
    """
    if force_device:
        return _build_config(force_device, max_memory_fraction)

    if prefer_gpu and torch.cuda.is_available():
        return _build_config("cuda", max_memory_fraction)

    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return _build_config("mps", max_memory_fraction)

    return _build_config("cpu", max_memory_fraction)


def _build_config(device_type: str, max_memory_fraction: float) -> DeviceConfig:
    if device_type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        max_gb = total_gb * max_memory_fraction
        gpu_name = props.name
        dtype = torch.float16
        logger.info(
            "Using CUDA device: %s (%.1f GB available, using %.1f GB)",
            gpu_name,
            total_gb,
            max_gb,
        )
        return DeviceConfig(
            device=torch.device("cuda"),
            dtype=dtype,
            max_memory_gb=max_gb,
            gpu_name=gpu_name,
        )

    if device_type == "mps":
        # MPS doesn't expose memory info easily; use a conservative default
        import psutil

        total_gb = psutil.virtual_memory().total / (1024**3)
        max_gb = total_gb * 0.5  # conservative for MPS
        logger.info("Using MPS device (%.1f GB system RAM)", total_gb)
        return DeviceConfig(
            device=torch.device("mps"),
            dtype=torch.float32,  # MPS float16 support varies
            max_memory_gb=max_gb,
            gpu_name="Apple MPS",
        )

    # CPU fallback
    try:
        import psutil

        total_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback for systems without psutil
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            total_gb = int(result.stdout.strip()) / (1024**3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        total_gb = int(line.split()[1]) / (1024**2)
                        break
                else:
                    total_gb = 8.0  # safe default

    max_gb = total_gb * max_memory_fraction
    logger.info("Using CPU device (%.1f GB RAM available, using %.1f GB)", total_gb, max_gb)
    return DeviceConfig(
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_memory_gb=max_gb,
        gpu_name=None,
    )


def estimate_model_memory_gb(num_params: int, dtype: torch.dtype) -> float:
    """Estimate memory needed to load a model."""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }
    bpp = bytes_per_param.get(dtype, 4)
    return (num_params * bpp) / (1024**3)
