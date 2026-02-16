"""Afterburn exception hierarchy."""

from __future__ import annotations


class AfterburnError(Exception):
    """Base exception for all Afterburn errors."""


class ModelLoadError(AfterburnError):
    """Failed to load a model checkpoint."""


class ModelNotFoundError(ModelLoadError):
    """Model ID or path does not exist."""


class IncompatibleModelsError(AfterburnError):
    """Base and trained models have incompatible architectures."""


class OutOfMemoryError(AfterburnError):
    """Insufficient memory (GPU or CPU) to complete analysis."""


class PromptSuiteError(AfterburnError):
    """Error loading or validating a prompt suite."""


class ConfigError(AfterburnError):
    """Error in configuration file."""


class ReportGenerationError(AfterburnError):
    """Error generating a report."""


class PathValidationError(AfterburnError):
    """Invalid or unsafe path detected."""
