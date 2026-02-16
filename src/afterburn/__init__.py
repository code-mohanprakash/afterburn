"""Afterburn: Post-training diagnostics for LLMs."""

from afterburn.diagnoser import Diagnoser
from afterburn.types import (
    ConfidenceInterval,
    DiagnosticReport,
    ModelPair,
    ReportFormat,
    TrainingMethod,
)
from afterburn.version import __version__

__all__ = [
    "__version__",
    "ConfidenceInterval",
    "Diagnoser",
    "DiagnosticReport",
    "ModelPair",
    "TrainingMethod",
    "ReportFormat",
]
