"""Afterburn: Post-training diagnostics for LLMs."""

from afterburn.version import __version__
from afterburn.diagnoser import Diagnoser
from afterburn.types import (
    DiagnosticReport,
    ModelPair,
    TrainingMethod,
    ReportFormat,
)

__all__ = [
    "__version__",
    "Diagnoser",
    "DiagnosticReport",
    "ModelPair",
    "TrainingMethod",
    "ReportFormat",
]
