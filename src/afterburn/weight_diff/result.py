"""Weight diff result types are defined in afterburn.types.

This module re-exports them for convenience.
"""

from afterburn.types import (
    AttentionHeadScore,
    EmbeddingDrift,
    LayerDiff,
    LayerNormShift,
    WeightDiffResult,
)

__all__ = [
    "AttentionHeadScore",
    "EmbeddingDrift",
    "LayerDiff",
    "LayerNormShift",
    "WeightDiffResult",
]
