"""Lightweight activation-level model diffing.

A simpler alternative to crosscoders/SAEs: compares mean activation patterns
between base and trained models to identify which hidden dimensions shifted
most during training.

This reveals *functional* changes — which internal representations actually
changed when processing text — complementing the weight-level metrics that
show *structural* changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class DimensionShift:
    """A single hidden dimension that shifted significantly."""

    layer_index: int
    dimension_index: int
    base_mean_activation: float
    trained_mean_activation: float
    absolute_shift: float
    relative_shift: float


@dataclass
class ActivationDiffResult:
    """Results from activation-level model comparison."""

    # Per-layer summary: mean activation shift magnitude
    layer_mean_shifts: list[tuple[int, float]] = field(default_factory=list)
    # Top shifted dimensions across all layers
    top_shifted_dimensions: list[DimensionShift] = field(default_factory=list)
    # Overall activation divergence score (mean shift across all layers)
    mean_activation_divergence: float = 0.0
    # Layer with maximum shift
    most_changed_layer: int = -1
    # Number of layers analyzed
    num_layers_analyzed: int = 0


def compute_activation_diff(
    base_activations: dict[int, torch.Tensor],
    trained_activations: dict[int, torch.Tensor],
    top_k_dimensions: int = 20,
) -> ActivationDiffResult:
    """Compare mean activations between base and trained models.

    Args:
        base_activations: {layer_index: tensor of shape (num_samples, hidden_dim)}
            Mean activation per dimension across samples for each layer.
        trained_activations: Same format for the trained model.
        top_k_dimensions: Number of top shifted dimensions to report.

    Returns:
        ActivationDiffResult with per-layer and per-dimension analysis.
    """
    common_layers = sorted(set(base_activations.keys()) & set(trained_activations.keys()))

    if not common_layers:
        return ActivationDiffResult()

    layer_mean_shifts: list[tuple[int, float]] = []
    all_dim_shifts: list[DimensionShift] = []

    for layer_idx in common_layers:
        base_act = base_activations[layer_idx].float()
        trained_act = trained_activations[layer_idx].float()

        # Compute mean activation per dimension
        if base_act.dim() == 2:
            base_mean = base_act.mean(dim=0)
            trained_mean = trained_act.mean(dim=0)
        else:
            base_mean = base_act
            trained_mean = trained_act

        # Per-dimension shift
        diff = (trained_mean - base_mean).abs()
        mean_shift = diff.mean().item()
        layer_mean_shifts.append((layer_idx, mean_shift))

        # Base magnitudes for relative shift
        base_magnitude = base_mean.abs()

        # Track individual dimension shifts
        for dim_idx in range(diff.numel()):
            abs_shift = diff[dim_idx].item()
            base_mag = base_magnitude[dim_idx].item()
            rel_shift = abs_shift / max(base_mag, 1e-10)

            all_dim_shifts.append(DimensionShift(
                layer_index=layer_idx,
                dimension_index=dim_idx,
                base_mean_activation=base_mean[dim_idx].item(),
                trained_mean_activation=trained_mean[dim_idx].item(),
                absolute_shift=abs_shift,
                relative_shift=rel_shift,
            ))

    # Sort by absolute shift, take top-k
    all_dim_shifts.sort(key=lambda d: d.absolute_shift, reverse=True)
    top_shifted = all_dim_shifts[:top_k_dimensions]

    # Overall metrics
    mean_divergence = sum(s for _, s in layer_mean_shifts) / len(layer_mean_shifts)
    most_changed = max(layer_mean_shifts, key=lambda x: x[1])[0]

    return ActivationDiffResult(
        layer_mean_shifts=layer_mean_shifts,
        top_shifted_dimensions=top_shifted,
        mean_activation_divergence=mean_divergence,
        most_changed_layer=most_changed,
        num_layers_analyzed=len(common_layers),
    )
