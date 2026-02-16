"""Tests for lightweight activation-level model diffing."""

import pytest
import torch

from afterburn.weight_diff.activation_diff import (
    DimensionShift,
    compute_activation_diff,
)


class TestComputeActivationDiff:
    def test_identical_activations(self):
        """Identical activations should have zero divergence."""
        act = {0: torch.randn(10, 64), 1: torch.randn(10, 64)}
        result = compute_activation_diff(act, act)
        assert result.mean_activation_divergence == pytest.approx(0.0, abs=1e-5)

    def test_different_activations(self):
        """Different activations should have positive divergence."""
        base = {0: torch.zeros(10, 64), 1: torch.zeros(10, 64)}
        trained = {0: torch.ones(10, 64), 1: torch.ones(10, 64) * 2}
        result = compute_activation_diff(base, trained)
        assert result.mean_activation_divergence > 0

    def test_layer_ordering(self):
        base = {0: torch.zeros(10, 32), 1: torch.zeros(10, 32), 2: torch.zeros(10, 32)}
        trained = {
            0: torch.randn(10, 32) * 0.1,
            1: torch.randn(10, 32) * 1.0,  # Layer 1 changes most
            2: torch.randn(10, 32) * 0.5,
        }
        result = compute_activation_diff(base, trained)
        assert result.most_changed_layer == 1

    def test_top_k_dimensions(self):
        base = {0: torch.zeros(10, 128)}
        trained = {0: torch.randn(10, 128)}
        result = compute_activation_diff(base, trained, top_k_dimensions=5)
        assert len(result.top_shifted_dimensions) == 5

    def test_top_dimensions_sorted(self):
        """Top dimensions should be sorted by absolute shift (descending)."""
        base = {0: torch.zeros(10, 64)}
        trained = {0: torch.randn(10, 64)}
        result = compute_activation_diff(base, trained, top_k_dimensions=10)
        shifts = [d.absolute_shift for d in result.top_shifted_dimensions]
        for i in range(len(shifts) - 1):
            assert shifts[i] >= shifts[i + 1]

    def test_num_layers_analyzed(self):
        base = {0: torch.randn(5, 32), 1: torch.randn(5, 32), 2: torch.randn(5, 32)}
        trained = {0: torch.randn(5, 32), 1: torch.randn(5, 32), 2: torch.randn(5, 32)}
        result = compute_activation_diff(base, trained)
        assert result.num_layers_analyzed == 3

    def test_empty_inputs(self):
        result = compute_activation_diff({}, {})
        assert result.num_layers_analyzed == 0
        assert result.mean_activation_divergence == 0.0

    def test_partial_overlap(self):
        """Only common layers should be analyzed."""
        base = {0: torch.randn(5, 32), 1: torch.randn(5, 32)}
        trained = {1: torch.randn(5, 32), 2: torch.randn(5, 32)}
        result = compute_activation_diff(base, trained)
        assert result.num_layers_analyzed == 1  # Only layer 1

    def test_1d_activations(self):
        """Should handle pre-averaged 1D activation vectors."""
        base = {0: torch.zeros(64)}
        trained = {0: torch.ones(64)}
        result = compute_activation_diff(base, trained)
        assert result.mean_activation_divergence > 0

    def test_dimension_shift_fields(self):
        base = {0: torch.zeros(5, 16)}
        trained = {0: torch.ones(5, 16)}
        result = compute_activation_diff(base, trained, top_k_dimensions=3)
        for dim in result.top_shifted_dimensions:
            assert isinstance(dim, DimensionShift)
            assert dim.layer_index == 0
            assert dim.absolute_shift > 0
            assert dim.relative_shift >= 0

    def test_concentrated_change(self):
        """Change in few dimensions should show those at the top."""
        base = {0: torch.zeros(10, 100)}
        trained_data = torch.zeros(10, 100)
        trained_data[:, 42] = 10.0  # Big change in dimension 42
        trained_data[:, 77] = 5.0   # Smaller change in dimension 77
        trained = {0: trained_data}
        result = compute_activation_diff(base, trained, top_k_dimensions=5)
        top_dims = [d.dimension_index for d in result.top_shifted_dimensions[:2]]
        assert 42 in top_dims
        assert 77 in top_dims
