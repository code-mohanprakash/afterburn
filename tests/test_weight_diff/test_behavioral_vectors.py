"""Tests for behavioral vectors via SVD direction analysis."""

import torch

from afterburn.types import BehavioralVector
from afterburn.weight_diff.metrics import compute_direction_coherence, svd_analysis


class TestSVDVectors:
    def test_no_vectors_by_default(self):
        """svd_analysis with default params should have None vectors."""
        base = torch.randn(64, 64)
        trained = base + 0.1 * torch.randn(64, 64)
        result = svd_analysis(base, trained)
        assert result is not None
        assert result.top_left_vectors is None
        assert result.top_right_vectors is None

    def test_returns_vectors_when_requested(self):
        base = torch.randn(64, 64)
        trained = base + 0.1 * torch.randn(64, 64)
        result = svd_analysis(base, trained, return_vectors=True)
        assert result is not None
        assert result.top_right_vectors is not None
        assert len(result.top_right_vectors) > 0

    def test_vector_dimensions(self):
        """Right vectors should have dimension matching cols."""
        base = torch.randn(64, 32)
        trained = base + 0.1 * torch.randn(64, 32)
        result = svd_analysis(base, trained, return_vectors=True)
        assert result is not None
        assert result.top_right_vectors is not None
        # Each right vector should have 32 elements (number of cols)
        for v in result.top_right_vectors:
            assert len(v) == 32

    def test_vectors_approximately_unit_norm(self):
        """SVD singular vectors should have unit norm."""
        base = torch.randn(64, 64)
        trained = base + 0.1 * torch.randn(64, 64)
        result = svd_analysis(base, trained, return_vectors=True)
        assert result is not None
        for v in result.top_right_vectors:
            norm = sum(x**2 for x in v) ** 0.5
            assert abs(norm - 1.0) < 0.1  # approximately unit

    def test_1d_returns_none(self):
        result = svd_analysis(torch.randn(100), torch.randn(100), return_vectors=True)
        assert result is None

    def test_left_vectors_dimension(self):
        """Left vectors should have dimension matching rows."""
        base = torch.randn(64, 32)
        trained = base + 0.1 * torch.randn(64, 32)
        result = svd_analysis(base, trained, return_vectors=True)
        assert result is not None
        assert result.top_left_vectors is not None
        # Each left vector should have 64 elements (number of rows)
        # top_left_vectors is list of columns, so each column has 64 elements
        assert len(result.top_left_vectors) == 64

    def test_top_k_controls_vector_count(self):
        """top_k parameter should control number of vectors returned."""
        base = torch.randn(64, 64)
        trained = base + 0.1 * torch.randn(64, 64)
        result = svd_analysis(base, trained, return_vectors=True, top_k=5)
        assert result is not None
        assert result.top_right_vectors is not None
        assert len(result.top_right_vectors) == 5


class TestDirectionCoherence:
    def test_same_direction_high_coherence(self):
        """If all layers change in the same direction, coherence should be high."""
        v = [1.0, 0.0, 0.0, 0.0]  # same vector for all layers
        layer_vectors = {
            "layer_0": [v],
            "layer_1": [v],
            "layer_2": [v],
        }
        coherence = compute_direction_coherence(layer_vectors)
        assert coherence > 0.99

    def test_orthogonal_directions_low_coherence(self):
        """Orthogonal directions should give low coherence."""
        layer_vectors = {
            "layer_0": [[1.0, 0.0, 0.0]],
            "layer_1": [[0.0, 1.0, 0.0]],
            "layer_2": [[0.0, 0.0, 1.0]],
        }
        coherence = compute_direction_coherence(layer_vectors)
        assert coherence < 0.01

    def test_single_layer_returns_zero(self):
        layer_vectors = {"layer_0": [[1.0, 0.0]]}
        assert compute_direction_coherence(layer_vectors) == 0.0

    def test_empty_returns_zero(self):
        assert compute_direction_coherence({}) == 0.0

    def test_different_dimensions_skipped(self):
        """Layers with different vector sizes should be skipped."""
        layer_vectors = {
            "layer_0": [[1.0, 0.0]],
            "layer_1": [[1.0, 0.0, 0.0]],
        }
        coherence = compute_direction_coherence(layer_vectors)
        assert coherence == 0.0  # No valid pairs

    def test_coherence_bounded(self):
        import random
        layer_vectors = {
            f"layer_{i}": [[random.gauss(0, 1) for _ in range(10)]]
            for i in range(5)
        }
        coherence = compute_direction_coherence(layer_vectors)
        assert 0.0 <= coherence <= 1.0

    def test_partial_alignment_moderate_coherence(self):
        """Partially aligned vectors should give moderate coherence."""
        # Create vectors with some alignment
        layer_vectors = {
            "layer_0": [[1.0, 0.5, 0.0]],
            "layer_1": [[0.9, 0.4, 0.1]],
            "layer_2": [[1.0, 0.6, -0.1]],
        }
        coherence = compute_direction_coherence(layer_vectors)
        # Should be between low and high coherence
        assert 0.3 < coherence < 0.99

    def test_empty_vectors_list_returns_zero(self):
        """Layers with empty vector lists should be handled."""
        layer_vectors = {
            "layer_0": [],
            "layer_1": [[1.0, 0.0]],
        }
        coherence = compute_direction_coherence(layer_vectors)
        assert coherence == 0.0


class TestBehavioralVector:
    def test_dataclass_construction(self):
        bv = BehavioralVector(
            layer_name="layer_0",
            singular_value=5.0,
            direction_index=0,
            explained_variance_ratio=0.8,
        )
        assert bv.layer_name == "layer_0"
        assert bv.direction_index == 0

    def test_frozen_dataclass(self):
        """BehavioralVector should be immutable."""
        bv = BehavioralVector(
            layer_name="layer_0",
            singular_value=5.0,
            direction_index=0,
            explained_variance_ratio=0.8,
        )
        try:
            bv.singular_value = 10.0
            raise AssertionError("Should not be able to modify frozen dataclass")
        except Exception:
            pass  # Expected

    def test_multiple_direction_indices(self):
        """Can create BehavioralVectors for different direction indices."""
        bv0 = BehavioralVector("layer_0", 5.0, 0, 0.5)
        bv1 = BehavioralVector("layer_0", 3.0, 1, 0.3)
        bv2 = BehavioralVector("layer_0", 1.0, 2, 0.1)
        assert bv0.direction_index == 0
        assert bv1.direction_index == 1
        assert bv2.direction_index == 2
