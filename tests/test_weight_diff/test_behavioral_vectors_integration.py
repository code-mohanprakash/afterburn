"""Integration tests for behavioral vectors in the full engine."""

import torch

from afterburn.types import BehavioralVector
from afterburn.weight_diff.metrics import compute_direction_coherence, svd_analysis


class TestBehavioralVectorsIntegration:
    """Test the full behavioral vector pipeline."""

    def test_svd_returns_vectors_with_flag(self):
        """Test that SVD analysis returns vectors when requested."""
        base = torch.randn(128, 64)
        trained = base + 0.05 * torch.randn(128, 64)

        result = svd_analysis(base, trained, return_vectors=True, top_k=5)

        assert result is not None
        assert result.top_right_vectors is not None
        assert result.top_left_vectors is not None
        assert len(result.top_right_vectors) == 5
        # top_left_vectors is the full U matrix with 128 rows
        assert len(result.top_left_vectors) == 128

    def test_coherence_with_rank1_updates(self):
        """Test coherence when all layers have rank-1 updates in same direction."""
        # Create a consistent direction vector
        direction = torch.randn(64)
        direction = direction / direction.norm()

        # Create SVD results for multiple layers with same direction
        layer_vectors = {}
        for i in range(5):
            base = torch.randn(128, 64)
            # Add rank-1 perturbation in the same direction
            update = torch.randn(128, 1) @ direction.unsqueeze(0)
            trained = base + 0.5 * update

            result = svd_analysis(base, trained, return_vectors=True)
            assert result is not None
            assert result.top_right_vectors is not None
            layer_vectors[f"layer_{i}"] = result.top_right_vectors

        # Coherence should be high since all updates are in same direction
        coherence = compute_direction_coherence(layer_vectors)
        assert coherence > 0.7, f"Expected high coherence, got {coherence}"

    def test_coherence_with_random_updates(self):
        """Test coherence with random independent updates."""
        layer_vectors = {}
        for i in range(5):
            base = torch.randn(128, 64)
            trained = base + 0.1 * torch.randn(128, 64)

            result = svd_analysis(base, trained, return_vectors=True)
            assert result is not None
            assert result.top_right_vectors is not None
            layer_vectors[f"layer_{i}"] = result.top_right_vectors

        # Coherence should be lower for random updates
        coherence = compute_direction_coherence(layer_vectors)
        assert 0.0 <= coherence <= 1.0

    def test_behavioral_vector_construction_from_svd(self):
        """Test creating BehavioralVectors from SVD results."""
        base = torch.randn(128, 64)
        trained = base + 0.1 * torch.randn(128, 64)

        result = svd_analysis(base, trained, return_vectors=True, top_k=3)
        assert result is not None

        # Compute total energy
        total_energy = sum(sv ** 2 for sv in result.top_singular_values)

        # Create BehavioralVector objects
        behavioral_vectors = []
        for idx, sv in enumerate(result.top_singular_values[:3]):
            explained_var = (sv ** 2) / total_energy
            bv = BehavioralVector(
                layer_name="test_layer",
                singular_value=sv,
                direction_index=idx,
                explained_variance_ratio=explained_var,
            )
            behavioral_vectors.append(bv)

        # Verify
        assert len(behavioral_vectors) == 3
        assert behavioral_vectors[0].direction_index == 0
        assert behavioral_vectors[1].direction_index == 1
        assert behavioral_vectors[2].direction_index == 2

        # Explained variance ratios should sum to close to 1.0 (might be less if there are more SVs)
        total_explained = sum(bv.explained_variance_ratio for bv in behavioral_vectors)
        assert 0.0 < total_explained <= 1.0

    def test_explained_variance_ordering(self):
        """Test that explained variance ratios decrease monotonically."""
        base = torch.randn(128, 64)
        # Create a rank-deficient update for clearer structure
        u = torch.randn(128, 3)
        v = torch.randn(3, 64)
        trained = base + 0.5 * (u @ v)

        result = svd_analysis(base, trained, return_vectors=True, top_k=10)
        assert result is not None

        total_energy = sum(sv ** 2 for sv in result.top_singular_values)
        explained_vars = [(sv ** 2) / total_energy for sv in result.top_singular_values]

        # Should be in descending order
        for i in range(len(explained_vars) - 1):
            assert explained_vars[i] >= explained_vars[i + 1], \
                f"Explained variance not monotonic: {explained_vars}"

    def test_vector_orthogonality(self):
        """Right singular vectors should be orthogonal."""
        base = torch.randn(128, 64)
        trained = base + 0.1 * torch.randn(128, 64)

        result = svd_analysis(base, trained, return_vectors=True, top_k=5)
        assert result is not None
        assert result.top_right_vectors is not None

        # Check pairwise orthogonality
        vectors = result.top_right_vectors
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dot_product = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=False))
                # Should be close to zero (orthogonal)
                assert abs(dot_product) < 0.1, \
                    f"Vectors {i} and {j} not orthogonal: dot={dot_product}"

    def test_coherence_with_opposing_directions(self):
        """Test coherence when half layers go one way, half the other."""
        direction = torch.randn(64)
        direction = direction / direction.norm()

        layer_vectors = {}
        for i in range(6):
            base = torch.randn(128, 64)
            # First 3 layers: positive direction, last 3: negative direction
            sign = 1.0 if i < 3 else -1.0
            update = torch.randn(128, 1) @ (sign * direction).unsqueeze(0)
            trained = base + 0.5 * update

            result = svd_analysis(base, trained, return_vectors=True)
            assert result is not None
            layer_vectors[f"layer_{i}"] = result.top_right_vectors

        # Coherence should still be high due to absolute value in computation
        coherence = compute_direction_coherence(layer_vectors)
        assert coherence > 0.7, f"Expected high coherence with opposing directions, got {coherence}"
