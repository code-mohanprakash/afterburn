"""Tests for SVD weight diff analysis."""

import torch

from afterburn.weight_diff.metrics import SVDResult, svd_analysis


class TestSVDAnalysis:
    def test_returns_none_for_1d(self):
        """SVD not meaningful for bias vectors."""
        base = torch.randn(100)
        trained = torch.randn(100)
        assert svd_analysis(base, trained) is None

    def test_returns_result_for_2d(self):
        base = torch.randn(64, 64)
        trained = base + torch.randn(64, 64) * 0.1
        result = svd_analysis(base, trained)
        assert result is not None
        assert isinstance(result, SVDResult)

    def test_top_singular_values_sorted_descending(self):
        base = torch.randn(32, 32)
        trained = base + torch.randn(32, 32) * 0.5
        result = svd_analysis(base, trained, top_k=5)
        assert result is not None
        svs = result.top_singular_values
        assert len(svs) <= 5
        for i in range(len(svs) - 1):
            assert svs[i] >= svs[i + 1] - 1e-5

    def test_effective_rank_positive(self):
        base = torch.randn(32, 32)
        trained = base + torch.randn(32, 32) * 0.1
        result = svd_analysis(base, trained)
        assert result is not None
        assert result.effective_rank >= 1

    def test_concentration_ratio_bounds(self):
        base = torch.randn(32, 32)
        trained = base + torch.randn(32, 32) * 0.1
        result = svd_analysis(base, trained)
        assert result is not None
        assert 0.0 <= result.concentration_ratio <= 1.0

    def test_stable_rank_bounds(self):
        """Stable rank >= 1 (equals 1 when all energy in one direction)."""
        base = torch.randn(32, 32)
        trained = base + torch.randn(32, 32) * 0.1
        result = svd_analysis(base, trained)
        assert result is not None
        assert result.stable_rank >= 1.0 - 1e-5

    def test_rank1_perturbation_has_high_concentration(self):
        """A rank-1 perturbation should have most energy in top singular value."""
        base = torch.randn(64, 64)
        # Rank-1 perturbation: outer product of two vectors
        u = torch.randn(64, 1)
        v = torch.randn(1, 64)
        trained = base + u @ v
        result = svd_analysis(base, trained, top_k=10)
        assert result is not None
        # Top SV should dominate — concentration should be high
        assert result.concentration_ratio > 0.5
        # Effective rank should be low (1 or 2 due to numerical noise)
        assert result.effective_rank <= 3

    def test_diffuse_perturbation_has_low_concentration(self):
        """Random noise perturbation should spread across many directions."""
        base = torch.randn(64, 64)
        trained = base + torch.randn(64, 64) * 0.5
        result = svd_analysis(base, trained, top_k=10)
        assert result is not None
        # Concentration should be lower for diffuse changes
        assert result.concentration_ratio < 0.5
        # Effective rank should be higher
        assert result.effective_rank > 5

    def test_identical_tensors(self):
        """Identical tensors should have zero singular values."""
        t = torch.randn(32, 32)
        result = svd_analysis(t, t, top_k=5)
        assert result is not None
        # All singular values should be ~0
        for sv in result.top_singular_values:
            assert abs(sv) < 1e-4

    def test_3d_tensor(self):
        """Should handle 3D+ tensors by reshaping to 2D."""
        base = torch.randn(8, 16, 32)
        trained = base + torch.randn(8, 16, 32) * 0.1
        result = svd_analysis(base, trained)
        assert result is not None
        assert len(result.top_singular_values) > 0

    def test_top_k_limits_output(self):
        base = torch.randn(64, 64)
        trained = base + torch.randn(64, 64) * 0.1
        result = svd_analysis(base, trained, top_k=3)
        assert result is not None
        assert len(result.top_singular_values) == 3

    def test_energy_threshold(self):
        """Higher threshold should require more singular values."""
        base = torch.randn(64, 64)
        trained = base + torch.randn(64, 64) * 0.5
        r90 = svd_analysis(base, trained, energy_threshold=0.9)
        r99 = svd_analysis(base, trained, energy_threshold=0.99)
        assert r90 is not None and r99 is not None
        assert r99.effective_rank >= r90.effective_rank
