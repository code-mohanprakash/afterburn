"""Tests for spectral analysis (alpha metric, stable rank)."""

import pytest
import torch

from afterburn.weight_diff.spectral import SpectralResult, spectral_analysis, _fit_power_law_alpha


class TestSpectralAnalysis:
    def test_returns_none_for_1d(self):
        assert spectral_analysis(torch.randn(100)) is None

    def test_returns_none_for_small_matrix(self):
        assert spectral_analysis(torch.randn(10, 10), min_size=50) is None

    def test_returns_result_for_large_matrix(self):
        result = spectral_analysis(torch.randn(128, 128))
        assert result is not None
        assert isinstance(result, SpectralResult)

    def test_alpha_positive(self):
        result = spectral_analysis(torch.randn(128, 128))
        assert result is not None
        assert result.alpha > 0

    def test_alpha_quality_assignment(self):
        result = spectral_analysis(torch.randn(128, 128))
        assert result is not None
        assert result.alpha_quality in ("good", "fair", "poor", "unstable")

    def test_stable_rank_positive(self):
        result = spectral_analysis(torch.randn(128, 128))
        assert result is not None
        assert result.stable_rank >= 1.0 - 1e-5

    def test_spectral_norm_positive(self):
        result = spectral_analysis(torch.randn(128, 128))
        assert result is not None
        assert result.spectral_norm > 0

    def test_identity_like_matrix(self):
        """Identity matrix should have stable_rank close to min(m,n)."""
        eye = torch.eye(64)
        result = spectral_analysis(eye)
        assert result is not None
        # All singular values = 1, so stable_rank = n
        assert result.stable_rank == pytest.approx(64.0, rel=0.01)

    def test_rank1_matrix_low_stable_rank(self):
        """Rank-1 matrix should have stable_rank close to 1."""
        u = torch.randn(128, 1)
        v = torch.randn(1, 128)
        mat = u @ v
        result = spectral_analysis(mat)
        assert result is not None
        assert result.stable_rank < 2.0

    def test_3d_tensor(self):
        """Should reshape 3D+ tensors to 2D."""
        result = spectral_analysis(torch.randn(64, 64, 2))
        assert result is not None

    def test_num_eigenvalues(self):
        result = spectral_analysis(torch.randn(100, 64))
        assert result is not None
        assert result.num_eigenvalues == 64  # min(m, n)

    def test_zero_matrix_returns_none(self):
        result = spectral_analysis(torch.zeros(128, 128))
        assert result is None


class TestFitPowerLawAlpha:
    def test_too_few_eigenvalues(self):
        eigs = torch.tensor([1.0, 0.5, 0.1])
        assert _fit_power_law_alpha(eigs) == 0.0

    def test_positive_eigenvalues(self):
        # Synthetic eigenvalue spectrum that follows a power law
        indices = torch.arange(1, 201, dtype=torch.float32)
        eigs = 1.0 / (indices ** 0.5)  # Known power law
        alpha = _fit_power_law_alpha(eigs)
        assert alpha > 0

    def test_alpha_bounds(self):
        eigs = torch.randn(100).abs()
        alpha = _fit_power_law_alpha(eigs)
        assert 0.0 <= alpha <= 20.0
