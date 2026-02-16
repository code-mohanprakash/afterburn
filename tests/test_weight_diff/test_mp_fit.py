"""Tests for Marchenko-Pastur law fitting in spectral analysis."""

import torch

from afterburn.weight_diff.spectral import marchenko_pastur_fit


def test_returns_none_for_1d():
    """1D tensor should return None."""
    vec = torch.randn(100)
    result = marchenko_pastur_fit(vec)
    assert result is None


def test_returns_none_for_small_matrix():
    """Matrix smaller than min_size should return None."""
    mat = torch.randn(30, 30)
    result = marchenko_pastur_fit(mat, min_size=50)
    assert result is None


def test_returns_result_for_large_matrix():
    """128x128 random matrix should return MPResult."""
    torch.manual_seed(42)
    mat = torch.randn(128, 128)
    result = marchenko_pastur_fit(mat)
    assert result is not None
    assert result.sigma_sq > 0
    assert result.lambda_plus > result.lambda_minus
    assert result.gamma > 0


def test_random_matrix_high_bulk_fraction():
    """Random Gaussian matrix should have bulk_fraction > 0.85."""
    torch.manual_seed(42)
    mat = torch.randn(128, 128) * 0.1  # Small variance for stability
    result = marchenko_pastur_fit(mat)
    assert result is not None
    assert result.bulk_fraction > 0.85, f"bulk_fraction={result.bulk_fraction} should be > 0.85 for random matrix"


def test_random_matrix_few_spikes():
    """Random Gaussian should have few or no spikes (num_spikes <= 5 for 128x128)."""
    torch.manual_seed(42)
    mat = torch.randn(128, 128) * 0.1
    result = marchenko_pastur_fit(mat)
    assert result is not None
    assert result.num_spikes <= 5, f"Random matrix has {result.num_spikes} spikes, expected <= 5"


def test_random_matrix_low_kl():
    """Random Gaussian should have low KL divergence (< 1.0)."""
    torch.manual_seed(42)
    mat = torch.randn(128, 128) * 0.1
    result = marchenko_pastur_fit(mat)
    assert result is not None
    assert result.kl_divergence < 1.0, f"KL divergence={result.kl_divergence} should be < 1.0 for random matrix"


def test_identity_many_spikes():
    """Scaled identity matrix with large entries should show high spike count or low bulk fraction.

    The MP law is designed for random matrices. A scaled identity deviates from randomness,
    which manifests as either spikes or a bulk distribution that doesn't match MP theory.
    """
    # Use a scaled identity with larger values to make it clearly non-random
    mat = torch.eye(128) * 10.0
    result = marchenko_pastur_fit(mat)
    assert result is not None
    # Identity matrix should show deviation from MP law via:
    # - Either many spikes (if eigenvalues are above lambda_plus)
    # - Or low bulk_fraction (if MP edges don't capture the distribution)
    # - Or high KL divergence (distribution doesn't match MP)
    assert (result.num_spikes > 50) or (result.bulk_fraction < 0.5) or (result.kl_divergence > 2.0), \
        f"Scaled identity should deviate from MP law: spikes={result.num_spikes}, " \
        f"bulk_fraction={result.bulk_fraction}, kl={result.kl_divergence}"


def test_random_plus_spike():
    """torch.randn(128,128) + 10*rank1 should have at least 1 spike above lambda_plus."""
    torch.manual_seed(42)
    mat = torch.randn(128, 128) * 0.1
    # Add a rank-1 spike
    u = torch.randn(128, 1)
    v = torch.randn(1, 128)
    mat = mat + 10.0 * (u @ v)

    result = marchenko_pastur_fit(mat)
    assert result is not None
    assert result.num_spikes >= 1, f"Matrix with rank-1 spike should have >= 1 spike, got {result.num_spikes}"


def test_lambda_bounds():
    """lambda_minus >= 0 and lambda_plus > lambda_minus for any valid result."""
    torch.manual_seed(42)
    mat = torch.randn(128, 128)
    result = marchenko_pastur_fit(mat)
    assert result is not None
    assert result.lambda_minus >= 0, "lambda_minus should be >= 0"
    assert result.lambda_plus > result.lambda_minus, "lambda_plus should be > lambda_minus"


def test_gamma_nonsquare():
    """200x100 matrix should have gamma = 2.0."""
    torch.manual_seed(42)
    mat = torch.randn(200, 100)
    result = marchenko_pastur_fit(mat)
    assert result is not None
    assert abs(result.gamma - 2.0) < 1e-6, f"Expected gamma=2.0, got {result.gamma}"


def test_sigma_sq_positive():
    """sigma_sq should always be > 0 for any valid result."""
    torch.manual_seed(42)
    # Test with various matrices
    for shape in [(64, 64), (128, 64), (100, 150)]:
        mat = torch.randn(*shape)
        result = marchenko_pastur_fit(mat)
        assert result is not None
        assert result.sigma_sq > 0, f"sigma_sq should be > 0, got {result.sigma_sq}"


def test_zero_matrix_returns_none():
    """Zero matrix should return None (all singular values 0)."""
    mat = torch.zeros(128, 128)
    result = marchenko_pastur_fit(mat)
    assert result is None


def test_bulk_fraction_range():
    """0 <= bulk_fraction <= 1 for any valid result."""
    torch.manual_seed(42)
    # Test with various matrices
    test_matrices = [
        torch.randn(64, 64),
        torch.randn(128, 64),
        torch.eye(100),
        torch.randn(100, 100) * 10.0,
    ]

    for mat in test_matrices:
        result = marchenko_pastur_fit(mat)
        if result is not None:
            assert 0.0 <= result.bulk_fraction <= 1.0, \
                f"bulk_fraction={result.bulk_fraction} should be in [0, 1]"


def test_spike_eigenvalues_above_lambda_plus():
    """All spike_eigenvalues should be > lambda_plus."""
    torch.manual_seed(42)
    mat = torch.randn(128, 128) * 0.1
    # Add some spikes
    u = torch.randn(128, 1)
    v = torch.randn(1, 128)
    mat = mat + 5.0 * (u @ v)

    result = marchenko_pastur_fit(mat)
    assert result is not None

    if result.num_spikes > 0:
        for spike in result.spike_eigenvalues:
            assert spike > result.lambda_plus, \
                f"Spike eigenvalue {spike} should be > lambda_plus {result.lambda_plus}"


def test_kl_divergence_non_negative():
    """KL divergence should be >= 0 for any valid result."""
    torch.manual_seed(42)
    # Test with various matrices
    test_matrices = [
        torch.randn(64, 64),
        torch.randn(128, 64),
        torch.randn(100, 100) * 0.1,
        torch.randn(80, 120),
    ]

    for mat in test_matrices:
        result = marchenko_pastur_fit(mat)
        if result is not None:
            assert result.kl_divergence >= 0.0, \
                f"KL divergence={result.kl_divergence} should be >= 0"
