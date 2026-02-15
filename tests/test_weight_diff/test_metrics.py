"""Tests for weight diff metrics."""

import math

import pytest
import torch

from afterburn.weight_diff.metrics import (
    cosine_similarity,
    frobenius_norm_diff,
    l2_norm_diff,
    max_abs_diff,
    mean_abs_diff,
    relative_change,
)


class TestL2NormDiff:
    def test_identical_tensors(self):
        t = torch.randn(10, 10)
        assert l2_norm_diff(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_known_diff(self):
        base = torch.zeros(3)
        trained = torch.tensor([3.0, 4.0, 0.0])
        assert l2_norm_diff(base, trained) == pytest.approx(5.0, abs=1e-5)

    def test_symmetry(self):
        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        assert l2_norm_diff(a, b) == pytest.approx(l2_norm_diff(b, a), abs=1e-5)

    def test_non_negative(self):
        a = torch.randn(10)
        b = torch.randn(10)
        assert l2_norm_diff(a, b) >= 0


class TestCosineSimilarity:
    def test_identical_tensors(self):
        t = torch.randn(10, 10)
        assert cosine_similarity(t, t) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_tensors(self):
        t = torch.randn(10)
        assert cosine_similarity(t, -t) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_bounds(self):
        a = torch.randn(20)
        b = torch.randn(20)
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_zero_tensor(self):
        a = torch.zeros(10)
        b = torch.randn(10)
        assert cosine_similarity(a, b) == 0.0


class TestFrobeniusNormDiff:
    def test_identical_tensors(self):
        t = torch.randn(5, 5)
        assert frobenius_norm_diff(t, t) == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self):
        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        assert frobenius_norm_diff(a, b) >= 0

    def test_1d_fallback(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        expected = math.sqrt(2.0)
        assert frobenius_norm_diff(a, b) == pytest.approx(expected, abs=1e-5)


class TestRelativeChange:
    def test_identical_tensors(self):
        t = torch.randn(5, 5)
        assert relative_change(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_zero_base(self):
        base = torch.zeros(5, 5)
        trained = torch.randn(5, 5)
        assert relative_change(base, trained) == 0.0

    def test_non_negative(self):
        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        assert relative_change(a, b) >= 0

    def test_small_perturbation(self):
        base = torch.randn(10, 10)
        perturbation = torch.randn(10, 10) * 0.001
        trained = base + perturbation
        rc = relative_change(base, trained)
        assert rc < 0.1  # Small perturbation = small relative change


class TestMaxAbsDiff:
    def test_identical(self):
        t = torch.randn(5)
        assert max_abs_diff(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        assert max_abs_diff(a, b) == pytest.approx(3.0, abs=1e-5)


class TestMeanAbsDiff:
    def test_identical(self):
        t = torch.randn(5)
        assert mean_abs_diff(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        a = torch.zeros(3)
        b = torch.tensor([1.0, 2.0, 3.0])
        assert mean_abs_diff(a, b) == pytest.approx(2.0, abs=1e-5)
