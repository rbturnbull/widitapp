import math

import torch

from widitapp.diffusion.diffusion_utils import continuous_gaussian_log_likelihood


def test_continuous_gaussian_log_likelihood_matches_standard_normal_formula():
    x = torch.tensor([-1.0, 0.0, 1.0])
    means = torch.zeros_like(x)
    log_scales = torch.zeros_like(x)

    log_probs = continuous_gaussian_log_likelihood(
        x,
        means=means,
        log_scales=log_scales,
    )

    expected = -0.5 * x**2 - 0.5 * math.log(2.0 * math.pi)
    assert torch.allclose(log_probs, expected)


def test_continuous_gaussian_log_likelihood_uses_means_and_log_scales():
    x = torch.tensor([[1.0, 3.0]])
    means = torch.tensor([[1.0, 1.0]])
    log_scales = torch.log(torch.tensor([[1.0, 2.0]]))

    log_probs = continuous_gaussian_log_likelihood(
        x,
        means=means,
        log_scales=log_scales,
    )

    normalized_x = torch.tensor([[0.0, 1.0]])
    expected = -0.5 * normalized_x**2 - 0.5 * math.log(2.0 * math.pi)
    assert log_probs.shape == x.shape
    assert torch.allclose(log_probs, expected)
