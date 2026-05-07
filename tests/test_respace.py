import numpy as np
import pytest
import torch

from widitapp.diffusion.gaussian_diffusion import LossType, ModelMeanType, ModelVarType
from widitapp.diffusion.respace import SpacedDiffusion, _WrappedModel, space_timesteps


def diffusion_kwargs(betas=None):
    return {
        "betas": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64) if betas is None else betas,
        "model_mean_type": ModelMeanType.EPSILON,
        "model_var_type": ModelVarType.FIXED_SMALL,
        "loss_type": LossType.MSE,
    }


def test_space_timesteps_accepts_list_counts():
    assert space_timesteps(10, [2, 3]) == {0, 4, 5, 7, 9}


def test_space_timesteps_accepts_comma_separated_counts():
    assert space_timesteps(10, "2,3") == {0, 4, 5, 7, 9}


def test_space_timesteps_single_count_section_uses_first_step_in_section():
    assert space_timesteps(6, [1, 2]) == {0, 3, 5}


def test_space_timesteps_accepts_ddim_stride():
    assert space_timesteps(10, "ddim5") == {0, 2, 4, 6, 8}


def test_space_timesteps_rejects_impossible_ddim_count():
    with pytest.raises(ValueError, match="cannot create exactly"):
        space_timesteps(10, "ddim6")


def test_space_timesteps_rejects_section_too_small():
    with pytest.raises(ValueError, match="cannot divide section"):
        space_timesteps(4, [3, 3])


def test_spaced_diffusion_builds_new_betas_and_timestep_map():
    diffusion = SpacedDiffusion(use_timesteps={0, 2, 3}, **diffusion_kwargs())

    base_alphas_cumprod = np.cumprod(1.0 - np.array([0.1, 0.2, 0.3, 0.4]))
    expected_betas = np.array(
        [
            1 - base_alphas_cumprod[0] / 1.0,
            1 - base_alphas_cumprod[2] / base_alphas_cumprod[0],
            1 - base_alphas_cumprod[3] / base_alphas_cumprod[2],
        ]
    )

    assert diffusion.original_num_steps == 4
    assert diffusion.timestep_map == [0, 2, 3]
    assert diffusion.num_timesteps == 3
    assert np.allclose(diffusion.betas, expected_betas)


def test_spaced_diffusion_wrap_model_is_idempotent_for_wrapped_model():
    diffusion = SpacedDiffusion(use_timesteps={0, 2}, **diffusion_kwargs())
    wrapped = _WrappedModel(lambda x, t: x, diffusion.timestep_map, diffusion.original_num_steps)

    assert diffusion._wrap_model(wrapped) is wrapped


def test_spaced_diffusion_scale_timesteps_returns_input_unchanged():
    diffusion = SpacedDiffusion(use_timesteps={0, 2}, **diffusion_kwargs())
    timesteps = torch.tensor([0, 1])

    assert diffusion._scale_timesteps(timesteps) is timesteps


def test_wrapped_model_maps_spaced_timesteps_to_original_timesteps():
    seen = {}

    def model(x, ts, scale=1.0):
        seen["ts"] = ts.clone()
        seen["scale"] = scale
        return x * scale

    wrapped = _WrappedModel(model, timestep_map=[0, 3, 7], original_num_steps=10)
    x = torch.ones(2, 1)

    result = wrapped(x, torch.tensor([2, 1]), scale=2.0)

    assert torch.equal(seen["ts"], torch.tensor([7, 3]))
    assert seen["scale"] == 2.0
    assert torch.equal(result, torch.full_like(x, 2.0))


def test_spaced_diffusion_training_losses_calls_model_with_original_timesteps():
    seen = {}

    class NoiseModel(torch.nn.Module):
        def forward(self, x, ts, **kwargs):
            seen["ts"] = ts.clone()
            return torch.zeros_like(x)

    diffusion = SpacedDiffusion(use_timesteps={1, 3}, **diffusion_kwargs())
    x_start = torch.zeros(2, 1, 2, 2)
    noise = torch.zeros_like(x_start)

    terms = diffusion.training_losses(NoiseModel(), x_start, torch.tensor([0, 1]), noise=noise)

    assert torch.equal(seen["ts"], torch.tensor([1, 3]))
    assert torch.allclose(terms["loss"], torch.zeros(2))


def test_spaced_diffusion_p_mean_variance_calls_model_with_original_timesteps():
    seen = {}

    class ZeroModel(torch.nn.Module):
        def forward(self, x, ts, **kwargs):
            seen["ts"] = ts.clone()
            return torch.zeros_like(x)

    diffusion = SpacedDiffusion(use_timesteps={1, 3}, **diffusion_kwargs())
    x = torch.zeros(2, 1, 2, 2)

    out = diffusion.p_mean_variance(ZeroModel(), x, torch.tensor([0, 1]))

    assert torch.equal(seen["ts"], torch.tensor([1, 3]))
    assert out["mean"].shape == x.shape
    assert out["pred_xstart"].shape == x.shape


def test_spaced_diffusion_condition_mean_wraps_condition_function_timesteps():
    seen = {}
    diffusion = SpacedDiffusion(use_timesteps={1, 3}, **diffusion_kwargs())
    x = torch.zeros(2, 1, 2, 2)
    p_mean_var = {
        "mean": torch.zeros_like(x),
        "variance": torch.ones_like(x),
    }

    def cond_fn(x, ts, value):
        seen["ts"] = ts.clone()
        return torch.full_like(x, value)

    result = diffusion.condition_mean(cond_fn, p_mean_var, x, torch.tensor([0, 1]), {"value": 2.0})

    assert torch.equal(seen["ts"], torch.tensor([1, 3]))
    assert torch.allclose(result, torch.full_like(x, 2.0))


def test_spaced_diffusion_condition_score_wraps_condition_function_timesteps():
    seen = {}
    diffusion = SpacedDiffusion(use_timesteps={1, 3}, **diffusion_kwargs())
    x = torch.zeros(2, 1, 2, 2)
    t = torch.tensor([0, 1])
    p_mean_var = diffusion.p_mean_variance(lambda x, ts: torch.zeros_like(x), x, t)

    def cond_fn(x, ts, value):
        seen["ts"] = ts.clone()
        return torch.full_like(x, value)

    result = diffusion.condition_score(cond_fn, p_mean_var, x, t, {"value": 1.0})

    assert torch.equal(seen["ts"], torch.tensor([1, 3]))
    assert result["mean"].shape == x.shape
    assert result["pred_xstart"].shape == x.shape
    assert not torch.allclose(result["pred_xstart"], p_mean_var["pred_xstart"])
