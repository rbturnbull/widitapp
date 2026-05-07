from unittest.mock import patch

import numpy as np

from widitapp.diffusion import create_diffusion
from widitapp.diffusion import gaussian_diffusion as gd


def test_create_diffusion_uses_defaults_and_expands_empty_respacing():
    betas = np.array([0.1, 0.2])
    diffusion = object()

    with patch("widitapp.diffusion.gd.get_named_beta_schedule", return_value=betas) as beta_schedule, patch(
        "widitapp.diffusion.space_timesteps", return_value={0, 1}
    ) as space_timesteps, patch("widitapp.diffusion.SpacedDiffusion", return_value=diffusion) as spaced_diffusion:
        result = create_diffusion(timestep_respacing="", diffusion_steps=2)

    assert result is diffusion
    beta_schedule.assert_called_once_with("linear", 2)
    space_timesteps.assert_called_once_with(2, [2])
    spaced_diffusion.assert_called_once_with(
        use_timesteps={0, 1},
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=gd.LossType.MSE,
    )


def test_create_diffusion_expands_none_respacing():
    with patch("widitapp.diffusion.gd.get_named_beta_schedule", return_value=np.array([0.1])), patch(
        "widitapp.diffusion.space_timesteps", return_value={0}
    ) as space_timesteps, patch("widitapp.diffusion.SpacedDiffusion", return_value=object()):
        create_diffusion(timestep_respacing=None, diffusion_steps=1)

    space_timesteps.assert_called_once_with(1, [1])


def test_create_diffusion_passes_explicit_respacing_and_noise_schedule():
    with patch("widitapp.diffusion.gd.get_named_beta_schedule", return_value=np.array([0.1] * 10)) as beta_schedule, patch(
        "widitapp.diffusion.space_timesteps", return_value={0, 2, 4, 6, 8}
    ) as space_timesteps, patch("widitapp.diffusion.SpacedDiffusion", return_value=object()):
        create_diffusion(
            timestep_respacing="ddim5",
            noise_schedule="squaredcos_cap_v2",
            diffusion_steps=10,
        )

    beta_schedule.assert_called_once_with("squaredcos_cap_v2", 10)
    space_timesteps.assert_called_once_with(10, "ddim5")


def test_create_diffusion_uses_kl_loss_when_requested():
    with patch("widitapp.diffusion.gd.get_named_beta_schedule", return_value=np.array([0.1])), patch(
        "widitapp.diffusion.space_timesteps", return_value={0}
    ), patch("widitapp.diffusion.SpacedDiffusion", return_value=object()) as spaced_diffusion:
        create_diffusion(timestep_respacing="", use_kl=True, rescale_learned_sigmas=True, diffusion_steps=1)

    assert spaced_diffusion.call_args.kwargs["loss_type"] == gd.LossType.RESCALED_KL


def test_create_diffusion_uses_rescaled_mse_loss_when_requested_without_kl():
    with patch("widitapp.diffusion.gd.get_named_beta_schedule", return_value=np.array([0.1])), patch(
        "widitapp.diffusion.space_timesteps", return_value={0}
    ), patch("widitapp.diffusion.SpacedDiffusion", return_value=object()) as spaced_diffusion:
        create_diffusion(timestep_respacing="", rescale_learned_sigmas=True, diffusion_steps=1)

    assert spaced_diffusion.call_args.kwargs["loss_type"] == gd.LossType.RESCALED_MSE


def test_create_diffusion_predict_xstart_changes_model_mean_type():
    with patch("widitapp.diffusion.gd.get_named_beta_schedule", return_value=np.array([0.1])), patch(
        "widitapp.diffusion.space_timesteps", return_value={0}
    ), patch("widitapp.diffusion.SpacedDiffusion", return_value=object()) as spaced_diffusion:
        create_diffusion(timestep_respacing="", predict_xstart=True, diffusion_steps=1)

    assert spaced_diffusion.call_args.kwargs["model_mean_type"] == gd.ModelMeanType.START_X


def test_create_diffusion_fixed_large_variance_when_not_learning_sigma():
    with patch("widitapp.diffusion.gd.get_named_beta_schedule", return_value=np.array([0.1])), patch(
        "widitapp.diffusion.space_timesteps", return_value={0}
    ), patch("widitapp.diffusion.SpacedDiffusion", return_value=object()) as spaced_diffusion:
        create_diffusion(timestep_respacing="", learn_sigma=False, sigma_small=False, diffusion_steps=1)

    assert spaced_diffusion.call_args.kwargs["model_var_type"] == gd.ModelVarType.FIXED_LARGE


def test_create_diffusion_fixed_small_variance_when_sigma_small():
    with patch("widitapp.diffusion.gd.get_named_beta_schedule", return_value=np.array([0.1])), patch(
        "widitapp.diffusion.space_timesteps", return_value={0}
    ), patch("widitapp.diffusion.SpacedDiffusion", return_value=object()) as spaced_diffusion:
        create_diffusion(timestep_respacing="", learn_sigma=False, sigma_small=True, diffusion_steps=1)

    assert spaced_diffusion.call_args.kwargs["model_var_type"] == gd.ModelVarType.FIXED_SMALL
