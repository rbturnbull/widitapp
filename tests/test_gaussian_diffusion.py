import math

import numpy as np
import pytest
import torch

from widitapp.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    _extract_into_tensor,
    betas_for_alpha_bar,
    get_beta_schedule,
    get_named_beta_schedule,
    mean_flat,
)


class ZeroModel(torch.nn.Module):
    def forward(self, x, t, **kwargs):
        return torch.zeros_like(x)


def make_diffusion(
    *,
    betas=None,
    model_mean_type=ModelMeanType.EPSILON,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
):
    return GaussianDiffusion(
        betas=np.array([0.1, 0.2, 0.3], dtype=np.float64) if betas is None else betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
    )


def test_mean_flat_averages_non_batch_dimensions():
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    result = mean_flat(tensor)

    assert torch.allclose(result, torch.tensor([5.5, 17.5]))


@pytest.mark.parametrize("schedule", ["quad", "linear", "warmup10", "warmup50", "const", "jsd"])
def test_get_beta_schedule_returns_valid_schedule(schedule):
    betas = get_beta_schedule(
        schedule,
        beta_start=0.001,
        beta_end=0.02,
        num_diffusion_timesteps=10,
    )

    assert betas.shape == (10,)
    assert betas.dtype == np.float64
    assert np.all(betas > 0)
    assert np.all(betas <= 1)


def test_get_beta_schedule_rejects_unknown_schedule():
    with pytest.raises(NotImplementedError):
        get_beta_schedule("unknown", beta_start=0.001, beta_end=0.02, num_diffusion_timesteps=10)


def test_get_named_beta_schedule_linear_scales_to_number_of_timesteps():
    betas = get_named_beta_schedule("linear", 1000)

    assert betas[0] == pytest.approx(0.0001)
    assert betas[-1] == pytest.approx(0.02)


def test_get_named_beta_schedule_squaredcos_is_bounded():
    betas = get_named_beta_schedule("squaredcos_cap_v2", 10)

    assert betas.shape == (10,)
    assert np.all(betas > 0)
    assert np.all(betas <= 0.999)


def test_get_named_beta_schedule_rejects_unknown_name():
    with pytest.raises(NotImplementedError, match="unknown beta schedule"):
        get_named_beta_schedule("missing", 10)


def test_betas_for_alpha_bar_applies_max_beta_cap():
    betas = betas_for_alpha_bar(4, lambda t: 1.0 - t, max_beta=0.5)

    assert betas.shape == (4,)
    assert betas.max() == pytest.approx(0.5)


def test_gaussian_diffusion_rejects_non_1d_betas():
    with pytest.raises(AssertionError, match="betas must be 1-D"):
        make_diffusion(betas=np.ones((2, 2), dtype=np.float64) * 0.1)


def test_gaussian_diffusion_rejects_invalid_betas():
    with pytest.raises(AssertionError, match="betas must be in"):
        make_diffusion(betas=np.array([0.1, 0.0, 0.2], dtype=np.float64))


def test_extract_into_tensor_broadcasts_values_to_shape():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    timesteps = torch.tensor([2, 0])

    result = _extract_into_tensor(values, timesteps, (2, 3, 4))

    assert result.shape == (2, 3, 4)
    assert torch.allclose(result[0], torch.full((3, 4), 3.0))
    assert torch.allclose(result[1], torch.full((3, 4), 1.0))


def test_q_mean_variance_returns_expected_shapes_and_values():
    diffusion = make_diffusion()
    x_start = torch.ones(2, 1, 2, 2)
    t = torch.tensor([0, 2])

    mean, variance, log_variance = diffusion.q_mean_variance(x_start, t)

    assert mean.shape == x_start.shape
    assert variance.shape == x_start.shape
    assert log_variance.shape == x_start.shape
    assert torch.allclose(mean[0], torch.full((1, 2, 2), math.sqrt(0.9)))
    assert torch.allclose(variance[0], torch.full((1, 2, 2), 0.1))


def test_q_sample_uses_provided_noise_deterministically():
    diffusion = make_diffusion()
    x_start = torch.ones(1, 1, 2, 2)
    noise = torch.full_like(x_start, 2.0)
    t = torch.tensor([0])

    sample = diffusion.q_sample(x_start, t, noise=noise)

    expected = math.sqrt(0.9) * x_start + math.sqrt(0.1) * noise
    assert torch.allclose(sample, expected)


def test_q_sample_rejects_noise_shape_mismatch():
    diffusion = make_diffusion()

    with pytest.raises(AssertionError, match="noise shape"):
        diffusion.q_sample(torch.ones(1, 1, 2, 2), torch.tensor([0]), noise=torch.ones(1, 1, 2, 1))


def test_q_posterior_mean_variance_rejects_shape_mismatch():
    diffusion = make_diffusion()

    with pytest.raises(AssertionError, match="x_start shape"):
        diffusion.q_posterior_mean_variance(
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 1),
            torch.tensor([0]),
        )


def test_predict_eps_and_xstart_are_inverse_operations():
    diffusion = make_diffusion()
    x_t = torch.randn(2, 1, 2, 2)
    eps = torch.randn_like(x_t)
    t = torch.tensor([0, 2])

    pred_xstart = diffusion._predict_xstart_from_eps(x_t, t, eps)
    recovered_eps = diffusion._predict_eps_from_xstart(x_t, t, pred_xstart)

    assert torch.allclose(recovered_eps, eps, atol=1e-5)


def test_p_mean_variance_passes_kwargs_and_returns_extra_tuple_value():
    diffusion = make_diffusion(model_mean_type=ModelMeanType.START_X)

    class TupleModel(torch.nn.Module):
        def forward(self, x, t, scale=1.0):
            return torch.full_like(x, scale), {"seen_t": t.clone()}

    x = torch.zeros(2, 1, 2, 2)
    t = torch.tensor([0, 1])

    out = diffusion.p_mean_variance(
        TupleModel(),
        x,
        t,
        clip_denoised=False,
        model_kwargs={"scale": 0.25},
    )

    assert out["mean"].shape == x.shape
    assert out["variance"].shape == x.shape
    assert out["log_variance"].shape == x.shape
    assert torch.allclose(out["pred_xstart"], torch.full_like(x, 0.25))
    assert torch.equal(out["extra"]["seen_t"], t)


def test_p_mean_variance_clips_denoised_start_prediction():
    diffusion = make_diffusion(model_mean_type=ModelMeanType.START_X)

    class LargeModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, 5.0)

    out = diffusion.p_mean_variance(LargeModel(), torch.zeros(1, 1, 2, 2), torch.tensor([0]))

    assert torch.all(out["pred_xstart"] == 1.0)


def test_p_mean_variance_rejects_bad_t_shape():
    diffusion = make_diffusion()

    with pytest.raises(AssertionError, match="t shape"):
        diffusion.p_mean_variance(ZeroModel(), torch.zeros(2, 1, 2, 2), torch.tensor([[0], [1]]))


def test_p_mean_variance_rejects_learned_variance_shape_mismatch():
    diffusion = make_diffusion(model_var_type=ModelVarType.LEARNED)

    with pytest.raises(AssertionError, match="model_output shape"):
        diffusion.p_mean_variance(ZeroModel(), torch.zeros(1, 1, 2, 2), torch.tensor([0]))


def test_condition_mean_adds_variance_weighted_gradient():
    diffusion = make_diffusion()
    x = torch.zeros(1, 1, 2, 2)
    t = torch.tensor([0])
    p_mean_var = {
        "mean": torch.ones_like(x),
        "variance": torch.full_like(x, 0.5),
    }

    result = diffusion.condition_mean(lambda x, t, scale: torch.full_like(x, scale), p_mean_var, x, t, {"scale": 2.0})

    assert torch.allclose(result, torch.full_like(x, 2.0))


def test_condition_score_updates_pred_xstart_and_mean():
    diffusion = make_diffusion()
    x = torch.zeros(1, 1, 2, 2)
    t = torch.tensor([1])
    p_mean_var = diffusion.p_mean_variance(ZeroModel(), x, t)

    out = diffusion.condition_score(
        lambda x, t, scale: torch.full_like(x, scale),
        p_mean_var,
        x,
        t,
        {"scale": 1.0},
    )

    assert out is not p_mean_var
    assert not torch.allclose(out["pred_xstart"], p_mean_var["pred_xstart"])
    assert not torch.allclose(out["mean"], p_mean_var["mean"])
    assert torch.equal(out["variance"], p_mean_var["variance"])


def test_vb_terms_bpd_returns_output_and_pred_xstart():
    diffusion = make_diffusion(model_mean_type=ModelMeanType.START_X)

    class StartModel(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.zeros_like(x)

    x_start = torch.zeros(2, 1, 2, 2)
    x_t = torch.zeros_like(x_start)

    out = diffusion._vb_terms_bpd(
        StartModel(),
        x_start=x_start,
        x_t=x_t,
        t=torch.tensor([0, 1]),
        clip_denoised=False,
    )

    assert out["output"].shape == (2,)
    assert out["pred_xstart"].shape == x_start.shape
    assert torch.isfinite(out["output"]).all()


def test_vb_terms_bpd_uses_decoder_nll_at_timestep_zero(monkeypatch):
    diffusion = make_diffusion()
    x_start = torch.zeros(2, 1, 2, 2)
    x_t = torch.zeros_like(x_start)

    def fake_p_mean_variance(*args, **kwargs):
        return {
            "mean": torch.zeros_like(x_start),
            "log_variance": torch.zeros_like(x_start),
            "pred_xstart": torch.full_like(x_start, 0.25),
        }

    monkeypatch.setattr(diffusion, "p_mean_variance", fake_p_mean_variance)
    monkeypatch.setattr(
        "widitapp.diffusion.gaussian_diffusion.normal_kl",
        lambda *args, **kwargs: torch.full_like(x_start, 8.0),
    )
    monkeypatch.setattr(
        "widitapp.diffusion.gaussian_diffusion.discretized_gaussian_log_likelihood",
        lambda *args, **kwargs: torch.full_like(x_start, -2.0),
    )

    out = diffusion._vb_terms_bpd(
        ZeroModel(),
        x_start=x_start,
        x_t=x_t,
        t=torch.tensor([0, 1]),
        clip_denoised=False,
    )

    assert out["output"][0] == pytest.approx(2.0 / math.log(2.0))
    assert out["output"][1] == pytest.approx(8.0 / math.log(2.0))
    assert torch.allclose(out["pred_xstart"], torch.full_like(x_start, 0.25))


def test_calc_bpd_loop_aggregates_terms_and_prior(monkeypatch):
    diffusion = make_diffusion()
    seen_timesteps = []
    x_start = torch.zeros(2, 1, 2, 2)

    def fake_vb_terms(model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        seen_timesteps.append(t.clone())
        value = t.float() + 1.0
        return {
            "output": value,
            "pred_xstart": torch.ones_like(x_start) * value.view(-1, 1, 1, 1),
        }

    monkeypatch.setattr(diffusion, "_vb_terms_bpd", fake_vb_terms)
    monkeypatch.setattr(diffusion, "_prior_bpd", lambda x_start: torch.tensor([0.5, 1.5]))
    monkeypatch.setattr(
        "widitapp.diffusion.gaussian_diffusion.th.randn_like",
        lambda tensor: torch.ones_like(tensor),
    )

    out = diffusion.calc_bpd_loop(ZeroModel(), x_start)

    assert [t[0].item() for t in seen_timesteps] == [2, 1, 0]
    assert out["vb"].shape == (2, 3)
    assert out["xstart_mse"].shape == (2, 3)
    assert out["mse"].shape == (2, 3)
    assert torch.allclose(out["vb"], torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]))
    assert torch.allclose(out["prior_bpd"], torch.tensor([0.5, 1.5]))
    assert torch.allclose(out["total_bpd"], torch.tensor([6.5, 7.5]))


def test_training_losses_mse_zero_when_model_predicts_noise():
    diffusion = make_diffusion(model_mean_type=ModelMeanType.EPSILON)
    x_start = torch.zeros(2, 1, 2, 2)
    noise = torch.full_like(x_start, 0.5)

    class NoiseModel(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return noise

    terms = diffusion.training_losses(NoiseModel(), x_start, torch.tensor([0, 1]), noise=noise)

    assert torch.allclose(terms["mse"], torch.zeros(2))
    assert torch.allclose(terms["loss"], torch.zeros(2))


def test_training_losses_start_x_uses_x_start_as_target():
    diffusion = make_diffusion(model_mean_type=ModelMeanType.START_X)
    x_start = torch.ones(2, 1, 2, 2)

    terms = diffusion.training_losses(ZeroModel(), x_start, torch.tensor([0, 1]), noise=torch.zeros_like(x_start))

    assert torch.allclose(terms["mse"], torch.ones(2))
    assert torch.allclose(terms["loss"], torch.ones(2))


def test_training_losses_rescaled_kl_scales_vb_output(monkeypatch):
    diffusion = make_diffusion(loss_type=LossType.RESCALED_KL)
    monkeypatch.setattr(
        diffusion,
        "_vb_terms_bpd",
        lambda **kwargs: {"output": torch.tensor([1.0, 2.0])},
    )

    terms = diffusion.training_losses(ZeroModel(), torch.zeros(2, 1, 2, 2), torch.tensor([0, 1]))

    assert torch.allclose(terms["loss"], torch.tensor([3.0, 6.0]))


def test_prior_bpd_returns_one_value_per_batch_item():
    diffusion = make_diffusion()

    prior = diffusion._prior_bpd(torch.zeros(2, 1, 2, 2))

    assert prior.shape == (2,)
    assert torch.isfinite(prior).all()


def test_p_sample_uses_mean_without_noise_at_timestep_zero(monkeypatch):
    diffusion = make_diffusion(model_mean_type=ModelMeanType.START_X)
    x = torch.zeros(1, 1, 2, 2)
    t = torch.tensor([0])

    class StartModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, 0.5)

    monkeypatch.setattr(
        "widitapp.diffusion.gaussian_diffusion.th.randn_like",
        lambda tensor: torch.full_like(tensor, 100.0),
    )

    out = diffusion.p_sample(StartModel(), x, t, clip_denoised=False)
    mean_out = diffusion.p_mean_variance(StartModel(), x, t, clip_denoised=False)

    assert torch.allclose(out["sample"], mean_out["mean"])
    assert torch.allclose(out["pred_xstart"], torch.full_like(x, 0.5))


def test_p_sample_applies_condition_mean(monkeypatch):
    diffusion = make_diffusion()
    x = torch.zeros(1, 1, 2, 2)
    t = torch.tensor([1])

    monkeypatch.setattr(
        "widitapp.diffusion.gaussian_diffusion.th.randn_like",
        lambda tensor: torch.zeros_like(tensor),
    )

    unconditioned = diffusion.p_sample(ZeroModel(), x, t)
    conditioned = diffusion.p_sample(
        ZeroModel(),
        x,
        t,
        cond_fn=lambda x, t, **kwargs: torch.ones_like(x),
        model_kwargs={},
    )

    assert torch.all(conditioned["sample"] > unconditioned["sample"])


def test_p_sample_loop_progressive_yields_reverse_timestep_order(monkeypatch):
    diffusion = make_diffusion()
    seen_timesteps = []

    def fake_p_sample(model, img, t, **kwargs):
        seen_timesteps.append(t.clone())
        return {"sample": img + t.float().view(-1, 1, 1, 1), "pred_xstart": img}

    monkeypatch.setattr(diffusion, "p_sample", fake_p_sample)

    outputs = list(
        diffusion.p_sample_loop_progressive(
            ZeroModel(),
            shape=(1, 1, 1, 1),
            noise=torch.zeros(1, 1, 1, 1),
            device=torch.device("cpu"),
        )
    )

    assert [t.item() for t in seen_timesteps] == [2, 1, 0]
    assert len(outputs) == diffusion.num_timesteps
    assert torch.allclose(outputs[-1]["sample"], torch.tensor([[[[3.0]]]]))


def test_p_sample_loop_returns_final_progressive_sample(monkeypatch):
    diffusion = make_diffusion()

    def fake_progressive(*args, **kwargs):
        yield {"sample": torch.tensor([1.0])}
        yield {"sample": torch.tensor([2.0])}

    monkeypatch.setattr(diffusion, "p_sample_loop_progressive", fake_progressive)

    sample = diffusion.p_sample_loop(ZeroModel(), shape=(1,))

    assert torch.equal(sample, torch.tensor([2.0]))


def test_ddim_sample_returns_pred_xstart_at_timestep_zero_with_eta_zero():
    diffusion = make_diffusion(model_mean_type=ModelMeanType.START_X)
    x = torch.zeros(1, 1, 2, 2)
    t = torch.tensor([0])

    class StartModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, 0.25)

    out = diffusion.ddim_sample(StartModel(), x, t, clip_denoised=False, eta=0.0)

    assert torch.allclose(out["sample"], torch.full_like(x, 0.25))
    assert torch.allclose(out["pred_xstart"], torch.full_like(x, 0.25))


def test_ddim_sample_applies_condition_score(monkeypatch):
    diffusion = make_diffusion()
    x = torch.zeros(1, 1, 2, 2)
    t = torch.tensor([0])

    def fake_condition_score(cond_fn, p_mean_var, x, t, model_kwargs=None):
        out = p_mean_var.copy()
        out["pred_xstart"] = torch.full_like(x, 0.75)
        return out

    monkeypatch.setattr(diffusion, "condition_score", fake_condition_score)

    out = diffusion.ddim_sample(
        ZeroModel(),
        x,
        t,
        cond_fn=lambda x, t, **kwargs: torch.ones_like(x),
    )

    assert torch.allclose(out["sample"], torch.full_like(x, 0.75))
    assert torch.allclose(out["pred_xstart"], torch.full_like(x, 0.75))


def test_ddim_sample_loop_progressive_yields_reverse_timestep_order(monkeypatch):
    diffusion = make_diffusion()
    seen_timesteps = []

    def fake_ddim_sample(model, img, t, **kwargs):
        seen_timesteps.append(t.clone())
        return {"sample": img + t.float().view(-1, 1, 1, 1), "pred_xstart": img}

    monkeypatch.setattr(diffusion, "ddim_sample", fake_ddim_sample)

    outputs = list(
        diffusion.ddim_sample_loop_progressive(
            ZeroModel(),
            shape=(1, 1, 1, 1),
            noise=torch.zeros(1, 1, 1, 1),
            device=torch.device("cpu"),
            eta=0.5,
        )
    )

    assert [t.item() for t in seen_timesteps] == [2, 1, 0]
    assert len(outputs) == diffusion.num_timesteps
    assert torch.allclose(outputs[-1]["sample"], torch.tensor([[[[3.0]]]]))


def test_ddim_sample_loop_returns_final_progressive_sample(monkeypatch):
    diffusion = make_diffusion()

    def fake_progressive(*args, **kwargs):
        yield {"sample": torch.tensor([3.0])}
        yield {"sample": torch.tensor([4.0])}

    monkeypatch.setattr(diffusion, "ddim_sample_loop_progressive", fake_progressive)

    sample = diffusion.ddim_sample_loop(ZeroModel(), shape=(1,))

    assert torch.equal(sample, torch.tensor([4.0]))


def test_ddim_reverse_sample_rejects_nonzero_eta():
    diffusion = make_diffusion()

    with pytest.raises(AssertionError, match="Reverse ODE only"):
        diffusion.ddim_reverse_sample(
            ZeroModel(),
            torch.zeros(1, 1, 2, 2),
            torch.tensor([0]),
            eta=0.1,
        )


def test_ddim_reverse_sample_returns_expected_shapes():
    diffusion = make_diffusion(model_mean_type=ModelMeanType.START_X)
    x = torch.zeros(1, 1, 2, 2)
    t = torch.tensor([0])

    class StartModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, 0.5)

    out = diffusion.ddim_reverse_sample(StartModel(), x, t, clip_denoised=False)

    assert out["sample"].shape == x.shape
    assert out["pred_xstart"].shape == x.shape
    assert torch.allclose(out["pred_xstart"], torch.full_like(x, 0.5))
