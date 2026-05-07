from unittest.mock import patch

import numpy as np
import pytest
import torch

from widitapp.diffusion.timestep_sampler import (
    LossAwareSampler,
    LossSecondMomentResampler,
    UniformSampler,
    create_named_schedule_sampler,
)


class FakeDiffusion:
    num_timesteps = 4


def test_create_named_schedule_sampler_returns_uniform_sampler():
    sampler = create_named_schedule_sampler("uniform", FakeDiffusion())

    assert isinstance(sampler, UniformSampler)


def test_create_named_schedule_sampler_returns_loss_second_moment_sampler():
    sampler = create_named_schedule_sampler("loss-second-moment", FakeDiffusion())

    assert isinstance(sampler, LossSecondMomentResampler)


def test_create_named_schedule_sampler_rejects_unknown_sampler():
    with pytest.raises(NotImplementedError, match="unknown schedule sampler"):
        create_named_schedule_sampler("missing", FakeDiffusion())


def test_uniform_sampler_weights_are_all_ones():
    sampler = UniformSampler(FakeDiffusion())

    assert np.array_equal(sampler.weights(), np.ones(FakeDiffusion.num_timesteps))


def test_schedule_sampler_sample_uses_weights_for_indices_and_importance_weights():
    sampler = UniformSampler(FakeDiffusion())
    device = torch.device("cpu")

    with patch("numpy.random.choice", return_value=np.array([0, 2])) as choice:
        timesteps, weights = sampler.sample(batch_size=2, device=device)

    choice.assert_called_once()
    args, kwargs = choice.call_args
    assert args == (FakeDiffusion.num_timesteps,)
    assert kwargs["size"] == (2,)
    assert np.array_equal(kwargs["p"], np.full(FakeDiffusion.num_timesteps, 0.25))
    assert torch.equal(timesteps, torch.tensor([0, 2], dtype=torch.long))
    assert torch.allclose(weights, torch.ones(2))
    assert timesteps.device == device
    assert weights.device == device


def test_loss_second_moment_sampler_starts_with_uniform_weights():
    sampler = LossSecondMomentResampler(FakeDiffusion(), history_per_term=2)

    assert np.array_equal(sampler.weights(), np.ones(FakeDiffusion.num_timesteps))
    assert not sampler._warmed_up()


def test_loss_second_moment_sampler_records_losses_until_history_is_full():
    sampler = LossSecondMomentResampler(FakeDiffusion(), history_per_term=2)

    sampler.update_with_all_losses([0, 0, 1], [1.0, 2.0, 3.0])

    assert np.array_equal(sampler._loss_counts, np.array([2, 1, 0, 0]))
    assert np.array_equal(sampler._loss_history[0], np.array([1.0, 2.0]))
    assert np.array_equal(sampler._loss_history[1], np.array([3.0, 0.0]))


def test_loss_second_moment_sampler_shifts_old_losses_when_history_is_full():
    sampler = LossSecondMomentResampler(FakeDiffusion(), history_per_term=2)

    sampler.update_with_all_losses([0, 0, 0], [1.0, 2.0, 4.0])

    assert sampler._loss_counts[0] == 2
    assert np.array_equal(sampler._loss_history[0], np.array([2.0, 4.0]))


def test_loss_second_moment_sampler_weights_use_second_moment_after_warmup():
    sampler = LossSecondMomentResampler(FakeDiffusion(), history_per_term=2, uniform_prob=0.1)
    sampler.update_with_all_losses(
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    )

    weights = sampler.weights()

    second_moment_weights = np.array([1.0, 2.0, 3.0, 4.0])
    second_moment_weights /= second_moment_weights.sum()
    expected = second_moment_weights * 0.9 + 0.1 / FakeDiffusion.num_timesteps
    assert sampler._warmed_up()
    assert np.allclose(weights, expected)
    assert weights.sum() == pytest.approx(1.0)


class RecordingLossAwareSampler(LossAwareSampler):
    def __init__(self):
        self.received_timesteps = None
        self.received_losses = None

    def weights(self):
        return np.ones(1)

    def update_with_all_losses(self, ts, losses):
        self.received_timesteps = ts
        self.received_losses = losses


def test_loss_aware_sampler_update_with_local_losses_gathers_distributed_batches():
    sampler = RecordingLossAwareSampler()
    local_ts = torch.tensor([1, 3], dtype=torch.long)
    local_losses = torch.tensor([0.5, 1.5])

    def fake_all_gather(outputs, value):
        if value.dtype == torch.int32:
            outputs[0].fill_(2)
            outputs[1].fill_(1)
        elif value.dtype == torch.long:
            outputs[0].copy_(torch.tensor([1, 3]))
            outputs[1][:1].copy_(torch.tensor([2]))
        else:
            outputs[0].copy_(torch.tensor([0.5, 1.5]))
            outputs[1][:1].copy_(torch.tensor([2.5]))

    with patch("widitapp.diffusion.timestep_sampler.dist.get_world_size", return_value=2), patch(
        "widitapp.diffusion.timestep_sampler.dist.all_gather", side_effect=fake_all_gather
    ):
        sampler.update_with_local_losses(local_ts, local_losses)

    assert sampler.received_timesteps == [1, 3, 2]
    assert sampler.received_losses == [0.5, 1.5, 2.5]
