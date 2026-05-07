import logging
import math
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import widitapp.training as training_module
from widitapp.training import (
    _run_validation_loop,
    build_loss_fn,
    build_val_log_payload,
    clear_after_cuda_oom,
    create_logger,
    get_state_dict_for_saving,
    is_cuda_out_of_memory,
    is_loss_finite,
    is_finite_tensor,
    loss_value_for_logging,
    requires_grad,
    tensor_info_for_logging,
    update_ema,
)
from accelerate import Accelerator


class FakeAccelerator:
    def __init__(self, mixed_precision="no"):
        self.mixed_precision = mixed_precision
        self.device = torch.device("cpu")
        self.is_main_process = True

    def prepare(self, *args):
        return args if len(args) > 1 else args[0]

    def reduce(self, tensor, reduction="mean"):
        return tensor

    def backward(self, loss):
        loss.backward()

    def autocast(self):
        return torch.autocast(device_type="cpu", enabled=False)


class IdentityModel(torch.nn.Module):
    def forward(self, x, timestep=None, **kwargs):
        return x


def run_train_on_cpu(**kwargs):
    logger = logging.getLogger("train")
    old_handlers = list(logger.handlers)
    logger.handlers.clear()
    try:
        with patch("widitapp.training.Accelerator", FakeAccelerator), patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.synchronize"):
            return training_module.train(**kwargs)
    finally:
        logger.handlers.clear()
        logger.handlers.extend(old_handlers)


def test_create_logger_writes_file(tmp_path):
    logger = logging.getLogger("train")
    old_handlers = list(logger.handlers)
    logger.handlers.clear()
    try:
        log_dir = tmp_path
        train_logger = create_logger(str(log_dir))
        train_logger.info("hello")
        log_path = log_dir / "log.txt"
        assert log_path.exists()
        assert "hello" in log_path.read_text()
    finally:
        logger.handlers.clear()
        logger.handlers.extend(old_handlers)


def test_get_state_dict_for_saving_prefers_module():
    model = torch.nn.Linear(2, 2)

    class Wrapper:
        def __init__(self, module):
            self.module = module

    wrapped = Wrapper(model)
    state = get_state_dict_for_saving(wrapped)
    assert state.keys() == model.state_dict().keys()


def test_update_ema_applies_decay():
    model = torch.nn.Linear(2, 2, bias=False)
    ema = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.constant_(model.weight, 1.0)
    torch.nn.init.constant_(ema.weight, 0.0)

    update_ema(ema, model, decay=0.5)
    assert torch.allclose(ema.weight, torch.full_like(ema.weight, 0.5))


def test_update_ema_handles_module_prefix():
    model = torch.nn.Linear(2, 2, bias=False)
    ema = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.constant_(model.weight, 2.0)
    torch.nn.init.constant_(ema.weight, 0.0)

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    update_ema(ema, Wrapper(model), decay=0.25)

    assert torch.allclose(ema.weight, torch.full_like(ema.weight, 1.5))


def test_requires_grad_toggles():
    model = torch.nn.Linear(2, 2)
    requires_grad(model, False)
    assert all(not p.requires_grad for p in model.parameters())
    requires_grad(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_run_validation_loop_supervised_zero_loss():
    x = torch.randn(4, 1, 8, 8)
    target = x.clone()
    ds = TensorDataset(x, target)
    dl = DataLoader(ds, batch_size=2)

    accelerator = Accelerator(mixed_precision="no")
    model = IdentityModel()

    loss = _run_validation_loop(
        accelerator=accelerator,
        model_for_eval=model,
        diffusion=None,
        dataloader=dl,
        device=accelerator.device,
        dtype=torch.float32,
        use_diffusion=False,
        criterion=build_loss_fn("mse"),
    )

    assert loss["loss"] == 0.0


def test_run_validation_loop_supervised_tracks_extra_criteria():
    x = torch.ones(4, 1)
    target = torch.zeros(4, 1)
    dl = DataLoader(TensorDataset(x, target), batch_size=2)
    accelerator = Accelerator(mixed_precision="no")
    model = IdentityModel()

    loss = _run_validation_loop(
        accelerator=accelerator,
        model_for_eval=model,
        diffusion=None,
        dataloader=dl,
        device=accelerator.device,
        dtype=torch.float32,
        use_diffusion=False,
        criterion=build_loss_fn("mse"),
        extra_criteria={
            "mse": torch.nn.MSELoss(reduction="mean"),
            "smoothl1": torch.nn.SmoothL1Loss(reduction="mean"),
        },
    )

    assert loss["loss"] == 1.0
    assert loss["mse"] == 1.0
    assert loss["smoothl1"] == 0.5


def test_run_validation_loop_empty_dataloader_returns_nan_loss():
    dl = DataLoader(TensorDataset(torch.empty(0, 1), torch.empty(0, 1)), batch_size=2)
    accelerator = Accelerator(mixed_precision="no")

    loss = _run_validation_loop(
        accelerator=accelerator,
        model_for_eval=IdentityModel(),
        diffusion=None,
        dataloader=dl,
        device=accelerator.device,
        dtype=torch.float32,
        use_diffusion=False,
        criterion=build_loss_fn("mse"),
    )

    assert math.isnan(loss["loss"])


def test_run_validation_loop_diffusion_uses_mocked_diffusion():
    x = torch.randn(4, 1, 2, 2)
    target = torch.randn(4, 1, 2, 2)
    dl = DataLoader(TensorDataset(x, target), batch_size=2)
    accelerator = Accelerator(mixed_precision="no")

    class FakeDiffusion:
        num_timesteps = 10

        def __init__(self):
            self.training_losses = Mock(return_value={"mse": torch.tensor([0.25, 0.75])})

    diffusion = FakeDiffusion()

    loss = _run_validation_loop(
        accelerator=accelerator,
        model_for_eval=IdentityModel(),
        diffusion=diffusion,
        dataloader=dl,
        device=accelerator.device,
        dtype=torch.float32,
        use_diffusion=True,
        criterion=build_loss_fn("mse"),
    )

    assert loss["loss"] == 0.5
    assert diffusion.training_losses.call_count == 2


def test_run_validation_loop_rejects_bad_batch():
    class BadDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.randn(1, 8, 8)

    dl = DataLoader(BadDataset(), batch_size=1)
    accelerator = Accelerator(mixed_precision="no")
    model = IdentityModel()

    try:
        _run_validation_loop(
            accelerator=accelerator,
            model_for_eval=model,
            diffusion=None,
            dataloader=dl,
            device=accelerator.device,
            dtype=torch.float32,
            use_diffusion=False,
            criterion=build_loss_fn("mse"),
        )
        assert False, "Expected ValueError for invalid batch format"
    except ValueError:
        pass


def test_train_requires_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(AssertionError, match="requires at least one GPU"):
            training_module.train(
                model=torch.nn.Linear(1, 1),
                training_dataloader=DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1))),
                use_diffusion=False,
            )


def test_train_supervised_updates_model_and_writes_log(tmp_path):
    class LinearWithTimestep(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False)

        def forward(self, x, timestep=None):
            return self.linear(x)

    model = LinearWithTimestep()
    torch.nn.init.constant_(model.linear.weight, 1.0)
    initial_weight = model.linear.weight.detach().clone()
    dataloader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=1)

    run_train_on_cpu(
        model=model,
        training_dataloader=dataloader,
        results_dir=str(tmp_path),
        epochs=1,
        log_every=1,
        use_diffusion=False,
        precision="fp32",
        learning_rate=0.1,
        run_name="unit-train",
    )

    assert not torch.allclose(model.linear.weight, initial_weight)
    log_text = (tmp_path / "unit-train" / "log.txt").read_text()
    assert "Training mode: supervised (no diffusion)" in log_text
    assert "train/loss=" in log_text


def test_train_logs_training_metrics_to_wandb(tmp_path):
    class LinearWithTimestep(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False)

        def forward(self, x, timestep=None):
            return self.linear(x)

    model = LinearWithTimestep()
    dataloader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1)
    fake_wandb = Mock()
    fake_wandb.summary = {}

    with patch("widitapp.training.wandb", fake_wandb):
        run_train_on_cpu(
            model=model,
            training_dataloader=dataloader,
            results_dir=str(tmp_path),
            epochs=1,
            log_every=1,
            use_diffusion=False,
            precision="fp32",
            learning_rate=0.01,
            run_name="wandb-train",
            wandb_logging=True,
            wandb_project="UnitProject",
            wandb_config={"custom": "value"},
        )

    fake_wandb.init.assert_called_once()
    _, init_kwargs = fake_wandb.init.call_args
    assert init_kwargs["project"] == "UnitProject"
    assert init_kwargs["name"] == "wandb-train"
    assert init_kwargs["dir"] == str(tmp_path / "wandb-train")
    assert init_kwargs["config"]["custom"] == "value"
    assert init_kwargs["config"]["use_diffusion"] is False
    assert init_kwargs["config"]["lr"] == 0.01
    fake_wandb.log.assert_called_once()
    log_payload, = fake_wandb.log.call_args.args
    assert "train/loss" in log_payload
    assert "train/steps_per_sec" in log_payload
    assert log_payload["epoch"] == 0
    assert log_payload["global_step"] == 1
    assert fake_wandb.log.call_args.kwargs["step"] == 1
    assert fake_wandb.summary["best/val_loss"] == float("inf")
    fake_wandb.finish.assert_called_once_with()


def test_train_skips_non_finite_input_batch_before_forward(tmp_path):
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.forward_calls = 0

        def forward(self, x, timestep=None):
            self.forward_calls += 1
            return self.linear(x)

    model = CountingModel()
    x = torch.tensor([[float("nan")], [1.0]])
    target = torch.zeros(2, 1)
    dataloader = DataLoader(TensorDataset(x, target), batch_size=1, shuffle=False)

    run_train_on_cpu(
        model=model,
        training_dataloader=dataloader,
        results_dir=str(tmp_path),
        epochs=1,
        log_every=100,
        use_diffusion=False,
        precision="fp32",
        run_name="skip-nan",
    )

    assert model.forward_calls == 1
    log_text = (tmp_path / "skip-nan" / "log.txt").read_text()
    assert "Skipping non-finite training batch" in log_text
    assert "x_is_finite=False" in log_text


def test_train_skips_cuda_oom_forward_batch_and_continues(tmp_path):
    class OOMOnceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.forward_calls = 0

        def forward(self, x, timestep=None):
            self.forward_calls += 1
            if self.forward_calls == 1:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return self.linear(x)

    model = OOMOnceModel()
    dataloader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=1)

    with patch("torch.cuda.empty_cache") as empty_cache:
        run_train_on_cpu(
            model=model,
            training_dataloader=dataloader,
            results_dir=str(tmp_path),
            epochs=1,
            log_every=100,
            use_diffusion=False,
            precision="fp32",
            run_name="skip-oom",
        )

    assert model.forward_calls == 2
    assert empty_cache.call_count == 2
    log_text = (tmp_path / "skip-oom" / "log.txt").read_text()
    assert "Skipping CUDA out-of-memory training batch" in log_text
    assert "shape=(1, 1)" in log_text


def test_train_validation_saves_best_checkpoint(tmp_path):
    class SaveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x, timestep=None):
            return self.linear(x)

        def save(self, path):
            torch.save(self.state_dict(), path)

    model = SaveModel()
    training_dataloader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1)
    validation_dataloader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1)

    with patch("widitapp.training._run_validation_loop", return_value={"loss": 0.25}) as validation_loop:
        run_train_on_cpu(
            model=model,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            results_dir=str(tmp_path),
            epochs=1,
            log_every=100,
            use_diffusion=False,
            precision="fp32",
            run_name="validation",
        )

    validation_loop.assert_called_once()
    assert (tmp_path / "validation" / "checkpoints" / "best.pt").exists()
    log_text = (tmp_path / "validation" / "log.txt").read_text()
    assert "val/loss=0.2500" in log_text
    assert "New best checkpoint" in log_text


def test_train_logs_validation_and_best_artifact_to_wandb(tmp_path):
    class SaveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x, timestep=None):
            return self.linear(x)

        def save(self, path):
            torch.save(self.state_dict(), path)

    model = SaveModel()
    training_dataloader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1)
    validation_dataloader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1)
    fake_artifact = Mock()
    fake_wandb = Mock()
    fake_wandb.summary = {}
    fake_wandb.Artifact.return_value = fake_artifact

    with patch("widitapp.training.wandb", fake_wandb), patch(
        "widitapp.training._run_validation_loop",
        return_value={"loss": 0.25, "mse": 0.5, "smoothl1": 0.125},
    ):
        run_train_on_cpu(
            model=model,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            results_dir=str(tmp_path),
            epochs=1,
            log_every=1,
            use_diffusion=False,
            precision="fp32",
            run_name="wandb-validation",
            wandb_logging=True,
            wandb_project="UnitProject",
            wandb_log_artifacts=True,
        )

    assert fake_wandb.log.call_count == 2
    train_payload = fake_wandb.log.call_args_list[0].args[0]
    val_payload = fake_wandb.log.call_args_list[1].args[0]
    assert "train/loss" in train_payload
    assert val_payload["val/loss"] == 0.25
    assert val_payload["val/mse"] == 0.5
    assert val_payload["val/smoothl1"] == 0.125
    fake_wandb.Artifact.assert_called_once_with("SaveModel-best", type="model")
    best_path = tmp_path / "wandb-validation" / "checkpoints" / "best.pt"
    fake_artifact.add_file.assert_called_once_with(str(best_path))
    fake_wandb.log_artifact.assert_called_once_with(fake_artifact)
    assert fake_wandb.summary["best/val_loss"] == 0.25
    fake_wandb.finish.assert_called_once_with()


def test_train_diffusion_uses_created_diffusion_and_updates_model(tmp_path):
    class DiffusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, t, conditioned=None):
            return x * self.weight

    class FakeDiffusion:
        num_timesteps = 5

        def __init__(self):
            self.calls = []

        def training_losses(self, model, target, timestep, model_kwargs):
            self.calls.append((target.detach().clone(), timestep.detach().clone(), model_kwargs))
            prediction = model(target, timestep, **model_kwargs)
            return {"loss": (prediction - 0.0).flatten(start_dim=1).mean(dim=1)}

    model = DiffusionModel()
    diffusion = FakeDiffusion()
    initial_weight = model.weight.detach().clone()
    dataloader = DataLoader(
        TensorDataset(torch.ones(2, 1, 1, 1), torch.ones(2, 1, 1, 1)),
        batch_size=1,
    )

    with patch("widitapp.training.create_diffusion", return_value=diffusion) as create_diffusion:
        run_train_on_cpu(
            model=model,
            training_dataloader=dataloader,
            results_dir=str(tmp_path),
            epochs=1,
            log_every=1,
            use_diffusion=True,
            precision="fp32",
            learning_rate=0.1,
            run_name="diffusion",
        )

    create_diffusion.assert_called_once_with(timestep_respacing="")
    assert len(diffusion.calls) == 2
    assert all(call[1].shape == (1,) for call in diffusion.calls)
    assert all("conditioned" in call[2] for call in diffusion.calls)
    assert not torch.allclose(model.weight, initial_weight)
    log_text = (tmp_path / "diffusion" / "log.txt").read_text()
    assert "Training mode: diffusion" in log_text
    assert "train/loss=" in log_text


def test_train_diffusion_skips_non_finite_loss_and_continues(tmp_path):
    class DiffusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, t, conditioned=None):
            return x * self.weight

    class FakeDiffusion:
        num_timesteps = 5

        def __init__(self):
            self.calls = 0

        def training_losses(self, model, target, timestep, model_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"loss": target.flatten(start_dim=1).mean(dim=1) * float("nan")}
            prediction = model(target, timestep, **model_kwargs)
            return {"loss": prediction.flatten(start_dim=1).mean(dim=1)}

    model = DiffusionModel()
    diffusion = FakeDiffusion()
    dataloader = DataLoader(
        TensorDataset(torch.ones(2, 1, 1, 1), torch.ones(2, 1, 1, 1)),
        batch_size=1,
    )

    with patch("widitapp.training.create_diffusion", return_value=diffusion):
        run_train_on_cpu(
            model=model,
            training_dataloader=dataloader,
            results_dir=str(tmp_path),
            epochs=1,
            log_every=100,
            use_diffusion=True,
            precision="fp32",
            learning_rate=0.1,
            run_name="diffusion-skip-loss",
        )

    assert diffusion.calls == 2
    log_text = (tmp_path / "diffusion-skip-loss" / "log.txt").read_text()
    assert "Skipping non-finite training loss" in log_text


def test_train_diffusion_validation_passes_diffusion_to_validation_loop(tmp_path):
    class SaveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, t, conditioned=None):
            return x * self.weight

        def save(self, path):
            torch.save(self.state_dict(), path)

    class FakeDiffusion:
        num_timesteps = 5

        def training_losses(self, model, target, timestep, model_kwargs):
            prediction = model(target, timestep, **model_kwargs)
            return {"loss": prediction.flatten(start_dim=1).mean(dim=1)}

    diffusion = FakeDiffusion()
    dataloader = DataLoader(
        TensorDataset(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1)),
        batch_size=1,
    )
    validation_dataloader = DataLoader(
        TensorDataset(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1)),
        batch_size=1,
    )

    with patch("widitapp.training.create_diffusion", return_value=diffusion), patch(
        "widitapp.training._run_validation_loop", return_value={"loss": 0.2}
    ) as validation_loop:
        run_train_on_cpu(
            model=SaveModel(),
            training_dataloader=dataloader,
            validation_dataloader=validation_dataloader,
            results_dir=str(tmp_path),
            epochs=1,
            log_every=100,
            use_diffusion=True,
            precision="fp32",
            run_name="diffusion-validation",
        )

    validation_loop.assert_called_once()
    _, kwargs = validation_loop.call_args
    assert kwargs["diffusion"] is diffusion
    assert kwargs["use_diffusion"] is True
    assert kwargs["extra_criteria"] is None
    assert (tmp_path / "diffusion-validation" / "checkpoints" / "best.pt").exists()


def test_build_loss_fn_returns_module_instance_unchanged():
    criterion = torch.nn.L1Loss()

    assert build_loss_fn(criterion) is criterion


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("mse", torch.nn.MSELoss),
        ("l2", torch.nn.MSELoss),
        ("smoothl1", torch.nn.SmoothL1Loss),
        ("huber", torch.nn.SmoothL1Loss),
    ],
)
def test_build_loss_fn_accepts_known_names(name, expected_type):
    criterion = build_loss_fn(name)

    assert isinstance(criterion, expected_type)
    assert criterion.reduction == "mean"


def test_build_loss_fn_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown loss_fn"):
        build_loss_fn("mae")


def test_build_val_log_payload_accepts_float():
    val_loss_value, payload = build_val_log_payload(
        1.25,
        use_diffusion=False,
        epoch=3,
        train_steps=10,
    )
    assert val_loss_value == 1.25
    assert payload["val/loss"] == 1.25
    assert payload["epoch"] == 3
    assert payload["global_step"] == 10
    assert "val/mse" not in payload
    assert "val/smoothl1" not in payload


def test_build_val_log_payload_accepts_dict():
    val_loss_value, payload = build_val_log_payload(
        {"loss": 0.5, "mse": 0.6, "smoothl1": 0.4},
        use_diffusion=False,
        epoch=1,
        train_steps=5,
    )
    assert val_loss_value == 0.5
    assert payload["val/loss"] == 0.5
    assert payload["val/mse"] == 0.6
    assert payload["val/smoothl1"] == 0.4


def test_build_val_log_payload_omits_extra_metrics_for_diffusion():
    val_loss_value, payload = build_val_log_payload(
        {"loss": 0.5, "mse": 0.6, "smoothl1": 0.4},
        use_diffusion=True,
        epoch=1,
        train_steps=5,
    )

    assert val_loss_value == 0.5
    assert payload == {"val/loss": 0.5, "epoch": 1, "global_step": 5}


def test_build_val_log_payload_uses_nan_for_missing_dict_loss():
    val_loss_value, payload = build_val_log_payload(
        {},
        use_diffusion=False,
        epoch=1,
        train_steps=5,
    )

    assert math.isnan(val_loss_value)
    assert math.isnan(payload["val/loss"])
    assert math.isnan(payload["val/mse"])
    assert math.isnan(payload["val/smoothl1"])


def test_is_loss_finite_accepts_finite_loss():
    accelerator = Accelerator(mixed_precision="no")

    assert is_loss_finite(
        torch.tensor(1.0, device=accelerator.device),
        accelerator,
    )


def test_is_loss_finite_rejects_nan_loss():
    accelerator = Accelerator(mixed_precision="no")

    assert not is_loss_finite(
        torch.tensor(float("nan"), device=accelerator.device),
        accelerator,
    )


def test_is_loss_finite_rejects_inf_loss():
    accelerator = Accelerator(mixed_precision="no")

    assert not is_loss_finite(
        torch.tensor(float("inf"), device=accelerator.device),
        accelerator,
    )


def test_loss_value_for_logging_returns_scalar_value():
    assert loss_value_for_logging(torch.tensor(1.25)) == 1.25


def test_loss_value_for_logging_returns_nan_for_non_scalar():
    assert math.isnan(loss_value_for_logging(torch.tensor([1.0, 2.0])))


def test_is_finite_tensor_accepts_finite_tensor():
    accelerator = Accelerator(mixed_precision="no")

    assert is_finite_tensor(
        torch.tensor([1.0, 2.0], device=accelerator.device),
        accelerator,
    )


def test_is_finite_tensor_rejects_nan_tensor():
    accelerator = Accelerator(mixed_precision="no")

    assert not is_finite_tensor(
        torch.tensor([1.0, float("nan")], device=accelerator.device),
        accelerator,
    )


def test_is_finite_tensor_rejects_inf_tensor():
    accelerator = Accelerator(mixed_precision="no")

    assert not is_finite_tensor(
        torch.tensor([1.0, float("inf")], device=accelerator.device),
        accelerator,
    )


def test_is_cuda_out_of_memory_accepts_torch_oom():
    out_of_memory_error = getattr(torch, "OutOfMemoryError", RuntimeError)

    assert is_cuda_out_of_memory(out_of_memory_error("CUDA out of memory"))


def test_is_cuda_out_of_memory_accepts_runtime_cuda_oom():
    assert is_cuda_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 10 MiB."))


def test_is_cuda_out_of_memory_rejects_unrelated_runtime_error():
    assert not is_cuda_out_of_memory(RuntimeError("invalid tensor shape"))


def test_tensor_info_for_logging_includes_shape_dtype_and_device():
    tensor = torch.zeros(2, 3)

    info = tensor_info_for_logging(tensor)

    assert "shape=(2, 3)" in info
    assert "dtype=torch.float32" in info
    assert "device=cpu" in info


def test_clear_after_cuda_oom_clears_gradients_and_cuda_cache():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty_cache:
        clear_after_cuda_oom(optimizer)

    assert all(parameter.grad is None for parameter in model.parameters())
    empty_cache.assert_called_once_with()
