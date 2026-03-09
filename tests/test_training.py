import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from widitapp.training import (
    _run_validation_loop,
    build_loss_fn,
    create_logger,
    get_state_dict_for_saving,
    requires_grad,
    update_ema,
)
from accelerate import Accelerator


class IdentityModel(torch.nn.Module):
    def forward(self, x, timestep=None, **kwargs):
        return x


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
