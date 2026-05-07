from unittest.mock import Mock, patch

import pytest
import torch

from widitapp import WiDiTApp


def test_app_model_builds():
    app = WiDiTApp()

    model = app.model(
        dim=2,
        in_channels=1,
        use_diffusion=False,
        hidden_size=64,
        depth=1,
        num_heads=4,
        patch_size=2,
        window_size=4,
        mlp_ratio=2.0,
        use_flash_attention=False,
    )

    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "config")


def test_app_model_builds_unet():
    from widit import Unet

    app = WiDiTApp()
    model = app.model(
        unet=True,
        dim=2,
        in_channels=1,
        use_diffusion=False,
        filters=32,
        kernel_size=3,
        layers=2,
    )

    assert isinstance(model, Unet)
    assert model.spatial_dim == 2


def test_app_train_builds_model_dataloaders_and_calls_training(tmp_path):
    app = WiDiTApp()
    model = torch.nn.Linear(1, 1)
    training_dataloader = object()
    validation_dataloader = object()
    app.model = Mock(return_value=model)
    app.dataloaders = Mock(return_value=(training_dataloader, validation_dataloader))

    with patch("widitapp.training.train") as train:
        app.train(
            epochs=3,
            log_every=7,
            learning_rate=0.01,
            results_dir=tmp_path,
            use_diffusion=False,
            wandb=True,
            run_name="unit-run",
            batch_size=4,
        )

    app.model.assert_called_once_with(
        use_diffusion=False,
        batch_size=4,
    )
    app.dataloaders.assert_called_once_with(
        batch_size=4,
    )
    train.assert_called_once_with(
        model=model,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        results_dir=tmp_path,
        use_diffusion=False,
        learning_rate=0.01,
        epochs=3,
        log_every=7,
        run_name="unit-run",
        wandb_logging=True,
        wandb_project="WiDiTApp",
    )


# def test_app_train_requires_cuda(monkeypatch):
#     app = WiDiTApp()

#     def _no_cuda():
#         return False

#     monkeypatch.setattr(torch.cuda, "is_available", _no_cuda)

#     with pytest.raises(AssertionError, match="requires at least one GPU"):
#         app.train(
#             epochs=1,
#             use_diffusion=False,
#             dim=2,
#             in_channels=1,
#             hidden_size=64,
#             depth=1,
#             num_heads=4,
#             patch_size=2,
#             window_size=4,
#             mlp_ratio=2.0,
#             use_flash_attention=False,
#             batch_size=1,
#             num_workers=0,
#         )
