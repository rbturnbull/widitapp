from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import TensorDataset

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


def test_app_datasets_is_not_implemented_on_base_class():
    app = WiDiTApp()

    with pytest.raises(NotImplementedError, match="Datasets method not yet implemented"):
        app.datasets()


def test_app_predict_is_not_implemented_on_base_class():
    app = WiDiTApp()

    with pytest.raises(NotImplementedError, match="Prediction not yet implemented"):
        app.predict()


def test_app_dataloaders_builds_loaders_from_datasets():
    app = WiDiTApp()
    training_dataset = TensorDataset(torch.arange(4).float().unsqueeze(1), torch.arange(4).float().unsqueeze(1))
    validation_dataset = TensorDataset(torch.arange(2).float().unsqueeze(1), torch.arange(2).float().unsqueeze(1))
    app.datasets = Mock(return_value=(training_dataset, validation_dataset))

    training_dataloader, validation_dataloader = app.dataloaders(
        batch_size=2,
        num_workers=0,
    )

    app.datasets.assert_called_once_with()
    assert training_dataloader.dataset is training_dataset
    assert training_dataloader.batch_size == 2
    assert training_dataloader.drop_last is False
    assert training_dataloader.num_workers == 0
    assert validation_dataloader.dataset is validation_dataset
    assert validation_dataloader.batch_size == 2
    assert validation_dataloader.drop_last is False
    assert validation_dataloader.num_workers == 0


def test_app_dataloaders_allows_missing_validation_dataset():
    app = WiDiTApp()
    training_dataset = TensorDataset(torch.arange(4).float().unsqueeze(1), torch.arange(4).float().unsqueeze(1))
    app.datasets = Mock(return_value=(training_dataset, None))

    training_dataloader, validation_dataloader = app.dataloaders(
        batch_size=2,
        num_workers=0,
    )

    assert training_dataloader.dataset is training_dataset
    assert validation_dataloader is None


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
