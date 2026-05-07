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


def test_app_model_loads_checkpoint_when_provided(tmp_path):
    app = WiDiTApp()
    checkpoint = tmp_path / "model.pt"
    loaded_model = object()

    with patch("widit.load_model", return_value=loaded_model) as load_model:
        model = app.model(checkpoint=checkpoint)

    assert model is loaded_model
    load_model.assert_called_once_with(checkpoint)


@pytest.mark.parametrize("dim_alias", ["spatial_dim", "spatial_dims"])
def test_app_model_accepts_spatial_dimension_aliases(dim_alias):
    app = WiDiTApp()
    built_model = torch.nn.Linear(1, 1)

    with patch("widit.WiDiT", return_value=built_model) as widit:
        model = WiDiTApp.model.func(
            app,
            use_diffusion=True,
            in_channels=2,
            hidden_size=32,
            depth=2,
            num_heads=4,
            patch_size=2,
            window_size=3,
            mlp_ratio=2.5,
            use_flash_attention=False,
            timestep_embed_dim=16,
            **{dim_alias: 3},
        )

    assert model is built_model
    widit.assert_called_once_with(
        in_channels=2,
        out_channels=2,
        use_conditioning=True,
        window_size=3,
        mlp_ratio=2.5,
        use_flash_attention=False,
        timestep_embed_dim=16,
        spatial_dim=3,
        hidden_size=32,
        depth=2,
        num_heads=4,
        patch_size=2,
    )


def test_app_model_respects_explicit_out_channels_and_conditioning():
    app = WiDiTApp()
    built_model = torch.nn.Linear(1, 1)

    with patch("widit.WiDiT", return_value=built_model) as widit:
        model = app.model(
            use_diffusion=True,
            use_conditioning=False,
            out_channels=7,
            hidden_size=32,
            depth=2,
            num_heads=4,
            patch_size=2,
        )

    assert model is built_model
    assert widit.call_args.kwargs["out_channels"] == 7
    assert widit.call_args.kwargs["use_conditioning"] is False


def test_app_model_uses_preset_instantiator():
    app = WiDiTApp()
    built_model = torch.nn.Linear(1, 1)
    preset_instantiator = Mock(return_value=built_model)

    with patch("widit.PRESETS", {"tiny": preset_instantiator}):
        model = app.model(
            preset="tiny",
            use_diffusion=False,
            in_channels=1,
            window_size=5,
            mlp_ratio=3.0,
            use_flash_attention=False,
            timestep_embed_dim=8,
        )

    assert model is built_model
    preset_instantiator.assert_called_once_with(
        in_channels=1,
        out_channels=1,
        use_conditioning=False,
        window_size=5,
        mlp_ratio=3.0,
        use_flash_attention=False,
        timestep_embed_dim=8,
    )


def test_app_model_rejects_unknown_preset():
    app = WiDiTApp()

    with pytest.raises(AssertionError, match="not in PRESETS"):
        app.model(preset="missing")


def test_app_model_unet_passes_constructor_arguments():
    app = WiDiTApp()
    built_model = torch.nn.Conv2d(1, 1, 1)

    with patch("widit.Unet", return_value=built_model) as unet:
        model = app.model(
            unet=True,
            use_diffusion=True,
            use_conditioning=False,
            in_channels=2,
            out_channels=4,
            dim=3,
            filters=16,
            kernel=5,
            layers=6,
            timestep_embed_dim=32,
        )

    assert model is built_model
    unet.assert_called_once_with(
        in_channels=2,
        out_channels=4,
        filters=16,
        kernel_size=5,
        layers=6,
        spatial_dim=3,
        use_conditioning=False,
        timestep_embed_dim=32,
    )


def test_app_model_verbose_prints_summary(capsys):
    app = WiDiTApp()
    built_model = torch.nn.Linear(2, 1)

    with patch("widit.WiDiT", return_value=built_model):
        model = app.model(
            use_diffusion=False,
            hidden_size=32,
            depth=2,
            num_heads=4,
            patch_size=2,
            verbose=True,
        )

    captured = capsys.readouterr()
    assert model is built_model
    assert "Model:" in captured.out
    assert "Model Summary:" in captured.out
    assert "Model has 3 parameters" in captured.out
    assert "Model has 3 trainable parameters" in captured.out


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
