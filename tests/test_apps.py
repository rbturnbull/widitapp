import torch

from widitapp import WiDiTApp


def test_app_model_builds():
    app = WiDiTApp()

    model = app.model(
        dim=2,
        input_size=32,
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
