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
