# convnext_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ============================================================
# Basic building blocks
# ============================================================

class ConvBlock(nn.Module):
    """Simple 2×(Conv → Norm → GELU) block for the decoder."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Stem conversion + stride utility
# ============================================================

def convert_stem_conv(
    conv: nn.Conv2d,
    in_channels: int,
    stride: int,
) -> nn.Conv2d:
    """
    Convert Conv2d(3 -> C) to Conv2d(in_channels -> C) and
    optionally reduce stride, preserving pretrained weights.
    """
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )

    with torch.no_grad():
        if in_channels == 3:
            new_conv.weight.copy_(conv.weight)
        elif in_channels == 1:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        else:
            repeat = conv.weight.repeat(1, in_channels // 3 + 1, 1, 1)
            new_conv.weight.copy_(repeat[:, :in_channels] / (in_channels / 3))

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


# ============================================================
# Encoder wrapper (variable depth)
# ============================================================

class DINOv3ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt encoder exposing intermediate feature maps.
    """

    def __init__(self, backbone, num_stages: int):
        super().__init__()
        assert 1 <= num_stages <= len(backbone.stages)
        self.stages = backbone.stages[:num_stages]

    def forward(self, x):
        features = []

        for stage in self.stages:
            for layer in stage.downsample_layers:
                x = layer(x)
            for block in stage.layers:
                x = block(x)
            features.append(x)

        return features


def infer_convnext_channels(backbone, num_stages: int) -> tuple[int, ...]:
    """
    Infer per-stage channel widths.
    """
    channels = []
    for stage in backbone.stages[:num_stages]:
        block = stage.layers[0]
        channels.append(block.depthwise_conv.out_channels)
    return tuple(channels)


# ============================================================
# U-Net (residual, variable depth)
# ============================================================

class ConvNeXtUNet(nn.Module):
    def __init__(
        self,
        backbone,
        channels: tuple[int, ...],
        out_channels: int,
    ):
        super().__init__()

        self.encoder = DINOv3ConvNeXtEncoder(backbone, len(channels))

        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoders.append(
                ConvBlock(channels[i] + channels[i - 1], channels[i - 1])
            )

        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        input_x = x
        input_hw = x.shape[-2:]

        feats = self.encoder(x)

        x = feats[-1]
        for skip, dec in zip(reversed(feats[:-1]), self.decoders):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        x = self.head(x)

        # Restore resolution if stem_stride > 1
        if x.shape[-2:] != input_hw:
            x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)

        # Residual output (SR / denoising)
        if input_x.shape[1] == x.shape[1]:
            x = x + input_x

        return x


# ============================================================
# Factory function
# ============================================================

def convnext_unet(
    size: str = "base",
    in_channels: int = 1,
    out_channels: int = 1,
    stem_stride: int = 1,
    num_stages: int = 2,
    freeze: bool = False,
) -> nn.Module:
    """
    ConvNeXt-U-Net adapted for super-resolution / denoising.
    """

    assert size in ("tiny", "small", "base", "large")
    assert stem_stride in (1, 2, 4)
    assert 1 <= num_stages <= 4

    model_name = f"facebook/dinov3-convnext-{size}-pretrain-lvd1689m"
    backbone = AutoModel.from_pretrained(model_name)

    # Modify stem (channels + stride)
    old_conv = backbone.stages[0].downsample_layers[0]
    backbone.stages[0].downsample_layers[0] = convert_stem_conv(
        old_conv,
        in_channels=in_channels,
        stride=stem_stride,
    )

    channels = infer_convnext_channels(backbone, num_stages)

    model = ConvNeXtUNet(
        backbone=backbone,
        channels=channels,
        out_channels=out_channels,
    )

    if freeze:
        for p in model.encoder.parameters():
            p.requires_grad = False

    return model


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    model = convnext_unet(
        size="tiny",
        in_channels=1,
        out_channels=1,
        stem_stride=1,
        num_stages=4,
        freeze=True,
    )

    input_shape = (2, 1, 256, 256)

    x = torch.randn(*input_shape)
    y = model(x)

    print("Output shape:", y.shape)
    assert y.shape == input_shape, f"Expected {input_shape}, got {y.shape}"
