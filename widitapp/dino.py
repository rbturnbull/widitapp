# convnext_dino.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ============================================================
# Timestep embedding (same semantics as your example)
# ============================================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        if t.ndim != 1:
            raise ValueError(f"`t` must be 1-D, got {t.shape}")

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb)


# ============================================================
# Basic blocks
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Stem conversion (channels + stride)
# ============================================================

def convert_stem_conv(conv: nn.Conv2d, in_channels: int, stride: int):
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
            rep = conv.weight.repeat(1, in_channels // 3 + 1, 1, 1)
            new_conv.weight.copy_(rep[:, :in_channels] / (in_channels / 3))

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


# ============================================================
# Encoder
# ============================================================

class DINOv3ConvNeXtEncoder(nn.Module):
    def __init__(self, backbone, num_stages: int):
        super().__init__()
        self.stages = backbone.stages[:num_stages]

    def forward(self, x, temb=None, time_projs=None):
        feats = []
        for i, stage in enumerate(self.stages):
            for layer in stage.downsample_layers:
                x = layer(x)
            for block in stage.layers:
                x = block(x)
            if temb is not None:
                proj = time_projs[i]
                t = proj(temb).view(temb.size(0), -1, 1, 1)
                x = x + t
            feats.append(x)
        return feats


def infer_convnext_channels(backbone, num_stages):
    ch = []
    for stage in backbone.stages[:num_stages]:
        ch.append(stage.layers[0].depthwise_conv.out_channels)
    return tuple(ch)


# ============================================================
# ConvNeXt-UNet with conditioning + timestep
# ============================================================

class ConvNeXtUNet(nn.Module):
    def __init__(
        self,
        backbone,
        channels,
        out_channels,
        use_conditioning: bool,
        timestep_embed_dim: int,
    ):
        super().__init__()

        self.use_conditioning = use_conditioning
        self.encoder = DINOv3ConvNeXtEncoder(backbone, len(channels))

        # timestep
        self.timestep_embedder = (
            TimestepEmbedder(timestep_embed_dim)
            if timestep_embed_dim is not None
            else None
        )

        self.enc_time_projs = nn.ModuleList(
            [nn.Linear(timestep_embed_dim, c) for c in channels]
        ) if timestep_embed_dim else None

        # decoder
        self.decoders = nn.ModuleList()
        self.dec_time_projs = nn.ModuleList()

        for i in range(len(channels) - 1, 0, -1):
            self.decoders.append(
                ConvBlock(channels[i] + channels[i - 1], channels[i - 1])
            )
            if timestep_embed_dim:
                self.dec_time_projs.append(
                    nn.Linear(timestep_embed_dim, channels[i - 1])
                )

        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        *,
        conditioned: torch.Tensor | None = None,
    ):
        input_x = x
        input_hw = x.shape[-2:]

        assert self.use_conditioning == (conditioned is not None), (
            "Conditioning must be provided iff use_conditioning=True"
        )

        if conditioned is not None:
            x = torch.cat([x, conditioned], dim=1)

        temb = self.timestep_embedder(timestep) if timestep is not None else None

        feats = self.encoder(x, temb, self.enc_time_projs)

        x = feats[-1]
        for i, (skip, dec) in enumerate(
            zip(reversed(feats[:-1]), self.decoders)
        ):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
            if temb is not None:
                t = self.dec_time_projs[i](temb).view(temb.size(0), -1, 1, 1)
                x = x + t

        x = self.head(x)

        if x.shape[-2:] != input_hw:
            x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)

        # residual
        if x.shape == input_x.shape:
            x = x + input_x

        return x


# ============================================================
# Factory
# ============================================================

def convnext_unet(
    size: str = "tiny",
    in_channels: int = 1,
    out_channels: int = 1,
    stem_stride: int = 1,
    num_stages: int = 2,
    freeze: bool = False,
    use_conditioning: bool = False,
    timestep_embed_dim: int | None = None,
):
    assert size in ("tiny", "small", "base", "large")
    assert 1 <= num_stages <= 4
    assert stem_stride in (1, 2, 4)

    model_name = f"facebook/dinov3-convnext-{size}-pretrain-lvd1689m"
    backbone = AutoModel.from_pretrained(model_name)

    # stem
    stem = backbone.stages[0].downsample_layers[0]
    backbone.stages[0].downsample_layers[0] = convert_stem_conv(
        stem,
        in_channels * (2 if use_conditioning else 1),
        stem_stride,
    )

    channels = infer_convnext_channels(backbone, num_stages)

    model = ConvNeXtUNet(
        backbone=backbone,
        channels=channels,
        out_channels=out_channels,
        use_conditioning=use_conditioning,
        timestep_embed_dim=timestep_embed_dim,
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
        num_stages=2,
        timestep_embed_dim=256,
        use_conditioning=True,
    )

    input_size = 500
    x = torch.randn(2, 1, input_size, input_size)
    cond = torch.randn(2, 1, input_size, input_size)
    t = torch.randint(0, 1000, (2,))

    y = model(x, timestep=t, conditioned=cond)
    print(y.shape)
