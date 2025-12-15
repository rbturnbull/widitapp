import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        if t.ndim != 1:
            raise ValueError(f"`t` must be 1-D of shape (N,), got {tuple(t.shape)}")

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))

    def init_weights(self, std: float = 0.02) -> None:
        """
        Initialize weights like your previous models.py:
        - Normal(0, std) on the two Linear *weights* in the MLP.
        - Leave biases unchanged.
        """
        nn.init.normal_(self.mlp[0].weight, std=std)
        nn.init.normal_(self.mlp[2].weight, std=std)


def get_conv(spatial_dims: int):
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d


def get_maxpool(spatial_dims: int):
    return nn.MaxPool3d if spatial_dims == 3 else nn.MaxPool2d


def get_upsample_mode(spatial_dims: int):
    return "trilinear" if spatial_dims == 3 else "bilinear"


def conv(
    spatial_dims: int,
    in_channels: int,
    filters: int,
    kernel_size: int,
    padding: int,
    activation: nn.Module | None,
):
    Conv = get_conv(spatial_dims)

    layers = [
        Conv(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
    ]

    if activation is not None:
        layers.append(activation)

    return nn.Sequential(*layers)


def conv_block(
    spatial_dims: int,
    in_channels: int,
    filters: int,
    kernel_size: int,
    padding: int,
    activation: nn.Module,
):
    return nn.Sequential(
        conv(spatial_dims, in_channels, filters, kernel_size, padding, activation),
        conv(spatial_dims, filters, filters, kernel_size, padding, activation),
    )


def down_block(
    spatial_dims: int,
    in_channels: int,
    filters: int,
    kernel_size: int,
    padding: int,
    activation: nn.Module,
):
    MaxPool = get_maxpool(spatial_dims)

    return nn.Sequential(
        MaxPool(kernel_size=2, stride=2),
        conv_block(spatial_dims, in_channels, filters, kernel_size, padding, activation),
    )


def up_block(
    spatial_dims: int,
    in_channels: int,
    filters: int,
    kernel_size: int,
    padding: int,
    activation: nn.Module,
):
    mode = get_upsample_mode(spatial_dims)

    return nn.Sequential(
        conv_block(spatial_dims, in_channels, filters, kernel_size, padding, activation),
        nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
    )


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int,
        layers: int,
        spatial_dims: int = 3,
        out_channels: int|None = None,
        use_conditioning: bool = True,
        timestep_embed_dim: int | None = None,
    ):
        super().__init__()

        assert spatial_dims in (2, 3), "spatial_dims must be 2 or 3"
        assert layers > 0, "Layers must be positive"

        out_channels = out_channels or in_channels

        if use_conditioning:
            in_channels *= 2  # Concatenate conditioning channel
        self.use_conditioning = use_conditioning

        self.layers = layers
        self.spatial_dims = spatial_dims
        padding = kernel_size // 2
        act = nn.ReLU()

        # Initial conv
        self.inconv = conv_block(
            spatial_dims,
            in_channels,
            filters,
            kernel_size,
            padding,
            act,
        )

        # Timestep embedding (shared size projected per block)
        self.timestep_embed_dim = timestep_embed_dim or (filters * 4)
        self.timestep_embedder = TimestepEmbedder(self.timestep_embed_dim)

        self.in_time_proj = nn.Linear(self.timestep_embed_dim, filters)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.down_time_projs = nn.ModuleList()
        for i in range(layers):
            in_f = filters * (2**i)
            out_f = in_f * 2
            self.down_blocks.append(
                down_block(
                    spatial_dims,
                    in_f,
                    out_f,
                    kernel_size,
                    padding,
                    act,
                )
            )
            self.down_time_projs.append(nn.Linear(self.timestep_embed_dim, out_f))

        # Bottleneck upsample
        self.bottleneck = nn.Upsample(
            scale_factor=2,
            mode=get_upsample_mode(spatial_dims),
            align_corners=False,
        )
        self.bottleneck_time_proj = nn.Linear(
            self.timestep_embed_dim, filters * (2**layers)
        )

        # Up path
        self.up_blocks = nn.ModuleList()
        self.up_time_projs = nn.ModuleList()
        for i in range(layers, 1, -1):
            out_f = filters * (2 ** (i - 1))
            in_f = filters * (2**i) + out_f
            self.up_blocks.append(
                up_block(
                    spatial_dims,
                    in_f,
                    out_f,
                    kernel_size,
                    padding,
                    act,
                )
            )
            self.up_time_projs.append(nn.Linear(self.timestep_embed_dim, out_f))

        # Output
        self.outconv = nn.Sequential(
            conv(
                spatial_dims,
                filters * 3,
                filters,
                kernel_size,
                padding,
                act,
            ),
            conv(
                spatial_dims,
                filters,
                out_channels,
                kernel_size=1,
                padding=0,
                activation=None,
            ),
        )
        self.out_time_proj = nn.Linear(self.timestep_embed_dim, filters * 3)

    def _add_timestep(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor | None,
        projection: nn.Linear | None,
    ) -> torch.Tensor:
        if timestep_embedding is None or projection is None:
            return x

        temb = projection(timestep_embedding)
        temb = temb.view(temb.shape[0], temb.shape[1], *([1] * (x.ndim - 2)))
        return x + temb

    def forward(
        self, 
        x, 
        timestep: torch.Tensor | None = None,
        *,
        conditioned: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skip_conn = []
        timestep_embedding = (
            self.timestep_embedder(timestep) if timestep is not None else None
        )

        # Concatenate conditioning if provided
        assert self.use_conditioning == (conditioned is not None), (
            "Conditioning tensor must be provided if and only if the model was "
            "initialized with `use_conditioning=True`."
        )
        if conditioned is not None:
            x = torch.cat((x, conditioned), dim=1)

        x = self.inconv(x)
        x = self._add_timestep(x, timestep_embedding, self.in_time_proj)
        skip_conn.append(x)

        for i in range(self.layers - 1):
            x = self.down_blocks[i](x)
            x = self._add_timestep(x, timestep_embedding, self.down_time_projs[i])
            skip_conn.append(x)

        x = self.down_blocks[-1](x)
        x = self._add_timestep(x, timestep_embedding, self.down_time_projs[-1])

        x = self.bottleneck(x)
        x = self._add_timestep(x, timestep_embedding, self.bottleneck_time_proj)

        for i in range(self.layers - 1):
            skip = skip_conn.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode=get_upsample_mode(self.spatial_dims),
                    align_corners=False,
                )
            x = self.up_blocks[i](torch.cat((skip, x), dim=1))
            x = self._add_timestep(x, timestep_embedding, self.up_time_projs[i])

        skip = skip_conn.pop()
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode=get_upsample_mode(self.spatial_dims),
                align_corners=False,
            )
        x = torch.cat((skip, x), dim=1)
        x = self._add_timestep(x, timestep_embedding, self.out_time_proj)
        x = self.outconv(x)

        return x
