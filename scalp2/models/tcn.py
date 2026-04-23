"""Temporal Convolutional Network — dilated causal convolutions with anti-overfitting."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class SpatialDropout1d(nn.Module):
    """Drop entire channels instead of individual activations.

    This forces the model to not rely on any single channel,
    preventing noise memorization in specific feature maps.
    """

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len) with entire channels zeroed out.
        """
        if not self.training or self.p == 0:
            return x
        # Create mask: (batch, channels, 1) — same mask across all timesteps
        mask = torch.bernoulli(
            torch.full((x.size(0), x.size(1), 1), 1 - self.p, device=x.device)
        )
        return x * mask / (1 - self.p)


class SqueezeExcite1d(nn.Module):
    """Channel attention — learn which channels are informative.

    Adaptively re-weights channels: amplifies useful channels
    and suppresses noisy ones that carry no predictive signal.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len) re-weighted by channel importance.
        """
        w = self.fc(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


class CausalConv1d(nn.Module):
    """Causal (left-padded) 1D convolution with dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len) — causally padded
        """
        out = self.conv(x)
        # Remove future padding (keep only causal)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """Single TCN residual block with SE, Spatial Dropout, Stochastic Depth.

    Architecture:
        x → CausalConv → ReLU → SpatialDrop → CausalConv → ReLU → SpatialDrop
          → SE → + residual (with stochastic depth)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3,
        spatial_dropout: bool = True,
        use_se: bool = True,
        stochastic_depth_p: float = 0.0,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()

        # Dropout: spatial (channel-wise) or standard
        if spatial_dropout:
            self.drop1 = SpatialDropout1d(dropout)
            self.drop2 = SpatialDropout1d(dropout)
        else:
            self.drop1 = nn.Dropout(dropout)
            self.drop2 = nn.Dropout(dropout)

        # Channel attention
        self.se = SqueezeExcite1d(out_channels) if use_se else nn.Identity()

        # 1x1 conv for residual connection when dimensions change
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Stochastic depth: probability of dropping this block's transform
        self.stochastic_depth_p = stochastic_depth_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        res = self.residual(x)

        # Stochastic depth: during training, randomly skip the conv path
        if self.training and self.stochastic_depth_p > 0:
            if torch.rand(1).item() < self.stochastic_depth_p:
                return res

        out = self.drop1(self.relu(self.conv1(x)))
        out = self.drop2(self.relu(self.conv2(out)))
        out = self.se(out)

        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder with anti-overfitting.

    4 layers with exponentially increasing dilations [1, 2, 4, 8].
    Receptive field with kernel_size=3, 4 layers:
        RF = 1 + 2*(3-1)*(1+2+4+8) = 61 bars = 15.25 hours on 15m data.

    Anti-overfitting additions:
        - Spatial Dropout: drops entire channels, not individual activations.
        - Squeeze-and-Excite: learns which channels carry signal vs noise.
        - Stochastic Depth: randomly bypasses blocks, prevents co-adaptation.

    Output: last timestep hidden state (batch, num_channels[-1]).
    """

    def __init__(
        self,
        input_channels: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        spatial_dropout: bool = True,
        squeeze_excite: bool = True,
        stochastic_depth: float = 0.1,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]

        layers = []
        n_layers = len(num_channels)
        for i in range(n_layers):
            in_ch = input_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            # Stochastic depth increases linearly with layer depth
            sd_p = stochastic_depth * (i / max(n_layers - 1, 1))
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, dilation, dropout,
                    spatial_dropout=spatial_dropout,
                    use_se=squeeze_excite,
                    stochastic_depth_p=sd_p,
                )
            )

        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features) — time-major input
            return_sequence: If True, return full temporal sequence instead of last timestep.
        Returns:
            If return_sequence: (batch, seq_len, output_dim) — full sequence
            Else: (batch, output_dim) — last timestep representation
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        if return_sequence:
            return out.transpose(1, 2)  # (B, T, C)
        # Take the last timestep
        return out[:, :, -1]
