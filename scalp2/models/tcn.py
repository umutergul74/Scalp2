"""Temporal Convolutional Network — dilated causal convolutions."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


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
    """Single TCN residual block.

    Architecture:
        x → CausalConv → ReLU → Dropout → CausalConv → ReLU → Dropout → + residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 conv for residual connection when dimensions change
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        res = self.residual(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder.

    4 layers with exponentially increasing dilations [1, 2, 4, 8].
    Receptive field with kernel_size=3, 4 layers:
        RF = 1 + 2*(3-1)*(1+2+4+8) = 61 bars = 15.25 hours on 15m data.

    Output: last timestep hidden state (batch, num_channels[-1]).
    """

    def __init__(
        self,
        input_channels: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
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
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features) — time-major input
        Returns:
            (batch, output_dim) — last timestep representation
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Take the last timestep
        return out[:, :, -1]
