"""GRU encoder for long-term temporal dependencies."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """Gated Recurrent Unit encoder.

    2 layers, unidirectional (causal), hidden_size=128.
    ~190K parameters. Output: last hidden state of top layer.

    GRU chosen over LSTM: fewer parameters (3 gates vs 4),
    comparable performance on sequences under 200 steps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_dim = hidden_size * self.num_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, hidden_size * num_directions) — last hidden state
        """
        # output: (batch, seq_len, hidden * directions)
        # h_n: (num_layers * directions, batch, hidden)
        _, h_n = self.gru(x)

        if self.num_directions == 2:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2]  # Last forward layer
            h_backward = h_n[-1]  # Last backward layer
            return torch.cat([h_forward, h_backward], dim=1)
        else:
            # Last layer, last direction
            return h_n[-1]
