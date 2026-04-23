"""GRU encoder with temporal attention pooling for long-term dependencies."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Attention-based pooling over GRU hidden states.

    Instead of only using the last hidden state (which loses early
    information), this module learns which timesteps are most
    informative and computes a weighted average.

    This prevents the model from being biased toward only the most
    recent bars and allows early chart formations to influence the output.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, gru_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gru_output: (batch, seq_len, hidden_size)
        Returns:
            (batch, hidden_size) — attention-weighted pooled representation
        """
        # Compute attention scores
        scores = self.attn(gru_output)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)

        # Weighted sum
        return (gru_output * weights).sum(dim=1)  # (B, H)


class GRUEncoder(nn.Module):
    """Gated Recurrent Unit encoder with optional attention pooling.

    2 layers, unidirectional (causal), hidden_size=128.
    ~190K parameters. 

    Two output modes:
        - attention_pooling=False: last hidden state of top layer (original)
        - attention_pooling=True: learned weighted average of all timesteps

    GRU chosen over LSTM: fewer parameters (3 gates vs 4),
    comparable performance on sequences under 200 steps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        attention_pooling: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = attention_pooling

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_dim = hidden_size * self.num_directions

        if self.use_attention:
            self.attention = TemporalAttention(self.output_dim)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
            return_sequence: If True, return full temporal sequence.
        Returns:
            If return_sequence: (batch, seq_len, hidden_size * num_directions)
            Else: (batch, hidden_size * num_directions)
        """
        # output: (batch, seq_len, hidden * directions)
        # h_n: (num_layers * directions, batch, hidden)
        output, h_n = self.gru(x)

        if return_sequence:
            return output  # (B, T, H)

        if self.use_attention:
            return self.attention(output)

        if self.num_directions == 2:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2]  # Last forward layer
            h_backward = h_n[-1]  # Last backward layer
            return torch.cat([h_forward, h_backward], dim=1)
        else:
            # Last layer, last direction
            return h_n[-1]
