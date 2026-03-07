"""PyTorch Dataset and DataLoader for time series sliding windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ScalpDataset(Dataset):
    """Sliding-window dataset for the hybrid TCN+GRU model.

    Each sample is a (seq_len, n_features) window of features,
    a scalar label, and the forward return for finance-aware losses.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        returns: np.ndarray,
        seq_len: int = 64,
    ):
        """
        Args:
            features: (n_samples, n_features) float32 array.
            labels: (n_samples,) int64 array — class labels {0, 1, 2}.
            returns: (n_samples,) float32 array — forward returns for loss.
            seq_len: Sliding window length.
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.returns = returns.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (seq_len, n_features)
            y: scalar label
            r: scalar forward return
        """
        x = torch.from_numpy(self.features[idx : idx + self.seq_len])
        y = torch.tensor(self.labels[idx + self.seq_len - 1], dtype=torch.long)
        r = torch.tensor(self.returns[idx + self.seq_len - 1], dtype=torch.float32)
        return x, y, r


def create_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    seq_len: int = 64,
    batch_size: int = 256,
    train_ratio: float = 1.0,
    num_workers: int = 0,
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """Create DataLoader(s) from numpy arrays.

    IMPORTANT: No shuffling — time series order must be preserved.

    Args:
        features: (n_samples, n_features) array.
        labels: (n_samples,) array.
        returns: (n_samples,) array.
        seq_len: Window length.
        batch_size: Batch size.
        train_ratio: If < 1.0, split into train/val loaders (temporal split).
        num_workers: DataLoader workers (0 for Colab compatibility).

    Returns:
        Single DataLoader if train_ratio=1.0, else (train_loader, val_loader).
    """
    if train_ratio >= 1.0:
        ds = ScalpDataset(features, labels, returns, seq_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,  # NEVER shuffle time series
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    split_idx = int(len(features) * train_ratio)

    train_ds = ScalpDataset(
        features[:split_idx], labels[:split_idx], returns[:split_idx], seq_len
    )
    val_ds = ScalpDataset(
        features[split_idx:], labels[split_idx:], returns[split_idx:], seq_len
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
