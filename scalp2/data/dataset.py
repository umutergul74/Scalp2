"""PyTorch Dataset and DataLoader for time series sliding windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ScalpDataset(Dataset):
    """Sliding-window dataset for the hybrid TCN+GRU model.

    Each sample is a (seq_len, n_features) window of features,
    a scalar label, and the forward return for finance-aware losses.

    Supports online data augmentation during training to improve
    generalization without increasing model parameters.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        returns: np.ndarray,
        seq_len: int = 64,
        augment: bool = False,
        noise_std: float = 0.01,
        scale_range: tuple[float, float] = (0.95, 1.05),
    ):
        """
        Args:
            features: (n_samples, n_features) float32 array.
            labels: (n_samples,) int64 array — class labels {0, 1, 2}.
            returns: (n_samples,) float32 array — forward returns for loss.
            seq_len: Sliding window length.
            augment: If True, apply random augmentation during __getitem__.
            noise_std: Std of Gaussian noise to add to features.
            scale_range: (min, max) range for random feature scaling.
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.returns = returns.astype(np.float32)
        self.seq_len = seq_len
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (seq_len, n_features)
            y: scalar label
            r: scalar forward return
        """
        x = self.features[idx : idx + self.seq_len].copy()

        if self.augment:
            x = self._apply_augmentation(x)

        x = torch.from_numpy(x)
        y = torch.tensor(self.labels[idx + self.seq_len - 1], dtype=torch.long)
        r = torch.tensor(self.returns[idx + self.seq_len - 1], dtype=torch.float32)
        return x, y, r

    def _apply_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a single window.

        Three augmentation techniques designed for financial time series:
        1. Gaussian noise injection (50% probability)
        2. Feature-wise random scaling (50% probability)
        3. Temporal masking — zero out random timesteps (20% probability)

        All augmentations preserve the label validity because:
        - Noise is small (std=0.01 on already-scaled features)
        - Scaling is mild (0.95-1.05x)
        - Masking forces the model to work with incomplete data
        """
        # 1. Gaussian noise — simulates market microstructure noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
            x = x + noise

        # 2. Feature-wise scaling — simulates different volatility regimes
        if np.random.random() < 0.5:
            scale = np.random.uniform(
                self.scale_range[0], self.scale_range[1],
                size=(1, x.shape[1])
            ).astype(np.float32)
            x = x * scale

        # 3. Temporal masking — forces robustness to missing data
        if np.random.random() < 0.2:
            n_mask = max(1, int(x.shape[0] * 0.05))  # Mask 5% of timesteps
            mask_idx = np.random.choice(x.shape[0], n_mask, replace=False)
            x[mask_idx] = 0.0

        return x


def create_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    seq_len: int = 64,
    batch_size: int = 256,
    train_ratio: float = 1.0,
    num_workers: int = 0,
    augment_train: bool = False,
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
        augment_train: If True, apply data augmentation to training data.

    Returns:
        Single DataLoader if train_ratio=1.0, else (train_loader, val_loader).
    """
    if train_ratio >= 1.0:
        ds = ScalpDataset(features, labels, returns, seq_len, augment=augment_train)
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
        features[:split_idx], labels[:split_idx], returns[:split_idx],
        seq_len, augment=augment_train,
    )
    val_ds = ScalpDataset(
        features[split_idx:], labels[split_idx:], returns[split_idx:],
        seq_len, augment=False,  # Never augment validation
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
