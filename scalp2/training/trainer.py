"""Stage-1 PyTorch training loop — AMP, AdamW, finance-aware loss."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from scalp2.config import TrainingConfig
from scalp2.data.dataset import ScalpDataset, create_dataloaders
from scalp2.losses.log_mdd_loss import LogMDDLoss, compute_combined_loss
from scalp2.losses.sharpe_loss import SharpeLoss
from scalp2.models.hybrid import HybridEncoder
from scalp2.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class Stage1Trainer:
    """Training loop for the HybridEncoder (TCN+GRU).

    Key features:
        - AdamW optimizer with weight decay
        - ReduceLROnPlateau scheduler
        - AMP/FP16 mixed precision for T4 GPU
        - Gradient clipping at max_norm=1.0
        - Combined CE + finance-aware loss with alpha annealing
        - Early stopping and model checkpointing
    """

    def __init__(
        self,
        model: HybridEncoder,
        config: TrainingConfig,
        device: torch.device | None = None,
        checkpoint_dir: str | Path = "./checkpoints",
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=tuple(config.optimizer.betas),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=config.scheduler.patience,
            factor=config.scheduler.factor,
            min_lr=config.scheduler.min_lr,
        )

        # AMP scaler
        self.scaler = GradScaler('cuda', enabled=config.use_amp)
        self.use_amp = config.use_amp

        # Auxiliary loss
        if config.loss.auxiliary == "sharpe":
            self.aux_loss_fn = SharpeLoss()
        else:
            self.aux_loss_fn = LogMDDLoss()

        # Callbacks
        self.checkpoint = ModelCheckpoint(save_dir=checkpoint_dir)

        logger.info(
            "Stage1Trainer initialized: device=%s, AMP=%s, params=%d",
            self.device,
            self.use_amp,
            model.count_parameters(),
        )

    def _compute_alpha(self, epoch: int) -> float:
        """Compute loss weighting alpha with linear annealing."""
        cfg = self.config.loss
        if epoch >= cfg.alpha_anneal_epochs:
            return cfg.alpha_end
        progress = epoch / cfg.alpha_anneal_epochs
        return cfg.alpha_start + (cfg.alpha_end - cfg.alpha_start) * progress

    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor | None:
        """Compute inverse-frequency class weights."""
        if self.config.class_weights != "inverse_frequency":
            return None

        classes, counts = np.unique(labels, return_counts=True)
        weights = len(labels) / (len(classes) * counts)
        weight_tensor = torch.zeros(3, dtype=torch.float32)
        for cls, w in zip(classes, weights):
            weight_tensor[int(cls)] = w

        return weight_tensor.to(self.device)

    def train_one_fold(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        train_returns: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        val_returns: np.ndarray,
        fold_idx: int,
        seq_len: int = 64,
    ) -> dict:
        """Full training loop for one walk-forward fold.

        Args:
            train_features: (n_train, n_features) training features.
            train_labels: (n_train,) training labels.
            train_returns: (n_train,) training forward returns.
            val_features: (n_val, n_features) validation features.
            val_labels: (n_val,) validation labels.
            val_returns: (n_val,) validation forward returns.
            fold_idx: Fold index for checkpointing.
            seq_len: Sequence length for sliding windows.

        Returns:
            Dict with training history and best metrics.
        """
        train_loader = create_dataloaders(
            train_features, train_labels, train_returns,
            seq_len=seq_len, batch_size=self.config.batch_size,
        )
        val_loader = create_dataloaders(
            val_features, val_labels, val_returns,
            seq_len=seq_len, batch_size=self.config.batch_size,
        )

        class_weights = self._compute_class_weights(train_labels)
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping.patience,
            min_delta=self.config.early_stopping.min_delta,
        )

        history = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(self.config.max_epochs):
            alpha = self._compute_alpha(epoch)

            # Train
            train_loss = self._train_epoch(train_loader, class_weights, alpha)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = self._validate(val_loader, class_weights, alpha)
            history["val_loss"].append(val_loss)

            # Scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["lr"].append(current_lr)

            # Checkpoint
            self.checkpoint.step(val_loss, self.model, fold_idx, epoch)

            # Log
            if epoch % 5 == 0 or epoch == self.config.max_epochs - 1:
                logger.info(
                    "Fold %d Epoch %d/%d: train_loss=%.6f val_loss=%.6f lr=%.2e alpha=%.3f",
                    fold_idx, epoch + 1, self.config.max_epochs,
                    train_loss, val_loss, current_lr, alpha,
                )

            # Early stopping
            if early_stopping.step(val_loss):
                break

        # Restore best model
        self.checkpoint.load_best(self.model)

        return {
            "history": history,
            "best_val_loss": self.checkpoint.best_loss,
            "epochs_trained": len(history["train_loss"]),
            "fold_idx": fold_idx,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        class_weights: torch.Tensor | None,
        alpha: float,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y, batch_r in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_r = batch_r.to(self.device)

            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                logits, _ = self.model(batch_x)
                loss, _ = compute_combined_loss(
                    logits, batch_y, batch_r,
                    class_weights, alpha, self.aux_loss_fn,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        class_weights: torch.Tensor | None,
        alpha: float,
    ) -> float:
        """Run validation pass."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y, batch_r in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_r = batch_r.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                logits, _ = self.model(batch_x)
                loss, _ = compute_combined_loss(
                    logits, batch_y, batch_r,
                    class_weights, alpha, self.aux_loss_fn,
                )

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def extract_latents(
        self,
        features: np.ndarray,
        seq_len: int = 64,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Extract latent vectors from trained model for Stage 2.

        Args:
            features: (n_samples, n_features) array.
            seq_len: Sequence length.
            batch_size: Inference batch size.

        Returns:
            (n_samples - seq_len, latent_dim) latent vectors.
        """
        self.model.eval()
        dummy_labels = np.zeros(len(features), dtype=np.int64)
        dummy_returns = np.zeros(len(features), dtype=np.float32)

        ds = ScalpDataset(features, dummy_labels, dummy_returns, seq_len)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        latents = []
        for batch_x, _, _ in loader:
            batch_x = batch_x.to(self.device)
            with autocast('cuda', enabled=self.use_amp):
                latent = self.model.extract_latent(batch_x)
            latents.append(latent.cpu().numpy())

        return np.concatenate(latents, axis=0)
