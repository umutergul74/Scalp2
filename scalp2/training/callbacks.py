"""Training callbacks — early stopping and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    self.patience,
                )

        return self.should_stop


class ModelCheckpoint:
    """Save model when validation loss improves.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Metric to monitor ('val_loss').
    """

    def __init__(self, save_dir: str | Path = "./checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.best_state = None

    def step(
        self,
        val_loss: float,
        model: torch.nn.Module,
        fold_idx: int,
        epoch: int,
    ) -> bool:
        """Save model if validation loss improved.

        Returns:
            True if model was saved (new best).
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

            path = self.save_dir / f"fold_{fold_idx:03d}_best.pt"
            torch.save(
                {
                    "model_state_dict": self.best_state,
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "fold_idx": fold_idx,
                },
                path,
            )
            logger.info(
                "Fold %d: saved checkpoint at epoch %d (val_loss=%.6f)",
                fold_idx,
                epoch,
                val_loss,
            )
            return True
        return False

    def load_best(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load the best model state into the given model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
