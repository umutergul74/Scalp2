"""Purged Walk-Forward Cross-Validation — rolling window with embargo."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from scalp2.config import WalkForwardConfig

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Indices for a single walk-forward fold."""

    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def train_indices(self) -> np.ndarray:
        return np.arange(self.train_start, self.train_end)

    @property
    def val_indices(self) -> np.ndarray:
        return np.arange(self.val_start, self.val_end)

    @property
    def test_indices(self) -> np.ndarray:
        return np.arange(self.test_start, self.test_end)


class PurgedWalkForwardCV:
    """Rolling-window walk-forward CV with purge and embargo.

    Timeline per fold:
        |---train---|--purge--|---val---|--purge--|---test---|--embargo--|

    The purge gap between train/val and val/test prevents label leakage
    when labels are computed using future bars (e.g. triple barrier).

    The embargo after the test set prevents the next fold's training data
    from including bars whose labels might depend on test-period prices.

    This is a ROLLING window (fixed train size), NOT expanding.
    """

    def __init__(self, config: WalkForwardConfig):
        self.train_size = config.train_bars
        self.val_size = config.val_bars
        self.test_size = config.test_bars
        self.purge_size = config.purge_bars
        self.embargo_size = config.embargo_bars
        self.step_size = config.step_bars

    @property
    def fold_total_size(self) -> int:
        """Total bars consumed by one fold (including purge/embargo)."""
        return (
            self.train_size
            + self.purge_size
            + self.val_size
            + self.purge_size
            + self.test_size
            + self.embargo_size
        )

    def n_folds(self, n_samples: int) -> int:
        """Compute the number of folds for a given dataset size."""
        min_required = self.fold_total_size
        if n_samples < min_required:
            return 0

        remaining = n_samples - min_required
        return 1 + remaining // self.step_size

    def split(
        self, n_samples: int
    ) -> Iterator[WalkForwardFold]:
        """Generate purged walk-forward splits.

        Args:
            n_samples: Total number of samples in the dataset.

        Yields:
            WalkForwardFold with train/val/test index ranges.
        """
        fold_idx = 0
        offset = 0

        while True:
            train_start = offset
            train_end = train_start + self.train_size

            val_start = train_end + self.purge_size
            val_end = val_start + self.val_size

            test_start = val_end + self.purge_size
            test_end = test_start + self.test_size

            # Check if fold fits within the data
            if test_end > n_samples:
                break

            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )

            logger.debug(
                "Fold %d: train[%d:%d] val[%d:%d] test[%d:%d]",
                fold_idx,
                train_start,
                train_end,
                val_start,
                val_end,
                test_start,
                test_end,
            )

            yield fold

            fold_idx += 1
            offset += self.step_size

        logger.info(
            "Walk-forward CV: %d folds from %d samples "
            "(train=%d, val=%d, test=%d, purge=%d, embargo=%d, step=%d)",
            fold_idx,
            n_samples,
            self.train_size,
            self.val_size,
            self.test_size,
            self.purge_size,
            self.embargo_size,
            self.step_size,
        )

    def validate_no_overlap(self, n_samples: int) -> bool:
        """Verify that no fold has overlapping train/val/test sets."""
        for fold in self.split(n_samples):
            train = set(range(fold.train_start, fold.train_end))
            val = set(range(fold.val_start, fold.val_end))
            test = set(range(fold.test_start, fold.test_end))

            if train & val:
                logger.error("Fold %d: train/val overlap!", fold.fold_idx)
                return False
            if train & test:
                logger.error("Fold %d: train/test overlap!", fold.fold_idx)
                return False
            if val & test:
                logger.error("Fold %d: val/test overlap!", fold.fold_idx)
                return False

            # Verify purge gap
            if fold.val_start - fold.train_end < self.purge_size:
                logger.error("Fold %d: insufficient purge between train/val!", fold.fold_idx)
                return False
            if fold.test_start - fold.val_end < self.purge_size:
                logger.error("Fold %d: insufficient purge between val/test!", fold.fold_idx)
                return False

        return True
