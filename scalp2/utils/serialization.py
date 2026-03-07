"""Model save/load helpers for checkpointing and deployment."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_fold_artifacts(
    save_dir: str | Path,
    fold_idx: int,
    model_state: dict,
    scaler,
    top_feature_indices: np.ndarray,
    feature_names: list[str],
    regime_model=None,
    metadata: dict | None = None,
) -> Path:
    """Save all artifacts from a single walk-forward fold.

    Args:
        save_dir: Base directory for checkpoints.
        fold_idx: Fold index.
        model_state: PyTorch model state dict.
        scaler: Fitted sklearn scaler.
        top_feature_indices: Selected feature indices for Stage 2.
        feature_names: All feature column names.
        regime_model: Fitted HMM regime detector.
        metadata: Optional metadata dict.

    Returns:
        Path to saved fold directory.
    """
    fold_dir = Path(save_dir) / f"fold_{fold_idx:03d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch model
    torch.save(model_state, fold_dir / "hybrid_encoder.pt")

    # Scaler
    with open(fold_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Feature selection
    np.save(fold_dir / "top_feature_indices.npy", top_feature_indices)
    with open(fold_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Regime model
    if regime_model is not None:
        with open(fold_dir / "regime_detector.pkl", "wb") as f:
            pickle.dump(regime_model, f)

    # Metadata
    if metadata:
        with open(fold_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, default=str)

    logger.info("Saved fold %d artifacts to %s", fold_idx, fold_dir)
    return fold_dir


def load_fold_artifacts(
    save_dir: str | Path,
    fold_idx: int,
    device: torch.device | None = None,
) -> dict:
    """Load all artifacts from a saved fold.

    Returns:
        Dict with keys: model_state, scaler, top_feature_indices,
        feature_names, regime_detector, metadata.
    """
    fold_dir = Path(save_dir) / f"fold_{fold_idx:03d}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    device = device or torch.device("cpu")

    artifacts = {}

    artifacts["model_state"] = torch.load(
        fold_dir / "hybrid_encoder.pt", map_location=device, weights_only=True
    )

    with open(fold_dir / "scaler.pkl", "rb") as f:
        artifacts["scaler"] = pickle.load(f)  # noqa: S301

    artifacts["top_feature_indices"] = np.load(
        fold_dir / "top_feature_indices.npy"
    )

    with open(fold_dir / "feature_names.json", "r") as f:
        artifacts["feature_names"] = json.load(f)

    regime_path = fold_dir / "regime_detector.pkl"
    if regime_path.exists():
        with open(regime_path, "rb") as f:
            artifacts["regime_detector"] = pickle.load(f)  # noqa: S301

    meta_path = fold_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            artifacts["metadata"] = json.load(f)

    logger.info("Loaded fold %d artifacts from %s", fold_idx, fold_dir)
    return artifacts
