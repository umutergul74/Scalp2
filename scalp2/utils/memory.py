"""GPU memory monitoring and optimization for Google Colab T4."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def get_gpu_memory() -> dict | None:
    """Get current GPU memory usage.

    Returns:
        Dict with allocated, reserved, and total memory in MB, or None if no GPU.
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2

    return {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "total_mb": round(total, 1),
        "free_mb": round(total - allocated, 1),
        "utilization_pct": round(100 * allocated / total, 1),
    }


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory status."""
    mem = get_gpu_memory()
    if mem:
        logger.info(
            "%sGPU Memory: %.1f/%.1f MB (%.1f%% used)",
            f"{prefix}: " if prefix else "",
            mem["allocated_mb"],
            mem["total_mb"],
            mem["utilization_pct"],
        )


def estimate_batch_memory(
    batch_size: int,
    seq_len: int,
    n_features: int,
    model_params: int,
    use_amp: bool = True,
) -> dict:
    """Pre-flight memory estimation for training.

    Args:
        batch_size: Training batch size.
        seq_len: Sequence length.
        n_features: Number of input features.
        model_params: Total trainable parameters.
        use_amp: Whether AMP/FP16 is enabled.

    Returns:
        Dict with estimated memory requirements.
    """
    bytes_per_float = 4  # float32

    input_bytes = batch_size * seq_len * n_features * bytes_per_float
    # AdamW: params + grads + 2 optimizer states
    model_bytes = model_params * bytes_per_float * 4
    # Rough activation estimate
    activation_bytes = batch_size * model_params * bytes_per_float * 0.5

    amp_factor = 0.6 if use_amp else 1.0
    total_bytes = (input_bytes + model_bytes + activation_bytes) * amp_factor

    total_mb = total_bytes / 1024**2

    return {
        "input_mb": round(input_bytes / 1024**2, 1),
        "model_mb": round(model_bytes / 1024**2, 1),
        "activation_mb": round(activation_bytes * amp_factor / 1024**2, 1),
        "total_estimated_mb": round(total_mb, 1),
        "fits_t4": total_mb < 12000,  # T4 has ~12GB usable
    }


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast DataFrame dtypes for memory efficiency.

    float64 → float32, int64 → int32 where safe.
    """
    df = df.copy()
    initial_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    for col in df.select_dtypes(include=["int64"]).columns:
        if (
            df[col].min() >= np.iinfo(np.int32).min
            and df[col].max() <= np.iinfo(np.int32).max
        ):
            df[col] = df[col].astype(np.int32)

    final_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(
        "DataFrame memory: %.1f MB → %.1f MB (%.0f%% reduction)",
        initial_mem,
        final_mem,
        100 * (1 - final_mem / max(initial_mem, 1e-10)),
    )
    return df
