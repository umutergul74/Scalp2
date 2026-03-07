"""Wavelet-based denoising for price and volume series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt


def wavelet_denoise(
    series: pd.Series,
    wavelet: str = "sym5",
    level: int = 4,
    threshold_mode: str = "soft",
    window: int = 256,
) -> pd.Series:
    """Apply causal wavelet denoising using a rolling window.

    For each point t, denoises using only data in [t-window, t] to prevent
    look-ahead bias. Uses VisuShrink (universal threshold) estimated from
    the median absolute deviation of the finest detail coefficients.

    Args:
        series: Input time series (e.g. close prices).
        wavelet: Wavelet family ('sym5' avoids phase shift).
        level: Decomposition level. Level 4 on 15m captures ~4h structures.
        threshold_mode: 'soft' preserves signal shape; 'hard' for sharper edges.
        window: Rolling window size for causal denoising.

    Returns:
        Denoised series (NaN for first `window` values).
    """
    values = series.values.astype(np.float64)
    result = np.full(len(values), np.nan)

    for i in range(window, len(values)):
        segment = values[i - window : i]
        result[i] = _denoise_segment(segment, wavelet, level, threshold_mode)

    return pd.Series(result, index=series.index, name=f"{series.name}_denoised")


def wavelet_denoise_fast(
    series: pd.Series,
    wavelet: str = "sym5",
    level: int = 4,
    threshold_mode: str = "soft",
) -> pd.Series:
    """Fast non-causal wavelet denoising of the full series.

    Use ONLY during offline feature building where the full training
    window is known. NOT suitable for live inference — use `wavelet_denoise`
    (causal version) instead.

    Args:
        series: Input time series.
        wavelet: Wavelet family.
        level: Decomposition level.
        threshold_mode: Thresholding mode.

    Returns:
        Denoised series (same length as input).
    """
    values = series.values.astype(np.float64)

    coeffs = pywt.wavedec(values, wavelet, level=level)

    # Estimate noise std from finest detail coefficients via MAD
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(values)))

    # Threshold all detail coefficients, keep approximation intact
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))

    reconstructed = pywt.waverec(denoised_coeffs, wavelet)

    # pywt.waverec may return array 1 element longer due to padding
    reconstructed = reconstructed[: len(values)]

    return pd.Series(
        reconstructed.astype(np.float32),
        index=series.index,
        name=f"{series.name}_denoised",
    )


def _denoise_segment(
    segment: np.ndarray,
    wavelet: str,
    level: int,
    threshold_mode: str,
) -> float:
    """Denoise a single segment and return the last value."""
    coeffs = pywt.wavedec(segment, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma < 1e-10:
        return segment[-1]

    threshold = sigma * np.sqrt(2 * np.log(len(segment)))

    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))

    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    return float(reconstructed[-1])
