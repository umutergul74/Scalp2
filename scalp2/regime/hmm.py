"""Hidden Markov Model regime detection — Bull, Bear, Choppy."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from scalp2.config import RegimeConfig

logger = logging.getLogger(__name__)


class RegimeDetector:
    """3-state Gaussian HMM for market regime detection.

    Fitted on [log_return, GK_volatility, volume_zscore].
    States are mapped post-hoc by sorting on mean return:
        - Highest mean return → Bull
        - Lowest mean return  → Bear
        - Middle (highest variance) → Choppy

    Usage:
        detector = RegimeDetector(config)
        detector.fit(train_features)
        probs = detector.predict_proba(test_features)  # (n, 3)
        tradeable = detector.is_tradeable(test_features)  # bool array
    """

    # State indices after remapping
    BULL = 0
    BEAR = 1
    CHOPPY = 2
    STATE_NAMES = {0: "bull", 1: "bear", 2: "choppy"}

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.model = GaussianHMM(
            n_components=config.n_states,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            min_covar=1e-3,
            random_state=42,
            verbose=False,
        )
        self.state_map: dict[int, int] = {}  # HMM state → regime index
        self.feature_names = config.features
        self._fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and validate regime features from DataFrame."""
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Missing regime features: {missing}")

        X = df[self.feature_names].values.astype(np.float64)

        # Replace inf/nan with 0 (can occur in early bars)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit(self, df: pd.DataFrame) -> RegimeDetector:
        """Fit HMM on training data and map states to regimes.

        Args:
            df: Training DataFrame with regime feature columns.

        Returns:
            self (for chaining).
        """
        X = self._prepare_features(df)

        logger.info(
            "Fitting %d-state HMM on %d samples with features: %s",
            self.config.n_states,
            len(X),
            self.feature_names,
        )

        # Standardize features for numerical stability
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._mean) / self._std

        try:
            self.model.fit(X_scaled)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning(
                "HMM fit failed (%s), using uniform regime prior", e
            )
            self.state_map = {0: self.BULL, 1: self.BEAR, 2: self.CHOPPY}
            self._fitted = True
            self._fallback = True
            return self

        self._fallback = False

        # Map states by mean return (first feature = log_return)
        means = self.model.means_[:, 0]
        sorted_idx = np.argsort(means)

        # sorted_idx[0] = lowest return → Bear
        # sorted_idx[-1] = highest return → Bull
        # sorted_idx[1] (middle) → Choppy
        self.state_map = {
            int(sorted_idx[0]): self.BEAR,
            int(sorted_idx[-1]): self.BULL,
            int(sorted_idx[1]): self.CHOPPY,
        }

        logger.info(
            "HMM state mapping: %s",
            {
                hmm_state: self.STATE_NAMES[regime_idx]
                for hmm_state, regime_idx in self.state_map.items()
            },
        )
        logger.info(
            "State means (return): Bull=%.6f, Bear=%.6f, Choppy=%.6f",
            means[sorted_idx[-1]],
            means[sorted_idx[0]],
            means[sorted_idx[1]],
        )

        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities.

        Args:
            df: DataFrame with regime feature columns.

        Returns:
            (n_samples, 3) array with columns [bull_prob, bear_prob, choppy_prob].
        """
        if not self._fitted:
            raise RuntimeError("RegimeDetector must be fitted before prediction.")

        X = self._prepare_features(df)

        # Fallback: return uniform probs if HMM fit failed
        if getattr(self, '_fallback', False):
            return np.full((len(X), 3), 1.0 / 3, dtype=np.float32)

        X_scaled = (X - self._mean) / self._std
        posteriors = self.model.predict_proba(X_scaled)

        # Reorder columns to [bull, bear, choppy]
        reordered = np.zeros((len(X), 3), dtype=np.float32)
        for hmm_state, regime_idx in self.state_map.items():
            reordered[:, regime_idx] = posteriors[:, hmm_state]

        return reordered

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict most likely regime per bar.

        Returns:
            (n_samples,) array of regime indices {0=bull, 1=bear, 2=choppy}.
        """
        probs = self.predict_proba(df)
        return probs.argmax(axis=1)

    def current_regime(self, df: pd.DataFrame) -> str:
        """Get the current regime name from the last bar.

        Returns:
            'bull', 'bear', or 'choppy'.
        """
        probs = self.predict_proba(df)
        regime_idx = probs[-1].argmax()
        return self.STATE_NAMES[regime_idx]

    def is_tradeable(self, df: pd.DataFrame) -> np.ndarray:
        """Check if regime permits trading.

        Returns False for bars where P(choppy) > threshold.

        Returns:
            (n_samples,) boolean array.
        """
        probs = self.predict_proba(df)
        return probs[:, self.CHOPPY] <= self.config.choppy_threshold
