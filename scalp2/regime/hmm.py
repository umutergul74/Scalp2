"""Hidden Markov Model regime detection — Bull, Bear, Choppy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from scalp2.config import RegimeConfig


@dataclass
class _OnlineStats:
    """Exponential-decay sufficient statistics for online HMM update."""

    N: np.ndarray  # (n_states,) weighted counts per state
    S: np.ndarray  # (n_states, n_features) weighted feature sums
    SS: np.ndarray  # (n_states, n_features) weighted feature sum-of-squares
    trans_counts: np.ndarray  # (n_states, n_states) transition counts
    feat_mean: np.ndarray  # (n_features,) running feature mean
    feat_var: np.ndarray  # (n_features,) running feature variance
    feat_count: float  # effective sample count for EMA
    total_bars_seen: int = 0
    bars_since_update: int = 0

logger = logging.getLogger(__name__)


def _logsumexp_1d(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp for a 1-D array."""
    a_max = a.max()
    return a_max + np.log(np.sum(np.exp(a - a_max)))


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

        if self.config.online_update_enabled and not self._fallback:
            self._init_online_stats()

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
        return self._reorder_to_regime(posteriors)

    def _forward_pass(self, X_scaled: np.ndarray) -> np.ndarray:
        """Run forward-only HMM pass in log space.

        Args:
            X_scaled: (n_samples, n_features) standardized features.

        Returns:
            log_alpha: (n_samples, n_states) log forward variables.
        """
        log_emissionprob = self.model._compute_log_likelihood(X_scaled)
        n_samples, n_states = log_emissionprob.shape
        log_startprob = np.log(self.model.startprob_ + 1e-300)
        log_transmat = np.log(self.model.transmat_ + 1e-300)

        log_alpha = np.zeros((n_samples, n_states))
        log_alpha[0] = log_startprob + log_emissionprob[0]

        for t in range(1, n_samples):
            for j in range(n_states):
                log_alpha[t, j] = (
                    _logsumexp_1d(log_alpha[t - 1] + log_transmat[:, j])
                    + log_emissionprob[t, j]
                )
        return log_alpha

    def _forward_only_gamma(self, X_scaled: np.ndarray) -> np.ndarray:
        """Forward-only state responsibilities: P(state_t | x_1:t).

        Returns:
            (n_samples, n_states) normalized forward probabilities.
        """
        log_alpha = self._forward_pass(X_scaled)
        log_norm = np.array([_logsumexp_1d(row) for row in log_alpha])
        log_proba = log_alpha - log_norm[:, None]
        return np.exp(log_proba)

    def _reorder_to_regime(self, posteriors: np.ndarray) -> np.ndarray:
        """Reorder HMM state columns to [bull, bear, choppy]."""
        reordered = np.zeros((len(posteriors), 3), dtype=np.float32)
        for hmm_state, regime_idx in self.state_map.items():
            reordered[:, regime_idx] = posteriors[:, hmm_state]
        return reordered

    def predict_proba_online(self, df: pd.DataFrame) -> np.ndarray:
        """Forward-only regime probabilities (no look-ahead bias).

        Uses only the forward pass of the HMM — P(state_t | x_1:t).
        Unlike predict_proba() which uses forward-backward and leaks
        future information via the backward pass.

        Args:
            df: DataFrame with regime feature columns.

        Returns:
            (n_samples, 3) array with columns [bull_prob, bear_prob, choppy_prob].
        """
        if not self._fitted:
            raise RuntimeError("RegimeDetector must be fitted before prediction.")

        X = self._prepare_features(df)

        if getattr(self, '_fallback', False):
            return np.full((len(X), 3), 1.0 / 3, dtype=np.float32)

        X_scaled = (X - self._mean) / self._std
        gamma = self._forward_only_gamma(X_scaled)
        return self._reorder_to_regime(gamma)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict most likely regime per bar.

        Returns:
            (n_samples,) array of regime indices {0=bull, 1=bear, 2=choppy}.
        """
        probs = self.predict_proba(df)
        return probs.argmax(axis=1)

    def current_regime(self, df: pd.DataFrame) -> str:
        """Get the current regime name from the last bar (forward-backward).

        WARNING: Uses forward-backward — only use for training data.
        For live/val/test, use current_regime_online().
        """
        probs = self.predict_proba(df)
        regime_idx = probs[-1].argmax()
        return self.STATE_NAMES[regime_idx]

    def current_regime_online(self, df: pd.DataFrame) -> str:
        """Get the current regime name from the last bar (forward-only).

        Safe for live inference and val/test — no look-ahead bias.
        """
        probs = self.predict_proba_online(df)
        regime_idx = probs[-1].argmax()
        return self.STATE_NAMES[regime_idx]

    def is_tradeable(self, df: pd.DataFrame) -> np.ndarray:
        """Check if regime permits trading (forward-backward).

        WARNING: Uses forward-backward — only use for training data.
        For live/val/test, use is_tradeable_online().
        """
        probs = self.predict_proba(df)
        return probs[:, self.CHOPPY] <= self.config.choppy_threshold

    def is_tradeable_online(self, df: pd.DataFrame) -> np.ndarray:
        """Check if regime permits trading (forward-only).

        Safe for live inference and val/test — no look-ahead bias.
        """
        probs = self.predict_proba_online(df)
        return probs[:, self.CHOPPY] <= self.config.choppy_threshold

    # ── Online HMM Update ────────────────────────────────────────────────

    def _init_online_stats(self) -> None:
        """Initialize online sufficient statistics from the fitted HMM."""
        n_states = self.config.n_states
        init_weight = float(self.config.online_min_samples)

        # Seed sufficient stats from trained model parameters
        self._online_stats = _OnlineStats(
            N=np.full(n_states, init_weight / n_states),
            S=self.model.means_.copy() * (init_weight / n_states),
            SS=(self.model.covars_.copy() + self.model.means_ ** 2)
            * (init_weight / n_states),
            trans_counts=self.model.transmat_.copy() * init_weight,
            feat_mean=self._mean.copy(),
            feat_var=(self._std**2).copy(),
            feat_count=init_weight,
        )

    def update_online(self, df: pd.DataFrame) -> bool:
        """Update HMM parameters with new data using exponential decay.

        Args:
            df: DataFrame with new bars (1 or more rows).

        Returns:
            True if parameters were re-estimated this call.
        """
        if not self._fitted:
            raise RuntimeError("Must fit() before update_online().")
        if not self.config.online_update_enabled:
            raise RuntimeError("Online updates not enabled in config.")
        if getattr(self, "_fallback", False):
            return False

        # Lazy init: handles old pickled models that predate online update
        if not hasattr(self, "_online_stats"):
            self._init_online_stats()

        stats = self._online_stats
        decay = self.config.online_decay_factor
        n_new = len(df)

        # 1. Prepare raw features
        X_raw = self._prepare_features(df)

        # 2. Update running feature standardization (EMA)
        for i in range(n_new):
            stats.feat_count = stats.feat_count * decay + 1.0
            alpha = 1.0 / stats.feat_count
            delta = X_raw[i] - stats.feat_mean
            stats.feat_mean = stats.feat_mean + alpha * delta
            stats.feat_var = stats.feat_var + alpha * (
                delta * (X_raw[i] - stats.feat_mean) - stats.feat_var
            )

        # 3. Standardize with current _mean/_std
        X_scaled = (X_raw - self._mean) / self._std

        # 4. Forward-only pass for state responsibilities
        gamma = self._forward_only_gamma(X_scaled)

        # 5. Decay existing stats then accumulate new
        batch_decay = decay**n_new
        stats.N *= batch_decay
        stats.S *= batch_decay
        stats.SS *= batch_decay
        stats.trans_counts *= batch_decay

        for t in range(n_new):
            g = gamma[t]
            x = X_scaled[t]
            stats.N += g
            stats.S += g[:, None] * x[None, :]
            stats.SS += g[:, None] * (x[None, :] ** 2)

            if t > 0:
                stats.trans_counts += np.outer(gamma[t - 1], gamma[t])

        stats.total_bars_seen += n_new
        stats.bars_since_update += n_new

        # 6. Re-estimate if enough data accumulated
        updated = False
        if (
            stats.total_bars_seen >= self.config.online_min_samples
            and stats.bars_since_update >= self.config.online_update_interval
        ):
            self._reestimate_from_stats()
            stats.bars_since_update = 0
            updated = True

        return updated

    def _reestimate_from_stats(self) -> None:
        """Re-estimate HMM parameters from accumulated sufficient statistics."""
        stats = self._online_stats
        eps = 1e-3
        blend = 0.3

        # Emission means
        new_means = stats.S / (stats.N[:, None] + eps)

        # Emission covariances (diagonal)
        new_covars = stats.SS / (stats.N[:, None] + eps) - new_means**2
        new_covars = np.maximum(new_covars, self.model.min_covar)

        # Transition matrix
        row_sums = stats.trans_counts.sum(axis=1, keepdims=True) + eps
        new_transmat = stats.trans_counts / row_sums

        # Smooth blend: 30% new + 70% old
        self.model.means_ = (1 - blend) * self.model.means_ + blend * new_means
        self.model.covars_ = (1 - blend) * self.model.covars_ + blend * new_covars
        self.model.transmat_ = (
            (1 - blend) * self.model.transmat_ + blend * new_transmat
        )

        # Re-normalize transition matrix rows
        self.model.transmat_ /= self.model.transmat_.sum(axis=1, keepdims=True)

        # Update feature standardization
        new_std = np.sqrt(np.maximum(stats.feat_var, 1e-8))
        self._mean = (1 - blend) * self._mean + blend * stats.feat_mean
        self._std = (1 - blend) * self._std + blend * new_std

        # state_map is NEVER changed — frozen at fit() time

        logger.info(
            "HMM online update: %d bars total, means=[%.6f, %.6f, %.6f]",
            stats.total_bars_seen,
            self.model.means_[0, 0],
            self.model.means_[1, 0],
            self.model.means_[2, 0],
        )

    def get_online_stats_dict(self) -> dict | None:
        """Export online stats as a JSON-serializable dict."""
        if not hasattr(self, "_online_stats"):
            return None
        s = self._online_stats
        return {
            "N": s.N.tolist(),
            "S": s.S.tolist(),
            "SS": s.SS.tolist(),
            "trans_counts": s.trans_counts.tolist(),
            "feat_mean": s.feat_mean.tolist(),
            "feat_var": s.feat_var.tolist(),
            "feat_count": s.feat_count,
            "total_bars_seen": s.total_bars_seen,
            "bars_since_update": s.bars_since_update,
        }

    def set_online_stats_dict(self, data: dict) -> None:
        """Restore online stats from a saved dict."""
        self._online_stats = _OnlineStats(
            N=np.array(data["N"]),
            S=np.array(data["S"]),
            SS=np.array(data["SS"]),
            trans_counts=np.array(data["trans_counts"]),
            feat_mean=np.array(data["feat_mean"]),
            feat_var=np.array(data["feat_var"]),
            feat_count=data["feat_count"],
            total_bars_seen=data["total_bars_seen"],
            bars_since_update=data["bars_since_update"],
        )
