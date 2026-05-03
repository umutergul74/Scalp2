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
    SS: np.ndarray  # (n_states, n_features) weighted feature sum-of-squares (diag)
    trans_counts: np.ndarray  # (n_states, n_states) transition counts
    feat_mean: np.ndarray  # (n_features,) running feature mean
    feat_var: np.ndarray  # (n_features,) running feature variance
    feat_count: float  # effective sample count for EMA
    total_bars_seen: int = 0
    bars_since_update: int = 0
    SX: np.ndarray | None = None  # (n_states, n_features, n_features) outer product sums (full cov)

logger = logging.getLogger(__name__)


def _logsumexp_1d(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp for a 1-D array."""
    a_max = a.max()
    return a_max + np.log(np.sum(np.exp(a - a_max)))


class RegimeDetector:
    """3-state Gaussian HMM for market regime detection.

    Two versions:
        v1: 3 features [log_return, gk_vol_14, volume_zscore], diag covariance.
            States mapped by sorting on mean return.
        v2: 5 features [log_return, gk_vol_14, atr_pct, adx, rsi_14], full covariance.
            Choppy = lowest ADX mean, then Bull/Bear by return.

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
        self.version = getattr(config, "version", "v1")
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

        # Map states to regime labels
        if self.version == "v2":
            self.state_map = self._map_states_v2()
        else:
            self.state_map = self._map_states_v1()

        logger.info(
            "HMM %s state mapping: %s",
            self.version,
            {
                hmm_state: self.STATE_NAMES[regime_idx]
                for hmm_state, regime_idx in self.state_map.items()
            },
        )
        means = self.model.means_[:, 0]
        inv_map = {v: k for k, v in self.state_map.items()}
        logger.info(
            "State means (return): Bull=%.6f, Bear=%.6f, Choppy=%.6f",
            means[inv_map[self.BULL]],
            means[inv_map[self.BEAR]],
            means[inv_map[self.CHOPPY]],
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

    def _map_states_v1(self) -> dict[int, int]:
        """Map states by sorting on mean return (original method)."""
        means = self.model.means_[:, 0]
        sorted_idx = np.argsort(means)
        return {
            int(sorted_idx[0]): self.BEAR,
            int(sorted_idx[-1]): self.BULL,
            int(sorted_idx[1]): self.CHOPPY,
        }

    def _map_states_v2(self) -> dict[int, int]:
        """Map states using ADX for choppy, return for bull/bear.

        v2 uses ADX (trend strength) to identify the choppy state directly,
        rather than assuming choppy = middle return. This is more robust
        because choppy markets can have any return direction.
        """
        means = self.model.means_

        # Find ADX feature index in configured feature list
        try:
            adx_idx = self.feature_names.index("adx")
        except ValueError:
            logger.warning("ADX not in regime features, falling back to v1 mapping")
            return self._map_states_v1()

        # Choppy = lowest ADX mean (weakest trend)
        adx_means = means[:, adx_idx]
        choppy_state = int(np.argmin(adx_means))

        # Remaining 2 states: higher return mean = Bull, lower = Bear
        remaining = [s for s in range(self.config.n_states) if s != choppy_state]
        ret_idx = 0  # log_return is always first feature
        if means[remaining[0], ret_idx] > means[remaining[1], ret_idx]:
            bull_state, bear_state = remaining[0], remaining[1]
        else:
            bull_state, bear_state = remaining[1], remaining[0]

        logger.info(
            "v2 mapping: ADX means=[%.2f, %.2f, %.2f], choppy=state%d (ADX=%.2f)",
            *adx_means, choppy_state, adx_means[choppy_state],
        )

        return {
            int(bull_state): self.BULL,
            int(bear_state): self.BEAR,
            int(choppy_state): self.CHOPPY,
        }

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

        # Snapshot trained parameters — used by _enforce_state_floor to
        # reconstruct starving states from their *original* distribution
        # instead of copying the dominant state's profile.
        self._trained_means_snapshot = self.model.means_.copy()
        self._trained_covars_snapshot = getattr(
            self.model, "_covars_", self.model.covars_
        ).copy()

        # Seed sufficient stats from trained model parameters
        internal_covars = getattr(self.model, "_covars_", self.model.covars_)
        w = init_weight / n_states

        self._online_stats = _OnlineStats(
            N=np.full(n_states, w),
            S=self.model.means_.copy() * w,
            SS=(internal_covars.copy() + self.model.means_ ** 2) * w
            if self.config.covariance_type == "diag"
            else np.zeros((n_states, len(self.feature_names))),  # unused for full
            trans_counts=self.model.transmat_.copy() * init_weight,
            feat_mean=self._mean.copy(),
            feat_var=(self._std**2).copy(),
            feat_count=init_weight,
        )

        # Full covariance: track outer product sums E[x x^T]
        if self.config.covariance_type == "full":
            means = self.model.means_
            self._online_stats.SX = (
                internal_covars.copy()
                + np.einsum("ij,ik->ijk", means, means)
            ) * w
        else:
            # Diagonal: SS is sufficient, keep SS init from above
            self._online_stats.SS = (
                internal_covars.copy() + self.model.means_ ** 2
            ) * w

    # ── Health monitoring ──────────────────────────────────────────────

    # Minimum fraction of total N any single state must hold
    _MIN_STATE_WEIGHT_FRAC = 0.08  # 8% — below this we redistribute
    # Absolute floor for SS values (below this = numerical underflow)
    _SS_UNDERFLOW_THRESHOLD = 1e-100
    # Maximum ratio between largest and smallest N before alarm
    _MAX_N_RATIO = 15.0
    # Minimum transition probability floor (prevents transition death)
    _MIN_TRANS_FRAC = 0.005  # 0.5% — any transition must have at least this prob
    # Minimum off-diagonal transition count ratio before warning
    _MIN_TRANS_COUNT_RATIO = 1e-4
    # Minimum gamma (responsibility) per state per bar — breaks the
    # positive feedback loop where gamma→1 for one state → only that
    # state accumulates → re-estimation reinforces → stuck forever.
    _GAMMA_FLOOR = 0.02

    def health_check(self) -> dict:
        """Check HMM online stats for collapse symptoms.

        Returns:
            dict with keys:
                'healthy': bool — True if no issues detected.
                'issues': list[str] — Human-readable issue descriptions.
                'collapsed': bool — True if irrecoverable collapse.
        """
        result = {'healthy': True, 'issues': [], 'collapsed': False}

        if not hasattr(self, '_online_stats'):
            return result

        stats = self._online_stats
        n_states = self.config.n_states
        total_n = stats.N.sum()

        if total_n < 1e-10:
            result['healthy'] = False
            result['collapsed'] = True
            result['issues'].append(
                f'Total N is near zero ({total_n:.2e}) — all states dead'
            )
            return result

        # Check N imbalance
        n_min, n_max = stats.N.min(), stats.N.max()
        if n_min < 1e-10:
            ratio = float('inf')
        else:
            ratio = n_max / n_min

        if ratio > self._MAX_N_RATIO:
            result['healthy'] = False
            result['issues'].append(
                f'N imbalance: max/min ratio = {ratio:.1f} '
                f'(N = [{stats.N[0]:.2f}, {stats.N[1]:.2f}, {stats.N[2]:.2f}])'
            )

        # Check SS / SX underflow
        if stats.SX is not None:
            # Full covariance: check SX diagonals
            for s in range(n_states):
                diag = np.diag(stats.SX[s]) if stats.SX[s].ndim == 2 else stats.SX[s]
                if np.any(np.abs(diag) < self._SS_UNDERFLOW_THRESHOLD):
                    result['healthy'] = False
                    result['collapsed'] = True
                    result['issues'].append(
                        f'State {s} SX diagonal underflowed '
                        f'(min={np.abs(diag).min():.2e})'
                    )
        else:
            for s in range(n_states):
                if np.any(np.abs(stats.SS[s]) < self._SS_UNDERFLOW_THRESHOLD):
                    result['healthy'] = False
                    result['collapsed'] = True
                    result['issues'].append(
                        f'State {s} SS underflowed '
                        f'(min={np.abs(stats.SS[s]).min():.2e})'
                    )

        # Check state fractions
        fracs = stats.N / total_n
        for s in range(n_states):
            if fracs[s] < self._MIN_STATE_WEIGHT_FRAC:
                result['healthy'] = False
                result['issues'].append(
                    f'State {s} starving: {fracs[s]*100:.2f}% of total weight'
                )

        # Check transition matrix degeneration
        row_sums = stats.trans_counts.sum(axis=1)
        for i in range(n_states):
            if row_sums[i] < 1e-10:
                result['healthy'] = False
                result['collapsed'] = True
                result['issues'].append(
                    f'Transition row {i} has zero total count'
                )
                continue
            for j in range(n_states):
                if i == j:
                    continue
                trans_prob = stats.trans_counts[i, j] / row_sums[i]
                if trans_prob < self._MIN_TRANS_COUNT_RATIO:
                    result['healthy'] = False
                    result['issues'].append(
                        f'Transition {i}->{j} degenerated: '
                        f'prob={trans_prob:.2e}, '
                        f'count={stats.trans_counts[i, j]:.2e}'
                    )

        return result

    def _enforce_state_floor(self) -> None:
        """Redistribute weight to prevent any state from dying.

        If a state's N drops below MIN_FRAC of total, steal weight from
        the dominant state and *reconstruct* the starving state's
        sufficient statistics from the trained model snapshot. This is
        critical: the old approach copied S/SX from the dominant state,
        which made all states converge to the same distribution and
        destroyed the HMM's discriminative ability.

        Also floors transition counts to prevent any transition path
        from dying (which causes the forward pass to get stuck).
        """
        stats = self._online_stats
        n_states = self.config.n_states
        total_n = stats.N.sum()
        if total_n < 1e-10:
            return

        min_weight = total_n * self._MIN_STATE_WEIGHT_FRAC
        has_snapshot = hasattr(self, '_trained_means_snapshot')

        for s in range(n_states):
            # Recalculate dominant each iteration — a previous
            # redistribution may have changed who the dominant is.
            dominant = int(np.argmax(stats.N))
            if s == dominant:
                continue
            if stats.N[s] < min_weight:
                deficit = min_weight - stats.N[s]
                # Transfer N from dominant, cap at 30% of dominant weight
                transfer_frac = deficit / (stats.N[dominant] + 1e-10)
                transfer_frac = min(transfer_frac, 0.30)
                actual_transfer = stats.N[dominant] * transfer_frac

                stats.N[s] += actual_transfer
                stats.N[dominant] -= actual_transfer

                # Reconstruct starving state's stats from TRAINED params
                # instead of copying dominant (prevents convergence)
                if has_snapshot:
                    mu = self._trained_means_snapshot[s]
                    stats.S[s] = mu * stats.N[s]
                    if stats.SX is not None:
                        stats.SX[s] = (
                            self._trained_covars_snapshot[s]
                            + np.outer(mu, mu)
                        ) * stats.N[s]
                    else:
                        stats.SS[s] = (
                            self._trained_covars_snapshot[s] + mu ** 2
                        ) * stats.N[s]
                else:
                    # Fallback: old transfer method
                    stats.S[s] += stats.S[dominant] * transfer_frac
                    stats.S[dominant] *= (1 - transfer_frac)
                    if stats.SX is not None:
                        stats.SX[s] += stats.SX[dominant] * transfer_frac
                        stats.SX[dominant] *= (1 - transfer_frac)
                    else:
                        stats.SS[s] += stats.SS[dominant] * transfer_frac
                        stats.SS[dominant] *= (1 - transfer_frac)

                logger.warning(
                    'HMM state floor: redistributed %.2f weight from '
                    'state %d -> state %d (N was %.4f, now %.4f), '
                    'reconstructed=%s',
                    actual_transfer, dominant, s,
                    stats.N[s] - actual_transfer, stats.N[s],
                    has_snapshot,
                )

        # -- Transition count floor ------------------------------------
        # Prevent any transition path from dying. If trans_counts[i][j]
        # falls below MIN_TRANS_FRAC of the row sum, redistribute from
        # the self-transition count. This breaks the positive feedback
        # loop where dead transitions -> stuck forward pass -> no gamma
        # for that transition -> stays dead forever.
        for i in range(n_states):
            row_sum = stats.trans_counts[i].sum()
            if row_sum < 1e-10:
                # Row is dead -- reinitialize uniformly
                stats.trans_counts[i] = total_n / (n_states * n_states)
                logger.warning(
                    'HMM trans floor: row %d was dead, reinitialized', i
                )
                continue
            floor = row_sum * self._MIN_TRANS_FRAC
            for j in range(n_states):
                if i == j:
                    continue
                if stats.trans_counts[i, j] < floor:
                    deficit = floor - stats.trans_counts[i, j]
                    stats.trans_counts[i, j] = floor
                    stats.trans_counts[i, i] -= deficit  # take from self-transition
                    logger.debug(
                        'HMM trans floor: boosted trans[%d->%d] to %.4f '
                        '(took %.4f from self-transition)',
                        i, j, floor, deficit,
                    )

    def reset_online_stats(self) -> None:
        """Reset online stats to trained model parameters.

        Call this when health_check reports a collapse.
        """
        logger.warning('HMM online stats RESET to trained parameters')
        self._init_online_stats()

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

        # Ensure trained parameter snapshots exist — pickle-loaded models
        # and set_online_stats_dict() restores may not have them.
        if not hasattr(self, '_trained_means_snapshot'):
            self._trained_means_snapshot = self.model.means_.copy()
            self._trained_covars_snapshot = getattr(
                self.model, "_covars_", self.model.covars_
            ).copy()

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

        # 4b. Gamma floor: ensure every state gets minimum responsibility
        # on every bar. Without this, prolonged single-regime periods
        # cause gamma≈[1,0,0] → only one state accumulates stats →
        # re-estimation reinforces that state → positive feedback → collapse.
        gamma = np.maximum(gamma, self._GAMMA_FLOOR)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # 5. Decay existing stats then accumulate new
        batch_decay = decay**n_new
        stats.N *= batch_decay
        stats.S *= batch_decay
        stats.trans_counts *= batch_decay
        if stats.SX is not None:
            stats.SX *= batch_decay
            # Note: SS is NOT decayed in full-cov mode — SX is used for
            # re-estimation, SS is vestigial. Decaying it caused underflow
            # to 1e-220 which, while harmless, was confusing.
        else:
            stats.SS *= batch_decay

        for t in range(n_new):
            g = gamma[t]
            x = X_scaled[t]
            stats.N += g
            stats.S += g[:, None] * x[None, :]
            if stats.SX is not None:
                stats.SX += g[:, None, None] * np.outer(x, x)[None, :, :]
            else:
                stats.SS += g[:, None] * (x[None, :] ** 2)

            if t > 0:
                stats.trans_counts += np.outer(gamma[t - 1], gamma[t])

        stats.total_bars_seen += n_new
        stats.bars_since_update += n_new

        # 5b. Enforce minimum state weight (prevent collapse)
        self._enforce_state_floor()

        # Progress log every 24 bars (~6 hours)
        if stats.total_bars_seen % 24 == 0 or stats.total_bars_seen <= 2:
            logger.info(
                "HMM online: %d/%d bars collected (next update at %d), "
                "N=[%.2f, %.2f, %.2f]",
                stats.total_bars_seen,
                self.config.online_min_samples,
                self.config.online_update_interval,
                stats.N[0], stats.N[1], stats.N[2],
            )

        # 6. Re-estimate if enough data accumulated
        updated = False
        if (
            stats.total_bars_seen >= self.config.online_min_samples
            and stats.bars_since_update >= self.config.online_update_interval
        ):
            # Health check before re-estimating
            health = self.health_check()
            if health['collapsed']:
                logger.critical(
                    'HMM COLLAPSED before re-estimation! Issues: %s. '
                    'Auto-resetting to trained parameters.',
                    health['issues'],
                )
                self.reset_online_stats()
                # Re-bind: reset_online_stats() created a new object
                self._online_stats.bars_since_update = 0
                return False

            self._reestimate_from_stats()
            stats.bars_since_update = 0
            updated = True

        return updated

    def _reestimate_from_stats(self) -> None:
        """Re-estimate HMM parameters from accumulated sufficient statistics."""
        stats = self._online_stats
        n_states = self.config.n_states
        eps = 1e-3
        blend = 0.3

        # Emission means
        new_means = stats.S / (stats.N[:, None] + eps)

        # Emission covariances
        if stats.SX is not None:
            # Full covariance: Cov = E[xx'] - E[x]E[x]'
            new_covars = (
                stats.SX / (stats.N[:, None, None] + eps)
                - np.einsum("ij,ik->ijk", new_means, new_means)
            )
            # Regularize: ensure positive semi-definite
            for s in range(n_states):
                # Symmetrize (numerical safety)
                new_covars[s] = 0.5 * (new_covars[s] + new_covars[s].T)
                # Floor diagonal to min_covar
                diag = np.diag(new_covars[s]).copy()
                diag = np.maximum(diag, self.model.min_covar)
                np.fill_diagonal(new_covars[s], diag)
        else:
            # Diagonal covariance
            new_covars = stats.SS / (stats.N[:, None] + eps) - new_means**2
            new_covars = np.maximum(new_covars, self.model.min_covar)

        # Transition matrix — Dirichlet smoothing to prevent zero transitions
        alpha = 0.01  # pseudo-count per cell
        smoothed_counts = stats.trans_counts + alpha
        row_sums = smoothed_counts.sum(axis=1, keepdims=True) + eps
        new_transmat = smoothed_counts / row_sums

        # Smooth blend: 30% new + 70% old
        self.model.means_ = (1 - blend) * self.model.means_ + blend * new_means

        # Use internal _covars_ to bypass hmmlearn property setter validation
        old_covars = getattr(self.model, "_covars_", self.model.covars_)
        blended_covars = (1 - blend) * old_covars + blend * new_covars
        if stats.SX is not None:
            # Full: regularize blended result
            for s in range(n_states):
                blended_covars[s] = 0.5 * (blended_covars[s] + blended_covars[s].T)
                diag = np.diag(blended_covars[s]).copy()
                diag = np.maximum(diag, self.model.min_covar)
                np.fill_diagonal(blended_covars[s], diag)
        else:
            blended_covars = np.maximum(blended_covars, self.model.min_covar)
        if hasattr(self.model, "_covars_"):
            self.model._covars_ = blended_covars
        else:
            self.model.covars_ = blended_covars

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
        d = {
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
        if s.SX is not None:
            d["SX"] = s.SX.tolist()
        return d

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
            SX=np.array(data["SX"]) if "SX" in data else None,
        )

        # Ensure trained parameter snapshots exist so _enforce_state_floor
        # can reconstruct starving states from trained params (not dominant).
        # Without this, bot restart → set_online_stats_dict → no snapshots
        # → fallback to old convergent copy-from-dominant method.
        if not hasattr(self, '_trained_means_snapshot') and self._fitted:
            self._trained_means_snapshot = self.model.means_.copy()
            self._trained_covars_snapshot = getattr(
                self.model, "_covars_", self.model.covars_
            ).copy()
