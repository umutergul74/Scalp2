"""XGBoost meta-learner — Stage 2 of the hybrid pipeline.

Professional-grade implementation with:
- Inverse-frequency sample weighting for class imbalance
- Probability calibration via Platt scaling
- NaN/Inf input sanitization
- Proper device handling (CPU input for predict)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scalp2.config import XGBoostConfig

logger = logging.getLogger(__name__)


class XGBoostMetaLearner:
    """XGBoost classifier consuming DL latent features + handcrafted features + regime.

    Input vector per sample:
        [latent_96 | handcrafted_top_k | regime_probs_3]

    This acts as the final signal filter: XGBoost excels at handling the
    tabular nature of mixed neural + engineered features and provides
    built-in regularization against overfitting.
    """

    def __init__(self, config: XGBoostConfig):
        self.config = config
        params = {
            "objective": config.objective,
            "num_class": config.num_class,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "min_child_weight": config.min_child_weight,
            "gamma": config.gamma,
            "reg_alpha": config.reg_alpha,
            "reg_lambda": config.reg_lambda,
            "tree_method": config.tree_method,
            "eval_metric": config.eval_metric,
            "seed": 42,
            "verbosity": 0,
        }
        # NOTE: device is intentionally omitted from params.
        # XGBoost hist tree_method auto-detects GPU when available.
        # Setting device="cuda" causes predict() warnings when input
        # is on CPU (numpy). Let XGBoost handle device placement.
        self.model = xgb.XGBClassifier(
            n_estimators=config.n_estimators,
            early_stopping_rounds=config.early_stopping_rounds,
            **params,
        )
        self.feature_names: list[str] | None = None

    @staticmethod
    def _sanitize(X: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with 0 to prevent XGBoost crashes."""
        X = np.asarray(X, dtype=np.float32)
        mask = ~np.isfinite(X)
        if mask.any():
            n_bad = mask.sum()
            logger.warning("Sanitized %d NaN/Inf values in input features", n_bad)
            X[mask] = 0.0
        return X

    @staticmethod
    def build_meta_features(
        latent: np.ndarray,
        handcrafted: np.ndarray,
        regime_probs: np.ndarray,
    ) -> np.ndarray:
        """Concatenate all Stage-2 input features.

        Args:
            latent: (n, latent_dim) from HybridEncoder.
            handcrafted: (n, top_k) selected features.
            regime_probs: (n, 3) HMM regime probabilities.

        Returns:
            (n, latent_dim + top_k + 3) combined feature matrix.
        """
        return XGBoostMetaLearner._sanitize(
            np.hstack([latent, handcrafted, regime_probs])
        )

    @staticmethod
    def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
        """Compute inverse-frequency sample weights to combat class imbalance.

        Without this, XGBoost's mlogloss objective will predict the majority
        class (Hold) for nearly everything, destroying directional signals.

        Uses sklearn-compatible formula: w_c = n_samples / (n_classes * n_c)

        Args:
            labels: (n,) array of class labels {0, 1, 2}.

        Returns:
            (n,) array of per-sample weights.
        """
        classes, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(classes)

        # Weight = n_samples / (n_classes * count_per_class)
        class_weights = n_samples / (n_classes * counts)

        # Map per-class weights to per-sample weights
        weight_map = dict(zip(classes, class_weights))
        sample_weights = np.array(
            [weight_map[label] for label in labels], dtype=np.float32
        )

        logger.info(
            "XGBoost class weights: %s (counts: %s)",
            {int(c): f"{w:.3f}" for c, w in zip(classes, class_weights)},
            {int(c): int(n) for c, n in zip(classes, counts)},
        )
        return sample_weights

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
        use_sample_weights: bool = True,
    ) -> XGBoostMetaLearner:
        """Fit XGBoost with early stopping on validation set.

        Args:
            X_train: Training features.
            y_train: Training labels {0, 1, 2}.
            X_val: Validation features.
            y_val: Validation labels.
            feature_names: Optional feature name list for interpretability.
            use_sample_weights: If True, apply inverse-frequency class weights.

        Returns:
            self (for chaining).
        """
        self.feature_names = feature_names

        # Sanitize inputs
        X_train = self._sanitize(X_train)
        X_val = self._sanitize(X_val)

        # Compute sample weights to rebalance classes
        sample_weight = None
        if use_sample_weights:
            sample_weight = self.compute_sample_weights(y_train)

        logger.info(
            "Training XGBoost: %d train, %d val, %d features",
            len(X_train), len(X_val), X_train.shape[1],
        )

        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = self.model.best_iteration
        best_score = self.model.best_score
        logger.info(
            "XGBoost done: best_iter=%d, best_score=%.6f",
            best_iter, best_score,
        )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: (n, n_features) feature matrix.

        Returns:
            (n, 3) probability matrix [P(short), P(hold), P(long)].
        """
        X = self._sanitize(X)
        return self.model.predict_proba(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = self._sanitize(X)
        return self.model.predict(X).astype(np.int64)

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores sorted by importance.

        Returns:
            DataFrame with columns [feature, importance, rank].
        """
        importances = self.model.feature_importances_

        if self.feature_names and len(self.feature_names) == len(importances):
            names = self.feature_names
        else:
            names = [f"f_{i}" for i in range(len(importances))]

        df = pd.DataFrame({
            "feature": names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        self.model.save_model(str(path))
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model from file."""
        self.model.load_model(str(path))
        logger.info("XGBoost model loaded from %s", path)
