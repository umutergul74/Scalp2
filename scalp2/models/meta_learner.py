"""XGBoost meta-learner — Stage 2 of the hybrid pipeline."""

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
        [latent_128 | handcrafted_top_k | regime_probs_3]

    This acts as the ultimate filter: XGBoost excels at handling the
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
            "verbosity": 1,
        }
        self.model = xgb.XGBClassifier(
            n_estimators=config.n_estimators,
            early_stopping_rounds=config.early_stopping_rounds,
            **params,
        )
        self.feature_names: list[str] | None = None

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
        return np.hstack([latent, handcrafted, regime_probs]).astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> XGBoostMetaLearner:
        """Fit XGBoost with early stopping on validation set.

        Args:
            X_train: Training features.
            y_train: Training labels {0, 1, 2}.
            X_val: Validation features.
            y_val: Validation labels.
            feature_names: Optional feature name list for interpretability.

        Returns:
            self (for chaining).
        """
        self.feature_names = feature_names

        logger.info(
            "Training XGBoost meta-learner: %d train, %d val, %d features",
            len(X_train), len(X_val), X_train.shape[1],
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = self.model.best_iteration
        best_score = self.model.best_score
        logger.info(
            "XGBoost training complete: best_iteration=%d, best_score=%.6f",
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
        return self.model.predict_proba(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X).astype(np.int64)

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.

        Returns:
            DataFrame with feature names and importance scores.
        """
        importances = self.model.feature_importances_

        if self.feature_names and len(self.feature_names) == len(importances):
            names = self.feature_names
        else:
            names = [f"f_{i}" for i in range(len(importances))]

        df = pd.DataFrame({
            "feature": names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        return df

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        self.model.save_model(str(path))
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model from file."""
        self.model.load_model(str(path))
        logger.info("XGBoost model loaded from %s", path)
