"""Stage-2 XGBoost meta-learner training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler

from scalp2.config import Config
from scalp2.models.meta_learner import XGBoostMetaLearner
from scalp2.regime.hmm import RegimeDetector
from scalp2.training.trainer import Stage1Trainer

logger = logging.getLogger(__name__)


class Stage2Trainer:
    """Orchestrates Stage-2 training: latent extraction + feature selection + XGBoost.

    For each walk-forward fold:
        1. Load best Stage-1 model
        2. Extract 128-dim latent vectors from train/val/test
        3. Select top-K handcrafted features by mutual information
        4. Get HMM regime probabilities
        5. Concatenate all → fit XGBoost → evaluate on test
    """

    def __init__(self, config: Config, checkpoint_dir: str | Path = "./checkpoints"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)

    def select_top_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        top_k: int = 40,
    ) -> tuple[np.ndarray, list[str]]:
        """Select top-K features by mutual information.

        Args:
            features: (n_samples, n_features) array.
            labels: (n_samples,) label array.
            feature_names: Column names.
            top_k: Number of features to select.

        Returns:
            selected_indices: Indices of selected features.
            selected_names: Names of selected features.
        """
        mi_scores = mutual_info_classif(
            features, labels, random_state=42, n_neighbors=5
        )
        top_indices = np.argsort(mi_scores)[-top_k:]

        selected_names = [feature_names[i] for i in top_indices]
        logger.info(
            "Selected top %d features by MI. Top 5: %s",
            top_k,
            selected_names[-5:],
        )
        return top_indices, selected_names

    def train_one_fold(
        self,
        stage1_trainer: Stage1Trainer,
        regime_detector: RegimeDetector,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        train_df_regime,
        val_df_regime,
        test_df_regime,
        feature_names: list[str],
        fold_idx: int,
    ) -> dict:
        """Run Stage-2 training for a single fold.

        Args:
            stage1_trainer: Trained Stage1Trainer with best model loaded.
            regime_detector: Fitted RegimeDetector.
            train/val/test_features: Scaled feature arrays.
            train/val/test_labels: Label arrays.
            train/val/test_df_regime: DataFrames with regime feature columns.
            feature_names: All feature column names.
            fold_idx: Current fold index.

        Returns:
            Dict with test predictions, probabilities, and metrics.
        """
        seq_len = self.config.model.seq_len
        top_k = self.config.model.handcrafted_top_k

        # 1. Extract latent vectors
        logger.info("Fold %d: extracting latent vectors...", fold_idx)
        latent_train = stage1_trainer.extract_latents(train_features, seq_len)
        latent_val = stage1_trainer.extract_latents(val_features, seq_len)
        latent_test = stage1_trainer.extract_latents(test_features, seq_len)

        # Align labels to latent (latent drops first seq_len samples)
        train_labels_aligned = train_labels[seq_len:]
        val_labels_aligned = val_labels[seq_len:]
        test_labels_aligned = test_labels[seq_len:]

        # 2. Select top-K handcrafted features
        top_indices, selected_names = self.select_top_features(
            train_features[seq_len:], train_labels_aligned, feature_names, top_k
        )

        hc_train = train_features[seq_len:][:, top_indices]
        hc_val = val_features[seq_len:][:, top_indices]
        hc_test = test_features[seq_len:][:, top_indices]

        # 3. Get regime probabilities
        regime_train = regime_detector.predict_proba(train_df_regime.iloc[seq_len:])
        regime_val = regime_detector.predict_proba(val_df_regime.iloc[seq_len:])
        regime_test = regime_detector.predict_proba(test_df_regime.iloc[seq_len:])

        # Ensure consistent lengths (truncate to shortest)
        min_train = min(len(latent_train), len(hc_train), len(regime_train))
        min_val = min(len(latent_val), len(hc_val), len(regime_val))
        min_test = min(len(latent_test), len(hc_test), len(regime_test))

        # 4. Build meta-feature matrices
        meta_train = XGBoostMetaLearner.build_meta_features(
            latent_train[:min_train], hc_train[:min_train], regime_train[:min_train]
        )
        meta_val = XGBoostMetaLearner.build_meta_features(
            latent_val[:min_val], hc_val[:min_val], regime_val[:min_val]
        )
        meta_test = XGBoostMetaLearner.build_meta_features(
            latent_test[:min_test], hc_test[:min_test], regime_test[:min_test]
        )

        # Build feature name list
        latent_names = [f"latent_{i}" for i in range(latent_train.shape[1])]
        regime_names = ["regime_bull", "regime_bear", "regime_choppy"]
        all_meta_names = latent_names + selected_names + regime_names

        # 5. Train XGBoost
        meta_learner = XGBoostMetaLearner(self.config.model.xgboost)
        meta_learner.fit(
            meta_train,
            train_labels_aligned[:min_train],
            meta_val,
            val_labels_aligned[:min_val],
            feature_names=all_meta_names,
        )

        # 6. Test predictions
        test_probs = meta_learner.predict_proba(meta_test)
        test_preds = meta_learner.predict(meta_test)

        # Save model
        model_path = self.checkpoint_dir / f"xgb_fold_{fold_idx:03d}.json"
        meta_learner.save(model_path)

        # Feature importance
        importance_df = meta_learner.feature_importance()
        logger.info(
            "Fold %d top-5 XGBoost features:\n%s",
            fold_idx,
            importance_df.head(5).to_string(index=False),
        )

        return {
            "fold_idx": fold_idx,
            "test_predictions": test_preds,
            "test_probabilities": test_probs,
            "test_labels": test_labels_aligned[:min_test],
            "feature_importance": importance_df,
            "top_feature_indices": top_indices,
            "top_feature_names": selected_names,
        }
