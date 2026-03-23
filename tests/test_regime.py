"""Tests for HMM regime detection — core + online update."""

import numpy as np
import pandas as pd
import pytest

from scalp2.config import RegimeConfig
from scalp2.regime.hmm import RegimeDetector, _OnlineStats


def _make_regime_df(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data with 3 distinct regimes."""
    rng = np.random.RandomState(seed)
    third = n // 3

    # Bull: positive returns, low vol
    bull_ret = rng.normal(0.002, 0.005, third)
    bull_vol = rng.uniform(0.005, 0.015, third)
    bull_vz = rng.normal(0.5, 0.3, third)

    # Bear: negative returns, high vol
    bear_ret = rng.normal(-0.002, 0.008, third)
    bear_vol = rng.uniform(0.015, 0.035, third)
    bear_vz = rng.normal(-0.5, 0.3, third)

    # Choppy: near-zero returns, medium vol
    choppy_ret = rng.normal(0.0, 0.003, n - 2 * third)
    choppy_vol = rng.uniform(0.008, 0.020, n - 2 * third)
    choppy_vz = rng.normal(0.0, 0.5, n - 2 * third)

    return pd.DataFrame({
        "log_return": np.concatenate([bull_ret, bear_ret, choppy_ret]),
        "gk_vol_14": np.concatenate([bull_vol, bear_vol, choppy_vol]),
        "volume_zscore": np.concatenate([bull_vz, bear_vz, choppy_vz]),
    })


@pytest.fixture
def config_offline():
    """Config with online updates disabled."""
    return RegimeConfig(online_update_enabled=False)


@pytest.fixture
def config_online():
    """Config with online updates enabled, low thresholds for testing."""
    return RegimeConfig(
        online_update_enabled=True,
        online_decay_factor=0.99,
        online_update_interval=5,
        online_min_samples=5,
    )


@pytest.fixture
def train_df():
    return _make_regime_df(600, seed=42)


@pytest.fixture
def new_bars_df():
    """Small batch of new bars for online update."""
    return _make_regime_df(10, seed=99)


class TestRegimeDetectorCore:
    """Tests for basic fit/predict functionality."""

    def test_fit_and_predict_shape(self, config_offline, train_df):
        det = RegimeDetector(config_offline)
        det.fit(train_df)
        probs = det.predict_proba(train_df)
        assert probs.shape == (600, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_online_shape(self, config_offline, train_df):
        det = RegimeDetector(config_offline)
        det.fit(train_df)
        probs = det.predict_proba_online(train_df)
        assert probs.shape == (600, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_online_differs_from_forward_backward(self, config_offline, train_df):
        det = RegimeDetector(config_offline)
        det.fit(train_df)
        fb = det.predict_proba(train_df)
        fo = det.predict_proba_online(train_df)
        # Forward-only and forward-backward should differ (at least on some bars)
        assert not np.allclose(fb, fo, atol=1e-6)

    def test_state_names_mapped(self, config_offline, train_df):
        det = RegimeDetector(config_offline)
        det.fit(train_df)
        assert set(det.state_map.values()) == {0, 1, 2}

    def test_predict_before_fit_raises(self, config_offline, train_df):
        det = RegimeDetector(config_offline)
        with pytest.raises(RuntimeError, match="fitted"):
            det.predict_proba(train_df)

    def test_current_regime_online(self, config_offline, train_df):
        det = RegimeDetector(config_offline)
        det.fit(train_df)
        regime = det.current_regime_online(train_df)
        assert regime in ("bull", "bear", "choppy")

    def test_fallback_on_bad_data(self, config_offline):
        """HMM fit fails on constant data → fallback uniform probs."""
        df = pd.DataFrame({
            "log_return": np.zeros(100),
            "gk_vol_14": np.ones(100),
            "volume_zscore": np.zeros(100),
        })
        det = RegimeDetector(config_offline)
        det.fit(df)
        probs = det.predict_proba(df)
        np.testing.assert_allclose(probs, 1 / 3, atol=1e-5)


class TestOnlineUpdate:
    """Tests for online HMM parameter update."""

    def test_update_not_enabled_raises(self, config_offline, train_df, new_bars_df):
        det = RegimeDetector(config_offline)
        det.fit(train_df)
        with pytest.raises(RuntimeError, match="not enabled"):
            det.update_online(new_bars_df)

    def test_update_before_fit_raises(self, config_online, new_bars_df):
        det = RegimeDetector(config_online)
        with pytest.raises(RuntimeError, match="fit"):
            det.update_online(new_bars_df)

    def test_update_accumulates_bars(self, config_online, train_df, new_bars_df):
        det = RegimeDetector(config_online)
        det.fit(train_df)
        det.update_online(new_bars_df)
        assert det._online_stats.total_bars_seen == 10
        assert det._online_stats.bars_since_update == 5  # 10 >= 5 → reestimate resets to 0, then +5 remainder? Let's check

    def test_update_triggers_reestimate(self, config_online, train_df):
        """With interval=5, min_samples=5, feeding 5 bars should trigger reestimate."""
        det = RegimeDetector(config_online)
        det.fit(train_df)

        # Record original params
        orig_means = det.model.means_.copy()

        # Feed 5 bars of clearly different data (strong bull)
        bull_df = pd.DataFrame({
            "log_return": np.full(5, 0.01),
            "gk_vol_14": np.full(5, 0.005),
            "volume_zscore": np.full(5, 2.0),
        })
        updated = det.update_online(bull_df)
        assert updated is True
        # Params should have shifted (blend=0.3 toward new)
        assert not np.allclose(det.model.means_, orig_means, atol=1e-8)

    def test_state_map_preserved_after_update(self, config_online, train_df):
        det = RegimeDetector(config_online)
        det.fit(train_df)
        original_map = dict(det.state_map)

        # Run many updates
        for i in range(10):
            bars = _make_regime_df(10, seed=i + 100)
            det.update_online(bars)

        assert det.state_map == original_map

    def test_transmat_normalized_after_update(self, config_online, train_df):
        det = RegimeDetector(config_online)
        det.fit(train_df)

        bars = _make_regime_df(10, seed=200)
        det.update_online(bars)

        row_sums = det.model.transmat_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_covariance_floor(self, config_online, train_df):
        """Identical data should not collapse covariance below min_covar."""
        det = RegimeDetector(config_online)
        det.fit(train_df)

        # Feed identical bars
        constant_df = pd.DataFrame({
            "log_return": np.full(10, 0.0),
            "gk_vol_14": np.full(10, 0.01),
            "volume_zscore": np.full(10, 0.0),
        })
        det.update_online(constant_df)
        assert np.all(det.model.covars_ >= det.model.min_covar)

    def test_fallback_returns_false(self, config_online, new_bars_df):
        """Fallback model should return False without error."""
        df = pd.DataFrame({
            "log_return": np.zeros(100),
            "gk_vol_14": np.ones(100),
            "volume_zscore": np.zeros(100),
        })
        det = RegimeDetector(config_online)
        det.fit(df)  # Will enter fallback
        result = det.update_online(new_bars_df)
        assert result is False

    def test_blend_smoothing(self, config_online, train_df):
        """Parameters should move toward new data but not fully adopt it."""
        det = RegimeDetector(config_online)
        det.fit(train_df)

        orig_means = det.model.means_.copy()

        # Feed extreme bull data
        extreme_df = pd.DataFrame({
            "log_return": np.full(5, 0.05),
            "gk_vol_14": np.full(5, 0.001),
            "volume_zscore": np.full(5, 5.0),
        })
        det.update_online(extreme_df)

        # Means should have moved but not fully to the extreme
        diff = np.abs(det.model.means_ - orig_means).max()
        assert diff > 1e-6, "Parameters should have shifted"
        # Should not have jumped all the way to extreme standardized values
        assert diff < 1.0, "Blend should prevent extreme jumps"


class TestOnlineStatsSerialization:
    """Tests for get/set online stats dict."""

    def test_roundtrip(self, config_online, train_df, new_bars_df):
        det = RegimeDetector(config_online)
        det.fit(train_df)
        det.update_online(new_bars_df)

        # Export
        stats_dict = det.get_online_stats_dict()
        assert stats_dict is not None
        assert stats_dict["total_bars_seen"] == 10

        # Create a new detector, fit, then restore stats
        det2 = RegimeDetector(config_online)
        det2.fit(train_df)
        det2.set_online_stats_dict(stats_dict)

        assert det2._online_stats.total_bars_seen == 10
        np.testing.assert_array_almost_equal(
            det2._online_stats.N, det._online_stats.N
        )
        np.testing.assert_array_almost_equal(
            det2._online_stats.S, det._online_stats.S
        )

    def test_no_stats_returns_none(self, config_offline, train_df):
        det = RegimeDetector(config_offline)
        det.fit(train_df)
        assert det.get_online_stats_dict() is None
