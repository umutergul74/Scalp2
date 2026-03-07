"""Typed configuration loader for Scalp2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class DateRange:
    start: str = "2021-01-01"
    end: str = "2026-02-20"


@dataclass
class TimeframeConfig:
    primary: str = "15m"
    mtf: List[str] = field(default_factory=lambda: ["1h", "4h"])


@dataclass
class DataConfig:
    symbol: str = "BTC/USDT"
    exchange: str = "binanceusdm"
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)
    date_range: DateRange = field(default_factory=DateRange)
    cache_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"


# ── Features ──────────────────────────────────────────────────────────────────

@dataclass
class WaveletConfig:
    wavelet: str = "sym5"
    level: int = 4
    threshold_mode: str = "soft"
    window: int = 256
    apply_to: List[str] = field(default_factory=lambda: ["close", "volume"])


@dataclass
class BollingerConfig:
    period: int = 20
    std: float = 2.0


@dataclass
class StochasticConfig:
    k: int = 14
    d: int = 3
    smooth: int = 3


@dataclass
class TechnicalConfig:
    rsi_period: int = 14
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 55])
    macd: List[int] = field(default_factory=lambda: [12, 26, 9])
    bollinger: BollingerConfig = field(default_factory=BollingerConfig)
    atr_period: int = 14
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)
    adx_period: int = 14


@dataclass
class VolatilityConfig:
    garman_klass_window: int = 14
    parkinson_window: int = 14


@dataclass
class OrderFlowConfig:
    cvd_proxy: bool = True
    funding_rate: bool = True
    open_interest: bool = True


@dataclass
class SmartMoneyConfig:
    fvg_min_gap_pct: float = 0.001
    liquidity_sweep_lookback: int = 20
    vwap_session_hours: int = 24


@dataclass
class FeatureConfig:
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    orderflow: OrderFlowConfig = field(default_factory=OrderFlowConfig)
    smart_money: SmartMoneyConfig = field(default_factory=SmartMoneyConfig)


# ── Labeling ──────────────────────────────────────────────────────────────────

@dataclass
class LabelConfig:
    method: str = "triple_barrier"
    tp_multiplier: float = 1.2
    sl_multiplier: float = 1.0
    max_holding_bars: int = 10
    min_return_threshold: float = 0.0005
    atr_period: int = 14


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass
class TCNConfig:
    num_channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    kernel_size: int = 3
    dropout: float = 0.2


@dataclass
class GRUConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False


@dataclass
class FusionConfig:
    latent_dim: int = 128
    dropout: float = 0.3


@dataclass
class XGBoostConfig:
    objective: str = "multi:softprob"
    num_class: int = 3
    max_depth: int = 5
    n_estimators: int = 500
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    min_child_weight: int = 50
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    device: str = "cuda"
    eval_metric: str = "mlogloss"
    early_stopping_rounds: int = 30


@dataclass
class ModelConfig:
    seq_len: int = 64
    tcn: TCNConfig = field(default_factory=TCNConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    handcrafted_top_k: int = 40


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    type: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    type: str = "ReduceLROnPlateau"
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 1e-5


@dataclass
class LossConfig:
    primary: str = "cross_entropy"
    auxiliary: str = "sharpe"
    alpha_start: float = 1.0
    alpha_end: float = 0.5
    alpha_anneal_epochs: int = 20


@dataclass
class WalkForwardConfig:
    train_bars: int = 20160
    val_bars: int = 4320
    test_bars: int = 2880
    purge_bars: int = 48
    embargo_bars: int = 12
    step_bars: int = 2880


@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    batch_size: int = 256
    max_epochs: int = 100
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    class_weights: str = "inverse_frequency"
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    scaler: str = "robust"


# ── Regime ────────────────────────────────────────────────────────────────────

@dataclass
class RegimeConfig:
    n_states: int = 3
    features: List[str] = field(
        default_factory=lambda: ["log_return", "gk_vol_14", "volume_zscore"]
    )
    covariance_type: str = "diag"
    n_iter: int = 100
    choppy_threshold: float = 0.5


# ── Execution ─────────────────────────────────────────────────────────────────

@dataclass
class PositionSizingConfig:
    method: str = "fractional_kelly"
    max_fraction: float = 0.02
    kelly_fraction: float = 0.25


@dataclass
class TradeManagementConfig:
    partial_tp_1_atr: float = 0.6
    partial_tp_1_pct: float = 0.5
    full_tp_atr: float = 1.2
    breakeven_trigger_atr: float = 0.6
    trailing_activation_atr: float = 0.8
    trailing_distance_atr: float = 0.5


@dataclass
class ModelRefreshConfig:
    frequency: str = "weekly"
    min_sharpe_threshold: float = 0.5
    fallback: bool = True


@dataclass
class OrderExecutionConfig:
    order_type: str = "limit"
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 4.0
    slippage_bps: float = 2.0


@dataclass
class ExecutionConfig:
    confidence_threshold: float = 0.70
    max_trades_per_day: int = 2
    min_adx: float = 20.0
    min_atr_percentile: float = 0.15
    order_execution: OrderExecutionConfig = field(
        default_factory=OrderExecutionConfig
    )
    position_sizing: PositionSizingConfig = field(
        default_factory=PositionSizingConfig
    )
    trade_management: TradeManagementConfig = field(
        default_factory=TradeManagementConfig
    )
    model_refresh: ModelRefreshConfig = field(default_factory=ModelRefreshConfig)


# ── Logging ───────────────────────────────────────────────────────────────────

@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "./logs"
    tensorboard: bool = True


# ── Root ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42


def _build_dataclass(cls, data: dict):
    """Recursively build a dataclass from a nested dict."""
    if data is None:
        return cls()
    import dataclasses

    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, val in data.items():
        if key not in fieldtypes:
            continue
        ft = fieldtypes[key]
        # Resolve string type annotations
        if isinstance(ft, str):
            ft = eval(ft)  # noqa: S307 – safe, our own annotations
        if dataclasses.is_dataclass(ft) and isinstance(val, dict):
            kwargs[key] = _build_dataclass(ft, val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from YAML file into typed dataclass."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _build_dataclass(Config, raw)
