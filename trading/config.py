"""
Trading Config — single source of truth for all strategy parameters.

Every tunable parameter lives here. Strategies reference this config,
never hardcode their own constants. This enables:
  - One place to see all parameters
  - Easy A/B testing (swap configs)
  - Backtester can override any parameter
  - No more scattered constants across 8+ files

Usage:
    from trading.config import config

    # Read a parameter
    sl = config.level_bounce.sl_beyond_extreme

    # Override for backtesting
    from trading.config import TradingConfig, LevelBounceConfig
    test_config = TradingConfig(level_bounce=LevelBounceConfig(min_rr=2.0))

    # Or load from a preset
    config = TradingConfig.aggressive()
"""
import os
from dataclasses import dataclass, field
from typing import List


# ══════════════════════════════════════════════
# Strategy Configs — one per setup type
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class LevelBounceConfig:
    """Level Bounce detector parameters."""
    lookback: int = 3              # Candles on each side to confirm local extreme
    level_proximity: float = 30    # Max pts from a level to count as "at level"
    min_reversal: float = 30       # Min pts price must move to confirm reversal
    sl_beyond_extreme: float = 20  # SL placed this many pts beyond the extreme
    min_rr: float = 1.5            # Minimum risk:reward ratio
    min_level_score: int = 20      # Minimum level strength to trade
    max_signals: int = 11           # Max signals per scan cycle


@dataclass(frozen=True)
class LevelRetestConfig:
    """Level Break + Retest detector parameters."""
    min_level_score: int = 25
    break_confirmation: int = 2    # Candles price must stay beyond level after break
    retest_proximity_atr: float = 0.15  # How close price must return (% of ATR)
    retest_hold_candles: int = 2   # Candles the retest must hold for
    sl_beyond_level_atr: float = 0.10
    min_rr: float = 2.0
    max_signals: int = 2


@dataclass(frozen=True)
class VWAPFadeConfig:
    """VWAP Fade detector parameters."""
    min_deviation_pct: float = 0.8   # Price must be > 0.8% from VWAP
    rsi_overbought: int = 65
    rsi_oversold: int = 35
    sl_beyond_extreme_atr: float = 0.2
    min_rr: float = 1.5
    max_signals: int = 1


@dataclass(frozen=True)
class TradeManagerConfig:
    """Active trade management parameters."""
    breakeven_at_r: float = 0.5    # Move SL to breakeven at this R multiple
    partial_at_r: float = 1.5      # Take 50% profit at this R multiple
    time_stop_candles: int = 6     # 30 min — protects against big losers
    time_stop_min_move: float = 0.2  # "No progress" threshold in R multiples
    trail_atr_factor: float = 0.5  # Trail at 0.5 ATR after partial
    tight_trail_atr_factor: float = 0.3  # Tighter trail at 2R+
    tight_trail_at_r: float = 2.0  # When to switch to tight trail


@dataclass(frozen=True)
class SweetSpotConfig:
    """Sweet Spot Filter parameters."""
    min_level_score: int = 20      # Reject signals at levels below this score
    confirmations_required: int = 3  # Need 3 of 5 checks to pass
    min_confluence_score: int = 40   # Indicator confluence threshold


# ══════════════════════════════════════════════
# Risk Management Config
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class RiskConfig:
    """Risk engine parameters."""
    max_risk_per_trade_pct: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
    max_position_size_pct: float = float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0"))
    min_risk_reward: float = 1.5
    min_confidence: float = 0.55
    max_open_positions: int = 11
    default_capital: float = float(os.getenv("DEFAULT_CAPITAL", "500000"))


# ══════════════════════════════════════════════
# Straddle Config
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class StraddleConfig:
    """Straddle lifecycle parameters."""
    hard_stop_multiplier: float = 1.3    # Close if combined > 1.3x sold
    shift_threshold: int = 250           # Shift when NIFTY moves > 250 pts
    max_shifts_per_day: int = 2
    vix_calm: float = 14
    vix_elevated: float = 20
    scenario_offsets: tuple = (300, 150, 0, -150, -400, -700)


# ══════════════════════════════════════════════
# Level Map Config
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class LevelConfig:
    """Level detection and scoring parameters."""
    cluster_distance: float = 30         # Levels within this distance reinforce each other
    pdh_pdl_base_score: int = 25
    camarilla_base_score: int = 25
    swing_base_score: int = 30
    round_base_score: int = 20
    vwap_base_score: int = 20
    orb_base_score: int = 15
    week_hl_base_score: int = 20
    gap_base_score: int = 15
    round_step_nifty: int = 100          # Round number interval for NIFTY
    round_step_stock: int = 50           # Round number interval for stocks > 500


# ══════════════════════════════════════════════
# Scanner Config
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class ScannerConfig:
    """Premarket scanner parameters."""
    top_n: int = 10                      # Top N stocks for watchlist
    scan_days: int = 5                   # Historical days for scoring
    atr_high_threshold: float = 2.0      # ATR% for full score
    volume_spike_threshold: float = 1.5  # Volume ratio for full score
    level_proximity_bonus: int = 15      # Extra points for being near a scored level


# ══════════════════════════════════════════════
# Broker Config
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class BrokerConfig:
    """Broker API parameters."""
    rate_limit_interval: float = 0.4     # Min seconds between API calls
    ltp_cache_ttl: int = 5               # Seconds to cache LTP data
    candle_retry_count: int = 2          # Retries on rate limit
    candle_retry_delay: float = 2.0      # Seconds between retries
    batch_size: int = 50                 # Max instruments per batch call


# ══════════════════════════════════════════════
# Backtest Config
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class BacktestConfig:
    """Backtester parameters."""
    slippage_atr_pct: float = 0.05       # Slippage = 5% of ATR
    include_slippage: bool = True
    include_commission: bool = True
    walk_forward_train_days: int = 10
    walk_forward_test_days: int = 5
    walk_forward_step_days: int = 5


# ══════════════════════════════════════════════
# Master Config — composes all sub-configs
# ══════════════════════════════════════════════

@dataclass
class TradingConfig:
    """
    Master config for the entire trading system.

    Compose sub-configs for each module. Override any parameter
    by passing a modified sub-config to the constructor.
    """
    # Strategy configs
    level_bounce: LevelBounceConfig = field(default_factory=LevelBounceConfig)
    level_retest: LevelRetestConfig = field(default_factory=LevelRetestConfig)
    vwap_fade: VWAPFadeConfig = field(default_factory=VWAPFadeConfig)
    trade_manager: TradeManagerConfig = field(default_factory=TradeManagerConfig)
    sweet_spot: SweetSpotConfig = field(default_factory=SweetSpotConfig)

    # System configs
    risk: RiskConfig = field(default_factory=RiskConfig)
    straddle: StraddleConfig = field(default_factory=StraddleConfig)
    levels: LevelConfig = field(default_factory=LevelConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Mode
    trading_mode: str = os.getenv("TRADING_MODE", "paper")
    planner_mode: str = os.getenv("PLANNER_MODE", "cli")

    @classmethod
    def default(cls) -> "TradingConfig":
        """Standard config — balanced risk/reward."""
        return cls()

    @classmethod
    def aggressive(cls) -> "TradingConfig":
        """Higher risk tolerance, more trades, tighter management."""
        return cls(
            level_bounce=LevelBounceConfig(min_rr=1.3, min_level_score=15, max_signals=5),
            level_retest=LevelRetestConfig(min_rr=1.5),
            trade_manager=TradeManagerConfig(
                partial_at_r=1.0,       # Take profit earlier
                time_stop_candles=4,     # Cut faster
            ),
            risk=RiskConfig(
                max_risk_per_trade_pct=1.5,
                max_open_positions=5,
                min_confidence=0.50,
            ),
            straddle=StraddleConfig(
                hard_stop_multiplier=1.5,   # Wider stop
                shift_threshold=150,         # Shift more often
                max_shifts_per_day=3,
            ),
        )

    @classmethod
    def conservative(cls) -> "TradingConfig":
        """Lower risk, fewer but higher-quality trades."""
        return cls(
            level_bounce=LevelBounceConfig(min_rr=2.0, min_level_score=30, max_signals=2),
            level_retest=LevelRetestConfig(min_rr=2.5, min_level_score=35),
            trade_manager=TradeManagerConfig(
                partial_at_r=2.0,        # Let winners run longer
                time_stop_candles=8,     # More patience
            ),
            sweet_spot=SweetSpotConfig(
                min_level_score=30,
                confirmations_required=4,  # Need 4 of 5
            ),
            risk=RiskConfig(
                max_risk_per_trade_pct=0.5,
                max_open_positions=2,
                min_confidence=0.65,
            ),
            straddle=StraddleConfig(
                hard_stop_multiplier=1.2,
                shift_threshold=250,
                max_shifts_per_day=1,
            ),
        )

    def summary(self) -> str:
        """Human-readable config summary."""
        return (
            f"Trading Config ({self.trading_mode.upper()}):\n"
            f"  Risk: {self.risk.max_risk_per_trade_pct}%/trade, "
            f"{self.risk.max_daily_loss_pct}%/day, "
            f"max {self.risk.max_open_positions} positions\n"
            f"  Level Bounce: min_rr={self.level_bounce.min_rr}, "
            f"min_score={self.level_bounce.min_level_score}\n"
            f"  Trade Mgmt: BE@{self.trade_manager.breakeven_at_r}R, "
            f"partial@{self.trade_manager.partial_at_r}R, "
            f"time_stop={self.trade_manager.time_stop_candles} candles\n"
            f"  Straddle: stop@{self.straddle.hard_stop_multiplier}x, "
            f"shift>{self.straddle.shift_threshold}pts, "
            f"max {self.straddle.max_shifts_per_day}/day"
        )


# ── Module-level singleton ──
config = TradingConfig.default()
