//! Trend-following trading strategies
//!
//! Strategies that identify and follow market trends using moving averages
//! and channel breakouts.

use crate::backtest::core::{
    IndicatorConfig, IndicatorValues, OHLCVBar, ParameterGrid, ParameterRange, Signal, Strategy,
};

// ============================================================================
// EMA Crossover Strategy
// ============================================================================

/// EMA Crossover Strategy (Golden Cross/Death Cross)
///
/// Classic trend-following strategy using fast and slow EMA crossovers.
/// Buys when fast EMA crosses above slow EMA (Golden Cross).
/// Sells when fast EMA crosses below slow EMA (Death Cross).
///
/// # Default Parameters
/// - Fast EMA: 50 (short-term trend)
/// - Slow EMA: 200 (long-term trend)
///
/// # Optimization Ranges
/// - Fast EMA: 20-100
/// - Slow EMA: 100-250
///
/// # Market Conditions
/// - Best: Strong trending markets (bull or bear)
/// - Avoid: Choppy, sideways markets (whipsaws)
///
/// # Risk Management
/// - Stop Loss: 3% below entry
/// - Take Profit: 20% above entry (ride the trend)
#[derive(Debug, Clone)]
pub struct EMACrossover {
    pub fast_period: usize,
    pub slow_period: usize,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
    prev_fast_ema: f64,
    prev_slow_ema: f64,
}

impl EMACrossover {
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            stop_loss_pct: 0.03,
            take_profit_pct: 0.20,
            initial_capital: 10_000.0,
            prev_fast_ema: 0.0,
            prev_slow_ema: 0.0,
        }
    }
}

impl Default for EMACrossover {
    fn default() -> Self {
        Self::new(50, 200)
    }
}

impl Strategy for EMACrossover {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let fast_key = format!("ema_{}", self.fast_period);
        let slow_key = format!("ema_{}", self.slow_period);

        let fast_ema = indicators.get(&fast_key).copied().unwrap_or(0.0);
        let slow_ema = indicators.get(&slow_key).copied().unwrap_or(0.0);

        let result = if fast_ema > slow_ema && self.prev_fast_ema <= self.prev_slow_ema {
            Signal::Buy
        } else if fast_ema < slow_ema && self.prev_fast_ema >= self.prev_slow_ema {
            Signal::Sell
        } else {
            Signal::Hold
        };

        self.prev_fast_ema = fast_ema;
        self.prev_slow_ema = slow_ema;

        result
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::EMA { period: self.fast_period },
            IndicatorConfig::EMA { period: self.slow_period },
        ]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "fast_period",
            ParameterRange::Int { min: 20, max: 100, step: 10 },
        );
        grid.add_range(
            "slow_period",
            ParameterRange::Int { min: 100, max: 250, step: 25 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// Triple EMA Trend Strategy
// ============================================================================

/// Triple EMA Trend Strategy
///
/// Uses three EMAs to identify trend strength and direction.
/// Buys when short > medium > long (strong uptrend).
/// Sells when short < medium < long (strong downtrend).
///
/// # Default Parameters
/// - Short EMA: 8 (immediate trend)
/// - Medium EMA: 21 (intermediate trend)
/// - Long EMA: 55 (major trend)
///
/// # Optimization Ranges
/// - Short EMA: 5-15
/// - Medium EMA: 15-30
/// - Long EMA: 40-100
///
/// # Market Conditions
/// - Best: Clear trending markets with momentum
/// - Avoid: Ranging markets with frequent crossovers
///
/// # Risk Management
/// - Stop Loss: 4% below entry
/// - Take Profit: 18% above entry
#[derive(Debug, Clone)]
pub struct TripleEMATrend {
    pub short_period: usize,
    pub medium_period: usize,
    pub long_period: usize,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
}

impl TripleEMATrend {
    pub fn new(short_period: usize, medium_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            medium_period,
            long_period,
            stop_loss_pct: 0.04,
            take_profit_pct: 0.18,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for TripleEMATrend {
    fn default() -> Self {
        Self::new(8, 21, 55)
    }
}

impl Strategy for TripleEMATrend {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let short_key = format!("ema_{}", self.short_period);
        let medium_key = format!("ema_{}", self.medium_period);
        let long_key = format!("ema_{}", self.long_period);

        let short_ema = indicators.get(&short_key).copied().unwrap_or(0.0);
        let medium_ema = indicators.get(&medium_key).copied().unwrap_or(0.0);
        let long_ema = indicators.get(&long_key).copied().unwrap_or(0.0);

        if short_ema > medium_ema && medium_ema > long_ema {
            Signal::Buy
        } else if short_ema < medium_ema && medium_ema < long_ema {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::EMA { period: self.short_period },
            IndicatorConfig::EMA { period: self.medium_period },
            IndicatorConfig::EMA { period: self.long_period },
        ]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "short_period",
            ParameterRange::Int { min: 5, max: 15, step: 2 },
        );
        grid.add_range(
            "medium_period",
            ParameterRange::Int { min: 15, max: 30, step: 5 },
        );
        grid.add_range(
            "long_period",
            ParameterRange::Int { min: 40, max: 100, step: 10 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// Donchian Channel Breakout Strategy
// ============================================================================

/// Donchian Channel Breakout Strategy (Turtle Trading)
///
/// Classic turtle trading strategy using Donchian channels.
/// Buys on breakout above upper channel (new 20-day high).
/// Sells on breakout below lower channel (new 20-day low).
///
/// # Default Parameters
/// - Channel Period: 20 (turtle standard)
/// - Breakout Confirmation: Price must close above/below channel
///
/// # Optimization Ranges
/// - Channel Period: 10-40
///
/// # Market Conditions
/// - Best: Markets breaking out of consolidation
/// - Avoid: Range-bound markets (false breakouts)
///
/// # Risk Management
/// - Stop Loss: 2% below entry (tight for breakouts)
/// - Take Profit: 25% above entry (capture full breakout)
#[derive(Debug, Clone)]
pub struct DonchianBreakout {
    pub channel_period: usize,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
}

impl DonchianBreakout {
    pub fn new(channel_period: usize) -> Self {
        Self {
            channel_period,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.25,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for DonchianBreakout {
    fn default() -> Self {
        Self::new(20)
    }
}

impl Strategy for DonchianBreakout {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("dc_{}", self.channel_period);
        let upper_key = format!("{}_upper", key);
        let lower_key = format!("{}_lower", key);

        let upper_channel = indicators.get(&upper_key).copied().unwrap_or(bar.high);
        let lower_channel = indicators.get(&lower_key).copied().unwrap_or(bar.low);

        if bar.close > upper_channel {
            Signal::Buy
        } else if bar.close < lower_channel {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "channel_period",
            ParameterRange::Int { min: 10, max: 40, step: 5 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// Keltner Channel Trend Strategy
// ============================================================================

/// Keltner Channel Trend Strategy
///
/// Uses Keltner Channels (EMA + ATR) to identify trend strength.
/// Buys when price breaks above upper channel (strong uptrend).
/// Sells when price breaks below lower channel (strong downtrend).
///
/// # Default Parameters
/// - EMA Period: 20 (middle line)
/// - ATR Period: 10 (channel width)
/// - ATR Multiplier: 2.0 (channel distance)
///
/// # Optimization Ranges
/// - EMA Period: 10-30
/// - ATR Period: 5-20
/// - ATR Multiplier: 1.5-3.0
///
/// # Market Conditions
/// - Best: Trending markets with expanding volatility
/// - Avoid: Low volatility consolidation
///
/// # Risk Management
/// - Stop Loss: ATR-based (1.5x ATR below entry)
/// - Take Profit: 15% above entry
#[derive(Debug, Clone)]
pub struct KeltnerTrend {
    pub ema_period: usize,
    pub atr_period: usize,
    pub atr_multiplier: f64,
    pub stop_loss_atr_mult: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
}

impl KeltnerTrend {
    pub fn new(ema_period: usize, atr_period: usize, atr_multiplier: f64) -> Self {
        Self {
            ema_period,
            atr_period,
            atr_multiplier,
            stop_loss_atr_mult: 1.5,
            take_profit_pct: 0.15,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for KeltnerTrend {
    fn default() -> Self {
        Self::new(20, 10, 2.0)
    }
}

impl Strategy for KeltnerTrend {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("kc_{}_{}", self.ema_period, self.atr_period);
        let upper_key = format!("{}_upper", key);
        let lower_key = format!("{}_lower", key);

        let upper_channel = indicators.get(&upper_key).copied().unwrap_or(bar.high);
        let lower_channel = indicators.get(&lower_key).copied().unwrap_or(bar.low);

        if bar.close > upper_channel {
            Signal::Buy
        } else if bar.close < lower_channel {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "ema_period",
            ParameterRange::Int { min: 10, max: 30, step: 5 },
        );
        grid.add_range(
            "atr_period",
            ParameterRange::Int { min: 5, max: 20, step: 5 },
        );
        grid.add_range(
            "atr_multiplier",
            ParameterRange::Float { min: 1.5, max: 3.0, step: 0.5 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}
