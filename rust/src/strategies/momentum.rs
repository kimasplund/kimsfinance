//! Momentum-based trading strategies
//!
//! Strategies that capitalize on price momentum and oscillator signals.

use crate::backtest::core::{
    IndicatorConfig, IndicatorValues, OHLCVBar, ParameterGrid, ParameterRange, Signal, Strategy,
};

// ============================================================================
// RSI Mean Reversion Strategy
// ============================================================================

/// RSI Mean Reversion Strategy
///
/// Buys when RSI is oversold (<30) and sells when RSI normalizes (>50).
/// Conservative approach that works well in ranging markets.
///
/// # Default Parameters
/// - RSI Period: 14
/// - Buy Threshold: 30.0 (oversold)
/// - Sell Threshold: 50.0 (neutral)
///
/// # Optimization Ranges
/// - RSI Period: 10-20
/// - Buy Threshold: 20-35
/// - Sell Threshold: 50-65
///
/// # Market Conditions
/// - Best: Ranging/sideways markets with mean-reverting behavior
/// - Avoid: Strong trending markets (up or down)
///
/// # Risk Management
/// - Stop Loss: 5% below entry
/// - Take Profit: 10% above entry
#[derive(Debug, Clone)]
pub struct RSIMeanReversion {
    pub rsi_period: usize,
    pub buy_threshold: f64,
    pub sell_threshold: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
}

impl RSIMeanReversion {
    pub fn new(
        rsi_period: usize,
        buy_threshold: f64,
        sell_threshold: f64,
    ) -> Self {
        Self {
            rsi_period,
            buy_threshold,
            sell_threshold,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for RSIMeanReversion {
    fn default() -> Self {
        Self::new(14, 30.0, 50.0)
    }
}

impl Strategy for RSIMeanReversion {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("rsi_{}", self.rsi_period);
        let rsi = indicators.get(&key).copied().unwrap_or(50.0);

        if rsi < self.buy_threshold {
            Signal::Buy
        } else if rsi > self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::RSI {
            period: self.rsi_period,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "rsi_period",
            ParameterRange::Int { min: 10, max: 20, step: 2 },
        );
        grid.add_range(
            "buy_threshold",
            ParameterRange::Float { min: 20.0, max: 35.0, step: 5.0 },
        );
        grid.add_range(
            "sell_threshold",
            ParameterRange::Float { min: 50.0, max: 65.0, step: 5.0 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// RSI Oversold/Overbought Strategy
// ============================================================================

/// RSI Oversold/Overbought Strategy (Aggressive)
///
/// Buys when RSI is deeply oversold (<20) and sells when overbought (>80).
/// More aggressive than mean reversion, aims for larger moves.
///
/// # Default Parameters
/// - RSI Period: 14
/// - Oversold: 20.0
/// - Overbought: 80.0
///
/// # Optimization Ranges
/// - RSI Period: 9-21
/// - Oversold: 15-30
/// - Overbought: 70-85
///
/// # Market Conditions
/// - Best: Volatile markets with strong reversals
/// - Avoid: Low volatility, choppy markets
///
/// # Risk Management
/// - Stop Loss: 7% below entry
/// - Take Profit: 15% above entry
#[derive(Debug, Clone)]
pub struct RSIOversoldOverbought {
    pub rsi_period: usize,
    pub oversold_threshold: f64,
    pub overbought_threshold: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
}

impl RSIOversoldOverbought {
    pub fn new(
        rsi_period: usize,
        oversold_threshold: f64,
        overbought_threshold: f64,
    ) -> Self {
        Self {
            rsi_period,
            oversold_threshold,
            overbought_threshold,
            stop_loss_pct: 0.07,
            take_profit_pct: 0.15,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for RSIOversoldOverbought {
    fn default() -> Self {
        Self::new(14, 20.0, 80.0)
    }
}

impl Strategy for RSIOversoldOverbought {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("rsi_{}", self.rsi_period);
        let rsi = indicators.get(&key).copied().unwrap_or(50.0);

        if rsi < self.oversold_threshold {
            Signal::Buy
        } else if rsi > self.overbought_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::RSI {
            period: self.rsi_period,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "rsi_period",
            ParameterRange::Int { min: 9, max: 21, step: 3 },
        );
        grid.add_range(
            "oversold_threshold",
            ParameterRange::Float { min: 15.0, max: 30.0, step: 5.0 },
        );
        grid.add_range(
            "overbought_threshold",
            ParameterRange::Float { min: 70.0, max: 85.0, step: 5.0 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// MACD Trend Following Strategy
// ============================================================================

/// MACD Trend Following Strategy
///
/// Follows the trend using MACD line and signal line crossovers.
/// Buys when MACD crosses above signal, sells when crosses below.
///
/// # Default Parameters
/// - Fast EMA: 12
/// - Slow EMA: 26
/// - Signal Line: 9
///
/// # Optimization Ranges
/// - Fast EMA: 8-16
/// - Slow EMA: 20-30
/// - Signal Line: 7-12
///
/// # Market Conditions
/// - Best: Trending markets with clear directional moves
/// - Avoid: Choppy, range-bound markets
///
/// # Risk Management
/// - Stop Loss: 4% below entry
/// - Take Profit: 12% above entry
#[derive(Debug, Clone)]
pub struct MACDTrendFollowing {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
    prev_macd: f64,
    prev_signal: f64,
}

impl MACDTrendFollowing {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            signal_period,
            stop_loss_pct: 0.04,
            take_profit_pct: 0.12,
            initial_capital: 10_000.0,
            prev_macd: 0.0,
            prev_signal: 0.0,
        }
    }
}

impl Default for MACDTrendFollowing {
    fn default() -> Self {
        Self::new(12, 26, 9)
    }
}

impl Strategy for MACDTrendFollowing {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("macd_{}_{}_{}", self.fast_period, self.slow_period, self.signal_period);
        let macd_key = format!("{}_macd", key);
        let signal_key = format!("{}_signal", key);

        let macd = indicators.get(&macd_key).copied().unwrap_or(0.0);
        let signal = indicators.get(&signal_key).copied().unwrap_or(0.0);

        let result = if macd > signal && self.prev_macd <= self.prev_signal {
            Signal::Buy
        } else if macd < signal && self.prev_macd >= self.prev_signal {
            Signal::Sell
        } else {
            Signal::Hold
        };

        self.prev_macd = macd;
        self.prev_signal = signal;

        result
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::MACD {
            fast: self.fast_period,
            slow: self.slow_period,
            signal: self.signal_period,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "fast_period",
            ParameterRange::Int { min: 8, max: 16, step: 2 },
        );
        grid.add_range(
            "slow_period",
            ParameterRange::Int { min: 20, max: 30, step: 2 },
        );
        grid.add_range(
            "signal_period",
            ParameterRange::Int { min: 7, max: 12, step: 1 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// MACD Divergence Strategy
// ============================================================================

/// MACD Divergence Strategy
///
/// Uses MACD histogram divergence to identify potential reversals.
/// Buys when histogram turns positive, sells when negative.
///
/// # Default Parameters
/// - Fast EMA: 12
/// - Slow EMA: 26
/// - Signal Line: 9
/// - Histogram Threshold: 0.0
///
/// # Optimization Ranges
/// - Fast EMA: 8-16
/// - Slow EMA: 20-30
/// - Signal Line: 7-12
///
/// # Market Conditions
/// - Best: Markets showing momentum shifts
/// - Avoid: Low volatility sideways markets
///
/// # Risk Management
/// - Stop Loss: 5% below entry
/// - Take Profit: 10% above entry
#[derive(Debug, Clone)]
pub struct MACDDivergence {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
    pub histogram_threshold: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
}

impl MACDDivergence {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            signal_period,
            histogram_threshold: 0.0,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for MACDDivergence {
    fn default() -> Self {
        Self::new(12, 26, 9)
    }
}

impl Strategy for MACDDivergence {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("macd_{}_{}_{}", self.fast_period, self.slow_period, self.signal_period);
        let histogram_key = format!("{}_histogram", key);

        let histogram = indicators.get(&histogram_key).copied().unwrap_or(0.0);

        if histogram > self.histogram_threshold {
            Signal::Buy
        } else if histogram < -self.histogram_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::MACD {
            fast: self.fast_period,
            slow: self.slow_period,
            signal: self.signal_period,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "fast_period",
            ParameterRange::Int { min: 8, max: 16, step: 2 },
        );
        grid.add_range(
            "slow_period",
            ParameterRange::Int { min: 20, max: 30, step: 2 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// Stochastic Oscillator Strategy
// ============================================================================

/// Stochastic Oscillator Strategy
///
/// Uses %K and %D lines to identify overbought/oversold conditions.
/// Buys when %K crosses above %D in oversold zone, sells when crosses below in overbought.
///
/// # Default Parameters
/// - K Period: 14
/// - D Period: 3
/// - Oversold: 20.0
/// - Overbought: 80.0
///
/// # Optimization Ranges
/// - K Period: 10-20
/// - D Period: 3-7
/// - Oversold: 15-25
/// - Overbought: 75-85
///
/// # Market Conditions
/// - Best: Ranging markets with clear support/resistance
/// - Avoid: Strong trending markets
///
/// # Risk Management
/// - Stop Loss: 4% below entry
/// - Take Profit: 8% above entry
#[derive(Debug, Clone)]
pub struct StochasticOscillator {
    pub k_period: usize,
    pub d_period: usize,
    pub oversold_threshold: f64,
    pub overbought_threshold: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
    prev_k: f64,
    prev_d: f64,
}

impl StochasticOscillator {
    pub fn new(
        k_period: usize,
        d_period: usize,
        oversold_threshold: f64,
        overbought_threshold: f64,
    ) -> Self {
        Self {
            k_period,
            d_period,
            oversold_threshold,
            overbought_threshold,
            stop_loss_pct: 0.04,
            take_profit_pct: 0.08,
            initial_capital: 10_000.0,
            prev_k: 50.0,
            prev_d: 50.0,
        }
    }
}

impl Default for StochasticOscillator {
    fn default() -> Self {
        Self::new(14, 3, 20.0, 80.0)
    }
}

impl Strategy for StochasticOscillator {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("stoch_{}_{}", self.k_period, self.d_period);
        let k_key = format!("{}_k", key);
        let d_key = format!("{}_d", key);

        let k = indicators.get(&k_key).copied().unwrap_or(50.0);
        let d = indicators.get(&d_key).copied().unwrap_or(50.0);

        let result = if k > d && self.prev_k <= self.prev_d && k < self.oversold_threshold {
            Signal::Buy
        } else if k < d && self.prev_k >= self.prev_d && k > self.overbought_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        };

        self.prev_k = k;
        self.prev_d = d;

        result
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::Stochastic {
            k_period: self.k_period,
            d_period: self.d_period,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "k_period",
            ParameterRange::Int { min: 10, max: 20, step: 2 },
        );
        grid.add_range(
            "d_period",
            ParameterRange::Int { min: 3, max: 7, step: 1 },
        );
        grid.add_range(
            "oversold_threshold",
            ParameterRange::Float { min: 15.0, max: 25.0, step: 5.0 },
        );
        grid.add_range(
            "overbought_threshold",
            ParameterRange::Float { min: 75.0, max: 85.0, step: 5.0 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// ROC Breakout Strategy
// ============================================================================

/// Rate of Change (ROC) Breakout Strategy
///
/// Identifies momentum breakouts using ROC acceleration.
/// Buys when ROC exceeds positive threshold, sells when falls below negative threshold.
///
/// # Default Parameters
/// - ROC Period: 12
/// - Buy Threshold: 2.0% (positive momentum)
/// - Sell Threshold: -2.0% (negative momentum)
///
/// # Optimization Ranges
/// - ROC Period: 8-20
/// - Buy Threshold: 1.0-5.0%
/// - Sell Threshold: -1.0 to -5.0%
///
/// # Market Conditions
/// - Best: Breakout markets with strong directional moves
/// - Avoid: Low volatility, consolidating markets
///
/// # Risk Management
/// - Stop Loss: 6% below entry
/// - Take Profit: 15% above entry
#[derive(Debug, Clone)]
pub struct ROCBreakout {
    pub roc_period: usize,
    pub buy_threshold: f64,
    pub sell_threshold: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
}

impl ROCBreakout {
    pub fn new(roc_period: usize, buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            roc_period,
            buy_threshold,
            sell_threshold,
            stop_loss_pct: 0.06,
            take_profit_pct: 0.15,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for ROCBreakout {
    fn default() -> Self {
        Self::new(12, 2.0, -2.0)
    }
}

impl Strategy for ROCBreakout {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("roc_{}", self.roc_period);
        let roc = indicators.get(&key).copied().unwrap_or(0.0);

        if roc > self.buy_threshold {
            Signal::Buy
        } else if roc < self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::ROC {
            period: self.roc_period,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "roc_period",
            ParameterRange::Int { min: 8, max: 20, step: 2 },
        );
        grid.add_range(
            "buy_threshold",
            ParameterRange::Float { min: 1.0, max: 5.0, step: 1.0 },
        );
        grid.add_range(
            "sell_threshold",
            ParameterRange::Float { min: -5.0, max: -1.0, step: 1.0 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// CCI Reversal Strategy
// ============================================================================

/// Commodity Channel Index (CCI) Reversal Strategy
///
/// Uses CCI extreme values to identify reversal points.
/// Buys when CCI is extremely oversold (<-100), sells when extremely overbought (>100).
///
/// # Default Parameters
/// - CCI Period: 20
/// - Oversold: -100.0
/// - Overbought: 100.0
///
/// # Optimization Ranges
/// - CCI Period: 14-30
/// - Oversold: -150 to -80
/// - Overbought: 80 to 150
///
/// # Market Conditions
/// - Best: Volatile markets with strong reversals
/// - Avoid: Low volatility trending markets
///
/// # Risk Management
/// - Stop Loss: 5% below entry
/// - Take Profit: 10% above entry
#[derive(Debug, Clone)]
pub struct CCIReversal {
    pub cci_period: usize,
    pub oversold_threshold: f64,
    pub overbought_threshold: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
}

impl CCIReversal {
    pub fn new(
        cci_period: usize,
        oversold_threshold: f64,
        overbought_threshold: f64,
    ) -> Self {
        Self {
            cci_period,
            oversold_threshold,
            overbought_threshold,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for CCIReversal {
    fn default() -> Self {
        Self::new(20, -100.0, 100.0)
    }
}

impl Strategy for CCIReversal {
    fn on_data(&mut self, _bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("cci_{}", self.cci_period);
        let cci = indicators.get(&key).copied().unwrap_or(0.0);

        if cci < self.oversold_threshold {
            Signal::Buy
        } else if cci > self.overbought_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::CCI {
            period: self.cci_period,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "cci_period",
            ParameterRange::Int { min: 14, max: 30, step: 2 },
        );
        grid.add_range(
            "oversold_threshold",
            ParameterRange::Float { min: -150.0, max: -80.0, step: 10.0 },
        );
        grid.add_range(
            "overbought_threshold",
            ParameterRange::Float { min: 80.0, max: 150.0, step: 10.0 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}
