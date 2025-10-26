//! Composite multi-indicator trading strategies
//!
//! Strategies that combine momentum, trend, and volatility indicators
//! for higher-confidence signals with reduced false positives.

use crate::backtest::core::{
    IndicatorConfig, IndicatorValues, OHLCVBar, ParameterGrid, ParameterRange, Signal, Strategy,
};

// ============================================================================
// RSI + ATR Strategy (Momentum + Volatility)
// ============================================================================

/// RSI with ATR Confirmation Strategy
///
/// Combines RSI momentum signals with ATR volatility filter.
/// Only takes RSI signals when ATR indicates sufficient volatility.
///
/// # Default Parameters
/// - RSI Period: 14
/// - ATR Period: 14
/// - RSI Oversold: 30.0
/// - RSI Overbought: 70.0
/// - Min ATR %: 0.5% (volatility filter)
///
/// # Optimization Ranges
/// - RSI Period: 10-20
/// - ATR Period: 10-20
/// - RSI Oversold: 25-35
/// - RSI Overbought: 65-75
/// - Min ATR: 0.3-1.0%
///
/// # Market Conditions
/// - Best: Volatile markets with clear momentum reversals
/// - Avoid: Low volatility, low momentum markets
///
/// # Risk Management
/// - Stop Loss: 1.5× ATR below entry
/// - Take Profit: 3× ATR above entry
#[derive(Debug, Clone)]
pub struct RSIWithATR {
    pub rsi_period: usize,
    pub atr_period: usize,
    pub rsi_oversold: f64,
    pub rsi_overbought: f64,
    pub min_atr_pct: f64,
    pub stop_loss_atr_mult: f64,
    pub take_profit_atr_mult: f64,
    initial_capital: f64,
}

impl RSIWithATR {
    pub fn new(
        rsi_period: usize,
        atr_period: usize,
        rsi_oversold: f64,
        rsi_overbought: f64,
        min_atr_pct: f64,
    ) -> Self {
        Self {
            rsi_period,
            atr_period,
            rsi_oversold,
            rsi_overbought,
            min_atr_pct,
            stop_loss_atr_mult: 1.5,
            take_profit_atr_mult: 3.0,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for RSIWithATR {
    fn default() -> Self {
        Self::new(14, 14, 30.0, 70.0, 0.005)
    }
}

impl Strategy for RSIWithATR {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi_key = format!("rsi_{}", self.rsi_period);
        let atr_key = format!("atr_{}", self.atr_period);

        let rsi = indicators.get(&rsi_key).copied().unwrap_or(50.0);
        let atr = indicators.get(&atr_key).copied().unwrap_or(0.0);

        let atr_pct = if bar.close > 0.0 {
            atr / bar.close
        } else {
            0.0
        };

        if atr_pct < self.min_atr_pct {
            return Signal::Hold;
        }

        if rsi < self.rsi_oversold {
            Signal::Buy
        } else if rsi > self.rsi_overbought {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::RSI { period: self.rsi_period },
            IndicatorConfig::ATR { period: self.atr_period },
        ]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "rsi_period",
            ParameterRange::Int { min: 10, max: 20, step: 2 },
        );
        grid.add_range(
            "atr_period",
            ParameterRange::Int { min: 10, max: 20, step: 2 },
        );
        grid.add_range(
            "rsi_oversold",
            ParameterRange::Float { min: 25.0, max: 35.0, step: 5.0 },
        );
        grid.add_range(
            "rsi_overbought",
            ParameterRange::Float { min: 65.0, max: 75.0, step: 5.0 },
        );
        grid.add_range(
            "min_atr_pct",
            ParameterRange::Float { min: 0.003, max: 0.010, step: 0.001 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// MACD + EMA Trend Confirmation Strategy
// ============================================================================

/// MACD with EMA Trend Confirmation Strategy
///
/// Uses MACD for entry signals, but only when price is above/below long-term EMA.
/// Ensures trades align with the broader trend.
///
/// # Default Parameters
/// - MACD Fast: 12
/// - MACD Slow: 26
/// - MACD Signal: 9
/// - Trend EMA: 200 (long-term trend filter)
///
/// # Optimization Ranges
/// - MACD Fast: 8-16
/// - MACD Slow: 20-30
/// - MACD Signal: 7-12
/// - Trend EMA: 100-250
///
/// # Market Conditions
/// - Best: Trending markets with momentum swings
/// - Avoid: Choppy markets without clear trend
///
/// # Risk Management
/// - Stop Loss: 4% below entry
/// - Take Profit: 12% above entry
#[derive(Debug, Clone)]
pub struct MACDWithEMA {
    pub macd_fast: usize,
    pub macd_slow: usize,
    pub macd_signal: usize,
    pub trend_ema_period: usize,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
    prev_macd: f64,
    prev_signal: f64,
}

impl MACDWithEMA {
    pub fn new(
        macd_fast: usize,
        macd_slow: usize,
        macd_signal: usize,
        trend_ema_period: usize,
    ) -> Self {
        Self {
            macd_fast,
            macd_slow,
            macd_signal,
            trend_ema_period,
            stop_loss_pct: 0.04,
            take_profit_pct: 0.12,
            initial_capital: 10_000.0,
            prev_macd: 0.0,
            prev_signal: 0.0,
        }
    }
}

impl Default for MACDWithEMA {
    fn default() -> Self {
        Self::new(12, 26, 9, 200)
    }
}

impl Strategy for MACDWithEMA {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let macd_key = format!("macd_{}_{}_{}",  self.macd_fast, self.macd_slow, self.macd_signal);
        let macd_line_key = format!("{}_macd", macd_key);
        let signal_line_key = format!("{}_signal", macd_key);
        let ema_key = format!("ema_{}", self.trend_ema_period);

        let macd = indicators.get(&macd_line_key).copied().unwrap_or(0.0);
        let signal = indicators.get(&signal_line_key).copied().unwrap_or(0.0);
        let trend_ema = indicators.get(&ema_key).copied().unwrap_or(bar.close);

        let macd_cross_up = macd > signal && self.prev_macd <= self.prev_signal;
        let macd_cross_down = macd < signal && self.prev_macd >= self.prev_signal;

        self.prev_macd = macd;
        self.prev_signal = signal;

        if macd_cross_up && bar.close > trend_ema {
            Signal::Buy
        } else if macd_cross_down && bar.close < trend_ema {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::MACD {
                fast: self.macd_fast,
                slow: self.macd_slow,
                signal: self.macd_signal,
            },
            IndicatorConfig::EMA { period: self.trend_ema_period },
        ]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "macd_fast",
            ParameterRange::Int { min: 8, max: 16, step: 2 },
        );
        grid.add_range(
            "macd_slow",
            ParameterRange::Int { min: 20, max: 30, step: 2 },
        );
        grid.add_range(
            "trend_ema_period",
            ParameterRange::Int { min: 100, max: 250, step: 50 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// Bollinger Bands + Stochastic Reversal Strategy
// ============================================================================

/// Bollinger Bands with Stochastic Oscillator Strategy
///
/// Identifies high-probability reversals using both price extremes (BB)
/// and momentum extremes (Stochastic).
///
/// # Default Parameters
/// - BB Period: 20
/// - BB Std Dev: 2.0
/// - Stochastic K: 14
/// - Stochastic D: 3
/// - Stochastic Oversold: 20.0
/// - Stochastic Overbought: 80.0
///
/// # Optimization Ranges
/// - BB Period: 15-30
/// - BB Std Dev: 1.5-2.5
/// - Stochastic K: 10-20
/// - Stochastic D: 3-7
///
/// # Market Conditions
/// - Best: Range-bound markets with clear reversals
/// - Avoid: Strong trending markets
///
/// # Risk Management
/// - Stop Loss: Opposite BB band
/// - Take Profit: Middle BB band
#[derive(Debug, Clone)]
pub struct BollingerWithStochastic {
    pub bb_period: usize,
    pub bb_std_dev: f64,
    pub stoch_k_period: usize,
    pub stoch_d_period: usize,
    pub stoch_oversold: f64,
    pub stoch_overbought: f64,
    initial_capital: f64,
}

impl BollingerWithStochastic {
    pub fn new(
        bb_period: usize,
        bb_std_dev: f64,
        stoch_k_period: usize,
        stoch_d_period: usize,
        stoch_oversold: f64,
        stoch_overbought: f64,
    ) -> Self {
        Self {
            bb_period,
            bb_std_dev,
            stoch_k_period,
            stoch_d_period,
            stoch_oversold,
            stoch_overbought,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for BollingerWithStochastic {
    fn default() -> Self {
        Self::new(20, 2.0, 14, 3, 20.0, 80.0)
    }
}

impl Strategy for BollingerWithStochastic {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let bb_key = format!("bb_{}_{}", self.bb_period, self.bb_std_dev);
        let bb_upper_key = format!("{}_upper", bb_key);
        let bb_lower_key = format!("{}_lower", bb_key);

        let stoch_key = format!("stoch_{}_{}", self.stoch_k_period, self.stoch_d_period);
        let stoch_k_key = format!("{}_k", stoch_key);

        let bb_upper = indicators.get(&bb_upper_key).copied().unwrap_or(bar.high);
        let bb_lower = indicators.get(&bb_lower_key).copied().unwrap_or(bar.low);
        let stoch_k = indicators.get(&stoch_k_key).copied().unwrap_or(50.0);

        if bar.close <= bb_lower && stoch_k < self.stoch_oversold {
            Signal::Buy
        } else if bar.close >= bb_upper && stoch_k > self.stoch_overbought {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::BollingerBands {
                period: self.bb_period,
                std_dev: self.bb_std_dev,
            },
            IndicatorConfig::Stochastic {
                k_period: self.stoch_k_period,
                d_period: self.stoch_d_period,
            },
        ]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "bb_period",
            ParameterRange::Int { min: 15, max: 30, step: 5 },
        );
        grid.add_range(
            "bb_std_dev",
            ParameterRange::Float { min: 1.5, max: 2.5, step: 0.25 },
        );
        grid.add_range(
            "stoch_k_period",
            ParameterRange::Int { min: 10, max: 20, step: 2 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// Triple Confirmation Strategy (RSI + MACD + EMA)
// ============================================================================

/// Triple Confirmation Strategy
///
/// Requires all three signals to align before entering:
/// 1. RSI momentum confirmation
/// 2. MACD trend confirmation
/// 3. EMA trend filter
///
/// # Default Parameters
/// - RSI Period: 14 (oversold <30, overbought >70)
/// - MACD: 12/26/9 (standard)
/// - EMA: 50 (intermediate trend)
///
/// # Optimization Ranges
/// - RSI Period: 10-20
/// - MACD Fast: 8-16
/// - EMA Period: 30-100
///
/// # Market Conditions
/// - Best: Strong trending markets with momentum
/// - Avoid: Choppy markets (few signals)
///
/// # Risk Management
/// - Stop Loss: 5% below entry
/// - Take Profit: 15% above entry
#[derive(Debug, Clone)]
pub struct TripleConfirmation {
    pub rsi_period: usize,
    pub macd_fast: usize,
    pub macd_slow: usize,
    pub macd_signal: usize,
    pub ema_period: usize,
    pub rsi_oversold: f64,
    pub rsi_overbought: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    initial_capital: f64,
    prev_macd: f64,
    prev_signal: f64,
}

impl TripleConfirmation {
    pub fn new(rsi_period: usize, macd_fast: usize, macd_slow: usize, macd_signal: usize, ema_period: usize) -> Self {
        Self {
            rsi_period,
            macd_fast,
            macd_slow,
            macd_signal,
            ema_period,
            rsi_oversold: 30.0,
            rsi_overbought: 70.0,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.15,
            initial_capital: 10_000.0,
            prev_macd: 0.0,
            prev_signal: 0.0,
        }
    }
}

impl Default for TripleConfirmation {
    fn default() -> Self {
        Self::new(14, 12, 26, 9, 50)
    }
}

impl Strategy for TripleConfirmation {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi_key = format!("rsi_{}", self.rsi_period);
        let macd_key = format!("macd_{}_{}_{}",  self.macd_fast, self.macd_slow, self.macd_signal);
        let macd_line_key = format!("{}_macd", macd_key);
        let signal_line_key = format!("{}_signal", macd_key);
        let ema_key = format!("ema_{}", self.ema_period);

        let rsi = indicators.get(&rsi_key).copied().unwrap_or(50.0);
        let macd = indicators.get(&macd_line_key).copied().unwrap_or(0.0);
        let signal = indicators.get(&signal_line_key).copied().unwrap_or(0.0);
        let ema = indicators.get(&ema_key).copied().unwrap_or(bar.close);

        let macd_cross_up = macd > signal && self.prev_macd <= self.prev_signal;
        let macd_cross_down = macd < signal && self.prev_macd >= self.prev_signal;

        self.prev_macd = macd;
        self.prev_signal = signal;

        if rsi < self.rsi_oversold && macd_cross_up && bar.close > ema {
            Signal::Buy
        } else if rsi > self.rsi_overbought && macd_cross_down && bar.close < ema {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::RSI { period: self.rsi_period },
            IndicatorConfig::MACD {
                fast: self.macd_fast,
                slow: self.macd_slow,
                signal: self.macd_signal,
            },
            IndicatorConfig::EMA { period: self.ema_period },
        ]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "rsi_period",
            ParameterRange::Int { min: 10, max: 20, step: 2 },
        );
        grid.add_range(
            "macd_fast",
            ParameterRange::Int { min: 8, max: 16, step: 2 },
        );
        grid.add_range(
            "ema_period",
            ParameterRange::Int { min: 30, max: 100, step: 10 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// Volatility + Momentum Strategy (ATR + ROC)
// ============================================================================

/// Volatility-Momentum Combined Strategy
///
/// Combines volatility expansion (ATR) with momentum acceleration (ROC).
/// Only trades when both volatility and momentum are elevated.
///
/// # Default Parameters
/// - ATR Period: 14
/// - ROC Period: 12
/// - Min ATR %: 0.5%
/// - ROC Buy Threshold: 2.0%
/// - ROC Sell Threshold: -2.0%
///
/// # Optimization Ranges
/// - ATR Period: 10-20
/// - ROC Period: 8-20
/// - Min ATR: 0.3-1.0%
/// - ROC Threshold: 1.0-5.0%
///
/// # Market Conditions
/// - Best: Volatile breakout markets with strong momentum
/// - Avoid: Low volatility, low momentum consolidation
///
/// # Risk Management
/// - Stop Loss: 2× ATR below entry
/// - Take Profit: 4× ATR above entry
#[derive(Debug, Clone)]
pub struct VolatilityMomentum {
    pub atr_period: usize,
    pub roc_period: usize,
    pub min_atr_pct: f64,
    pub roc_buy_threshold: f64,
    pub roc_sell_threshold: f64,
    pub stop_loss_atr_mult: f64,
    pub take_profit_atr_mult: f64,
    initial_capital: f64,
}

impl VolatilityMomentum {
    pub fn new(
        atr_period: usize,
        roc_period: usize,
        min_atr_pct: f64,
        roc_buy_threshold: f64,
        roc_sell_threshold: f64,
    ) -> Self {
        Self {
            atr_period,
            roc_period,
            min_atr_pct,
            roc_buy_threshold,
            roc_sell_threshold,
            stop_loss_atr_mult: 2.0,
            take_profit_atr_mult: 4.0,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for VolatilityMomentum {
    fn default() -> Self {
        Self::new(14, 12, 0.005, 2.0, -2.0)
    }
}

impl Strategy for VolatilityMomentum {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let atr_key = format!("atr_{}", self.atr_period);
        let roc_key = format!("roc_{}", self.roc_period);

        let atr = indicators.get(&atr_key).copied().unwrap_or(0.0);
        let roc = indicators.get(&roc_key).copied().unwrap_or(0.0);

        let atr_pct = if bar.close > 0.0 {
            atr / bar.close
        } else {
            0.0
        };

        if atr_pct < self.min_atr_pct {
            return Signal::Hold;
        }

        if roc > self.roc_buy_threshold {
            Signal::Buy
        } else if roc < self.roc_sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::ATR { period: self.atr_period },
            IndicatorConfig::ROC { period: self.roc_period },
        ]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "atr_period",
            ParameterRange::Int { min: 10, max: 20, step: 2 },
        );
        grid.add_range(
            "roc_period",
            ParameterRange::Int { min: 8, max: 20, step: 2 },
        );
        grid.add_range(
            "min_atr_pct",
            ParameterRange::Float { min: 0.003, max: 0.010, step: 0.001 },
        );
        grid.add_range(
            "roc_buy_threshold",
            ParameterRange::Float { min: 1.0, max: 5.0, step: 1.0 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}
