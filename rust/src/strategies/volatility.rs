//! Volatility-based trading strategies
//!
//! Strategies that capitalize on volatility expansion, contraction, and breakouts.

use crate::backtest::core::{
    IndicatorConfig, IndicatorValues, OHLCVBar, ParameterGrid, ParameterRange, Signal, Strategy,
};

// ============================================================================
// Bollinger Bands Squeeze Strategy
// ============================================================================

/// Bollinger Bands Squeeze Strategy
///
/// Identifies periods of low volatility (squeeze) followed by breakouts.
/// Buys when price breaks above upper band after squeeze.
/// Sells when price breaks below lower band after squeeze.
///
/// # Default Parameters
/// - Period: 20 (standard Bollinger)
/// - Std Dev: 2.0 (standard deviation multiplier)
/// - Squeeze Threshold: 0.05 (band width as % of middle)
///
/// # Optimization Ranges
/// - Period: 15-30
/// - Std Dev: 1.5-2.5
/// - Squeeze Threshold: 0.03-0.08
///
/// # Market Conditions
/// - Best: Markets alternating between consolidation and breakout
/// - Avoid: Continuously trending markets without consolidation
///
/// # Risk Management
/// - Stop Loss: Lower band (dynamic based on volatility)
/// - Take Profit: 2x bandwidth above entry
#[derive(Debug, Clone)]
pub struct BollingerBandsSqueeze {
    pub period: usize,
    pub std_dev: f64,
    pub squeeze_threshold: f64,
    pub take_profit_mult: f64,
    initial_capital: f64,
    in_squeeze: bool,
}

impl BollingerBandsSqueeze {
    pub fn new(period: usize, std_dev: f64, squeeze_threshold: f64) -> Self {
        Self {
            period,
            std_dev,
            squeeze_threshold,
            take_profit_mult: 2.0,
            initial_capital: 10_000.0,
            in_squeeze: false,
        }
    }
}

impl Default for BollingerBandsSqueeze {
    fn default() -> Self {
        Self::new(20, 2.0, 0.05)
    }
}

impl Strategy for BollingerBandsSqueeze {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("bb_{}_{}", self.period, self.std_dev);
        let upper_key = format!("{}_upper", key);
        let middle_key = format!("{}_middle", key);
        let lower_key = format!("{}_lower", key);

        let upper = indicators.get(&upper_key).copied().unwrap_or(bar.high);
        let middle = indicators.get(&middle_key).copied().unwrap_or(bar.close);
        let lower = indicators.get(&lower_key).copied().unwrap_or(bar.low);

        let bandwidth = if middle > 0.0 {
            (upper - lower) / middle
        } else {
            0.0
        };

        if bandwidth < self.squeeze_threshold {
            self.in_squeeze = true;
        }

        if self.in_squeeze && bar.close > upper {
            self.in_squeeze = false;
            return Signal::Buy;
        } else if self.in_squeeze && bar.close < lower {
            self.in_squeeze = false;
            return Signal::Sell;
        }

        Signal::Hold
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::BollingerBands {
            period: self.period,
            std_dev: self.std_dev,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "period",
            ParameterRange::Int { min: 15, max: 30, step: 5 },
        );
        grid.add_range(
            "std_dev",
            ParameterRange::Float { min: 1.5, max: 2.5, step: 0.25 },
        );
        grid.add_range(
            "squeeze_threshold",
            ParameterRange::Float { min: 0.03, max: 0.08, step: 0.01 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// Bollinger Bands Expansion Strategy
// ============================================================================

/// Bollinger Bands Expansion Strategy (Mean Reversion)
///
/// Fades extreme moves by buying at lower band and selling at upper band.
/// Assumes price will revert to the mean after extreme volatility.
///
/// # Default Parameters
/// - Period: 20 (standard Bollinger)
/// - Std Dev: 2.0 (standard deviation multiplier)
/// - Exit at Middle: true (exit when price returns to middle band)
///
/// # Optimization Ranges
/// - Period: 15-30
/// - Std Dev: 1.5-2.5
///
/// # Market Conditions
/// - Best: Range-bound markets with mean-reverting behavior
/// - Avoid: Strong trending markets (bands will keep expanding)
///
/// # Risk Management
/// - Stop Loss: 3% beyond band (price continues trend)
/// - Take Profit: Middle band (mean reversion)
#[derive(Debug, Clone)]
pub struct BollingerBandsExpansion {
    pub period: usize,
    pub std_dev: f64,
    pub exit_at_middle: bool,
    pub stop_loss_pct: f64,
    initial_capital: f64,
}

impl BollingerBandsExpansion {
    pub fn new(period: usize, std_dev: f64, exit_at_middle: bool) -> Self {
        Self {
            period,
            std_dev,
            exit_at_middle,
            stop_loss_pct: 0.03,
            initial_capital: 10_000.0,
        }
    }
}

impl Default for BollingerBandsExpansion {
    fn default() -> Self {
        Self::new(20, 2.0, true)
    }
}

impl Strategy for BollingerBandsExpansion {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("bb_{}_{}", self.period, self.std_dev);
        let upper_key = format!("{}_upper", key);
        let middle_key = format!("{}_middle", key);
        let lower_key = format!("{}_lower", key);

        let upper = indicators.get(&upper_key).copied().unwrap_or(bar.high);
        let middle = indicators.get(&middle_key).copied().unwrap_or(bar.close);
        let lower = indicators.get(&lower_key).copied().unwrap_or(bar.low);

        if bar.close <= lower {
            Signal::Buy
        } else if bar.close >= upper {
            Signal::Sell
        } else if self.exit_at_middle && (bar.close >= middle * 0.995 && bar.close <= middle * 1.005) {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::BollingerBands {
            period: self.period,
            std_dev: self.std_dev,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "period",
            ParameterRange::Int { min: 15, max: 30, step: 5 },
        );
        grid.add_range(
            "std_dev",
            ParameterRange::Float { min: 1.5, max: 2.5, step: 0.25 },
        );
        grid
    }

    fn initial_capital(&self) -> f64 {
        self.initial_capital
    }
}

// ============================================================================
// ATR Volatility Breakout Strategy
// ============================================================================

/// ATR Volatility Breakout Strategy
///
/// Uses Average True Range (ATR) to identify volatility expansion.
/// Buys when price moves more than N×ATR above previous close.
/// Sells when price moves more than N×ATR below previous close.
///
/// # Default Parameters
/// - ATR Period: 14 (standard ATR)
/// - Breakout Multiplier: 2.0 (2× ATR movement)
/// - Min ATR: 0.5% (avoid low volatility trades)
///
/// # Optimization Ranges
/// - ATR Period: 10-20
/// - Breakout Multiplier: 1.5-3.0
/// - Min ATR: 0.3-1.0%
///
/// # Market Conditions
/// - Best: Volatile markets with clear directional moves
/// - Avoid: Low volatility, choppy markets
///
/// # Risk Management
/// - Stop Loss: 1× ATR below entry
/// - Take Profit: 3× ATR above entry
#[derive(Debug, Clone)]
pub struct ATRVolatilityBreakout {
    pub atr_period: usize,
    pub breakout_multiplier: f64,
    pub min_atr_pct: f64,
    pub stop_loss_atr_mult: f64,
    pub take_profit_atr_mult: f64,
    initial_capital: f64,
    prev_close: f64,
}

impl ATRVolatilityBreakout {
    pub fn new(atr_period: usize, breakout_multiplier: f64, min_atr_pct: f64) -> Self {
        Self {
            atr_period,
            breakout_multiplier,
            min_atr_pct,
            stop_loss_atr_mult: 1.0,
            take_profit_atr_mult: 3.0,
            initial_capital: 10_000.0,
            prev_close: 0.0,
        }
    }
}

impl Default for ATRVolatilityBreakout {
    fn default() -> Self {
        Self::new(14, 2.0, 0.005)
    }
}

impl Strategy for ATRVolatilityBreakout {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let key = format!("atr_{}", self.atr_period);
        let atr = indicators.get(&key).copied().unwrap_or(0.0);

        if self.prev_close == 0.0 {
            self.prev_close = bar.close;
            return Signal::Hold;
        }

        let atr_pct = if self.prev_close > 0.0 {
            atr / self.prev_close
        } else {
            0.0
        };

        if atr_pct < self.min_atr_pct {
            self.prev_close = bar.close;
            return Signal::Hold;
        }

        let breakout_threshold = atr * self.breakout_multiplier;
        let price_move = bar.close - self.prev_close;

        let result = if price_move > breakout_threshold {
            Signal::Buy
        } else if price_move < -breakout_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        };

        self.prev_close = bar.close;

        result
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![IndicatorConfig::ATR {
            period: self.atr_period,
        }]
    }

    fn parameters(&self) -> ParameterGrid {
        let mut grid = ParameterGrid::new();
        grid.add_range(
            "atr_period",
            ParameterRange::Int { min: 10, max: 20, step: 2 },
        );
        grid.add_range(
            "breakout_multiplier",
            ParameterRange::Float { min: 1.5, max: 3.0, step: 0.5 },
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
