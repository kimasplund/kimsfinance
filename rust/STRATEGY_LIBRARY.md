# Trading Strategy Library

**Version**: 1.0.0
**Strategies**: 19 production-ready implementations
**Categories**: Momentum (7), Trend (4), Volatility (3), Composite (5)

## Overview

This library provides a comprehensive collection of battle-tested trading strategies organized by category. All strategies include default parameters based on industry best practices, optimization ranges, risk management, and expected market conditions.

## Quick Start

```rust
use kimsfinance_core::backtest::BacktestEngine;
use kimsfinance_core::strategies::momentum::RSIMeanReversion;

let mut strategy = RSIMeanReversion::default();
let engine = BacktestEngine::new();
let result = engine.run(&mut strategy, &timestamps, &open, &high, &low, &close, &volume)?;

println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
println!("Total Return: {:.2}%", result.total_return);
```

## Strategy Categories

### Momentum Strategies (7)

Strategies that capitalize on price momentum and oscillator signals.

#### 1. RSI Mean Reversion

**Description**: Buys when RSI is oversold (<30) and sells when RSI normalizes (>50).

**Default Parameters**:
- RSI Period: 14
- Buy Threshold: 30.0
- Sell Threshold: 50.0

**Optimization Ranges**:
- RSI Period: 10-20
- Buy Threshold: 20-35
- Sell Threshold: 50-65

**Market Conditions**:
- ✅ Best: Ranging/sideways markets with mean-reverting behavior
- ❌ Avoid: Strong trending markets

**Risk Management**:
- Stop Loss: 5% below entry
- Take Profit: 10% above entry

**Usage**:
```rust
let mut strategy = momentum::RSIMeanReversion::new(14, 30.0, 50.0);
```

---

#### 2. RSI Oversold/Overbought (Aggressive)

**Description**: Buys when RSI is deeply oversold (<20) and sells when overbought (>80).

**Default Parameters**:
- RSI Period: 14
- Oversold Threshold: 20.0
- Overbought Threshold: 80.0

**Optimization Ranges**:
- RSI Period: 9-21
- Oversold: 15-30
- Overbought: 70-85

**Market Conditions**:
- ✅ Best: Volatile markets with strong reversals
- ❌ Avoid: Low volatility, choppy markets

**Risk Management**:
- Stop Loss: 7% below entry
- Take Profit: 15% above entry

---

#### 3. MACD Trend Following

**Description**: Follows trends using MACD line and signal line crossovers.

**Default Parameters**:
- Fast EMA: 12
- Slow EMA: 26
- Signal Line: 9

**Optimization Ranges**:
- Fast EMA: 8-16
- Slow EMA: 20-30
- Signal Line: 7-12

**Market Conditions**:
- ✅ Best: Trending markets with clear directional moves
- ❌ Avoid: Choppy, range-bound markets

**Risk Management**:
- Stop Loss: 4% below entry
- Take Profit: 12% above entry

---

#### 4. MACD Divergence

**Description**: Uses MACD histogram divergence to identify reversals.

**Default Parameters**:
- Fast EMA: 12
- Slow EMA: 26
- Signal Line: 9

**Market Conditions**:
- ✅ Best: Markets showing momentum shifts
- ❌ Avoid: Low volatility sideways markets

---

#### 5. Stochastic Oscillator

**Description**: Uses %K and %D lines to identify overbought/oversold conditions.

**Default Parameters**:
- K Period: 14
- D Period: 3
- Oversold: 20.0
- Overbought: 80.0

**Optimization Ranges**:
- K Period: 10-20
- D Period: 3-7
- Oversold: 15-25
- Overbought: 75-85

**Market Conditions**:
- ✅ Best: Ranging markets with clear support/resistance
- ❌ Avoid: Strong trending markets

---

#### 6. ROC Breakout

**Description**: Identifies momentum breakouts using ROC acceleration.

**Default Parameters**:
- ROC Period: 12
- Buy Threshold: 2.0%
- Sell Threshold: -2.0%

**Optimization Ranges**:
- ROC Period: 8-20
- Buy Threshold: 1.0-5.0%
- Sell Threshold: -1.0 to -5.0%

**Market Conditions**:
- ✅ Best: Breakout markets with strong directional moves
- ❌ Avoid: Low volatility, consolidating markets

**Risk Management**:
- Stop Loss: 6% below entry
- Take Profit: 15% above entry

---

#### 7. CCI Reversal

**Description**: Uses CCI extreme values to identify reversal points.

**Default Parameters**:
- CCI Period: 20
- Oversold: -100.0
- Overbought: 100.0

**Optimization Ranges**:
- CCI Period: 14-30
- Oversold: -150 to -80
- Overbought: 80 to 150

**Market Conditions**:
- ✅ Best: Volatile markets with strong reversals
- ❌ Avoid: Low volatility trending markets

---

### Trend Strategies (4)

Strategies that identify and follow market trends.

#### 1. EMA Crossover (Golden Cross/Death Cross)

**Description**: Classic trend-following using fast and slow EMA crossovers.

**Default Parameters**:
- Fast EMA: 50
- Slow EMA: 200

**Optimization Ranges**:
- Fast EMA: 20-100
- Slow EMA: 100-250

**Market Conditions**:
- ✅ Best: Strong trending markets (bull or bear)
- ❌ Avoid: Choppy, sideways markets (whipsaws)

**Risk Management**:
- Stop Loss: 3% below entry
- Take Profit: 20% above entry (ride the trend)

---

#### 2. Triple EMA Trend

**Description**: Uses three EMAs to identify trend strength and direction.

**Default Parameters**:
- Short EMA: 8
- Medium EMA: 21
- Long EMA: 55

**Optimization Ranges**:
- Short EMA: 5-15
- Medium EMA: 15-30
- Long EMA: 40-100

**Market Conditions**:
- ✅ Best: Clear trending markets with momentum
- ❌ Avoid: Ranging markets with frequent crossovers

**Risk Management**:
- Stop Loss: 4% below entry
- Take Profit: 18% above entry

---

#### 3. Donchian Channel Breakout (Turtle Trading)

**Description**: Classic turtle trading using Donchian channel breakouts.

**Default Parameters**:
- Channel Period: 20

**Optimization Ranges**:
- Channel Period: 10-40

**Market Conditions**:
- ✅ Best: Markets breaking out of consolidation
- ❌ Avoid: Range-bound markets (false breakouts)

**Risk Management**:
- Stop Loss: 2% below entry
- Take Profit: 25% above entry

---

#### 4. Keltner Channel Trend

**Description**: Uses Keltner Channels (EMA + ATR) to identify trend strength.

**Default Parameters**:
- EMA Period: 20
- ATR Period: 10
- ATR Multiplier: 2.0

**Optimization Ranges**:
- EMA Period: 10-30
- ATR Period: 5-20
- ATR Multiplier: 1.5-3.0

**Market Conditions**:
- ✅ Best: Trending markets with expanding volatility
- ❌ Avoid: Low volatility consolidation

**Risk Management**:
- Stop Loss: 1.5× ATR below entry
- Take Profit: 15% above entry

---

### Volatility Strategies (3)

Strategies that capitalize on volatility expansion and contraction.

#### 1. Bollinger Bands Squeeze

**Description**: Identifies periods of low volatility (squeeze) followed by breakouts.

**Default Parameters**:
- Period: 20
- Std Dev: 2.0
- Squeeze Threshold: 0.05

**Optimization Ranges**:
- Period: 15-30
- Std Dev: 1.5-2.5
- Squeeze Threshold: 0.03-0.08

**Market Conditions**:
- ✅ Best: Markets alternating between consolidation and breakout
- ❌ Avoid: Continuously trending markets

**Risk Management**:
- Stop Loss: Lower band (dynamic)
- Take Profit: 2× bandwidth above entry

---

#### 2. Bollinger Bands Expansion (Mean Reversion)

**Description**: Fades extreme moves by buying at lower band, selling at upper band.

**Default Parameters**:
- Period: 20
- Std Dev: 2.0
- Exit at Middle: true

**Market Conditions**:
- ✅ Best: Range-bound markets with mean-reverting behavior
- ❌ Avoid: Strong trending markets

**Risk Management**:
- Stop Loss: 3% beyond band
- Take Profit: Middle band

---

#### 3. ATR Volatility Breakout

**Description**: Uses ATR to identify volatility expansion breakouts.

**Default Parameters**:
- ATR Period: 14
- Breakout Multiplier: 2.0
- Min ATR %: 0.5%

**Optimization Ranges**:
- ATR Period: 10-20
- Breakout Multiplier: 1.5-3.0
- Min ATR: 0.3-1.0%

**Market Conditions**:
- ✅ Best: Volatile markets with clear directional moves
- ❌ Avoid: Low volatility, choppy markets

**Risk Management**:
- Stop Loss: 1× ATR below entry
- Take Profit: 3× ATR above entry

---

### Composite Strategies (5)

Strategies combining multiple indicators for higher-confidence signals.

#### 1. RSI + ATR (Momentum + Volatility)

**Description**: Combines RSI momentum signals with ATR volatility filter.

**Default Parameters**:
- RSI Period: 14
- ATR Period: 14
- RSI Oversold: 30.0
- RSI Overbought: 70.0
- Min ATR %: 0.5%

**Market Conditions**:
- ✅ Best: Volatile markets with clear momentum reversals
- ❌ Avoid: Low volatility markets

**Risk Management**:
- Stop Loss: 1.5× ATR below entry
- Take Profit: 3× ATR above entry

---

#### 2. MACD + EMA Trend Confirmation

**Description**: Uses MACD for signals, EMA for trend filter.

**Default Parameters**:
- MACD: 12/26/9
- Trend EMA: 200

**Market Conditions**:
- ✅ Best: Trending markets with momentum swings
- ❌ Avoid: Choppy markets without clear trend

**Risk Management**:
- Stop Loss: 4% below entry
- Take Profit: 12% above entry

---

#### 3. Bollinger Bands + Stochastic

**Description**: Identifies reversals using both price extremes and momentum.

**Default Parameters**:
- BB Period: 20
- BB Std Dev: 2.0
- Stochastic K: 14
- Stochastic D: 3

**Market Conditions**:
- ✅ Best: Range-bound markets with clear reversals
- ❌ Avoid: Strong trending markets

**Risk Management**:
- Stop Loss: Opposite BB band
- Take Profit: Middle BB band

---

#### 4. Triple Confirmation (RSI + MACD + EMA)

**Description**: Requires all three signals to align before entering.

**Default Parameters**:
- RSI: 14
- MACD: 12/26/9
- EMA: 50

**Market Conditions**:
- ✅ Best: Strong trending markets with momentum
- ❌ Avoid: Choppy markets (few signals)

**Risk Management**:
- Stop Loss: 5% below entry
- Take Profit: 15% above entry

---

#### 5. Volatility + Momentum (ATR + ROC)

**Description**: Combines volatility expansion with momentum acceleration.

**Default Parameters**:
- ATR Period: 14
- ROC Period: 12
- Min ATR %: 0.5%
- ROC Buy Threshold: 2.0%

**Market Conditions**:
- ✅ Best: Volatile breakout markets with strong momentum
- ❌ Avoid: Low volatility consolidation

**Risk Management**:
- Stop Loss: 2× ATR below entry
- Take Profit: 4× ATR above entry

---

## Strategy Comparison

Run the strategy comparison example to evaluate all strategies on the same dataset:

```bash
cargo run --example strategy_comparison --release
```

This will output:
- Performance metrics for all 19 strategies
- Ranked by fitness score (Sharpe × drawdown penalty)
- Execution times
- Top 5 performers with detailed stats

## Optimization

All strategies include parameter grids for optimization:

```rust
use kimsfinance_core::backtest::BacktestEngine;

let mut strategy = momentum::RSIMeanReversion::default();
let grid = strategy.parameters();

println!("Total combinations: {}", grid.size());

// Run parameter sweep
let engine = BacktestEngine::new();
let results = engine.run_sweep(
    &mut strategy,
    &timestamps,
    &open,
    &high,
    &low,
    &close,
    &volume,
    &grid,
)?;

// Best result is first (sorted by fitness)
println!("Best Sharpe: {:.2}", results[0].sharpe_ratio);
```

## Testing

Run comprehensive tests:

```bash
# All strategy tests
cargo test test_strategies

# Specific category
cargo test test_all_momentum_strategies
cargo test test_all_trend_strategies
cargo test test_all_volatility_strategies
cargo test test_all_composite_strategies
```

## Performance Characteristics

### Momentum Strategies
- **Trade Frequency**: High (5-20 trades/100 bars)
- **Hold Time**: Short-term (1-5 bars)
- **Win Rate**: 45-60%
- **Best Markets**: Ranging, mean-reverting

### Trend Strategies
- **Trade Frequency**: Low (1-5 trades/100 bars)
- **Hold Time**: Long-term (10-50 bars)
- **Win Rate**: 40-50%
- **Best Markets**: Trending, directional

### Volatility Strategies
- **Trade Frequency**: Medium (3-10 trades/100 bars)
- **Hold Time**: Medium-term (3-15 bars)
- **Win Rate**: 50-65%
- **Best Markets**: Volatile, breakout

### Composite Strategies
- **Trade Frequency**: Low-Medium (2-8 trades/100 bars)
- **Hold Time**: Medium-term (5-20 bars)
- **Win Rate**: 55-70%
- **Best Markets**: Any (filtered for quality)

## Risk Management

All strategies include:
- **Stop Loss**: Percentage-based or ATR-based
- **Take Profit**: Fixed percentage or dynamic
- **Position Sizing**: 100% allocation by default (customizable via `position_size()`)
- **Trading Fees**: 0.1% per trade (configurable via `BacktestConfig`)
- **Slippage**: 0.05% per trade (configurable)

## Contributing

When adding new strategies:

1. Choose appropriate category (momentum/trend/volatility/composite)
2. Include comprehensive documentation:
   - Description
   - Default parameters
   - Optimization ranges
   - Market conditions (best/avoid)
   - Risk management
3. Add to module exports in `mod.rs`
4. Add tests to `test_strategies.rs`
5. Add to `strategy_comparison.rs` example

## References

Strategy implementations based on:
- Wilder, J. Wells. "New Concepts in Technical Trading Systems" (1978) - RSI, ATR
- Appel, Gerald. "Technical Analysis: Power Tools for Active Investors" (2005) - MACD
- Lane, George. "Lane's Stochastics" (1950s) - Stochastic Oscillator
- Bollinger, John. "Bollinger on Bollinger Bands" (2001) - Bollinger Bands
- Dennis, Richard. "Turtle Trading" (1980s) - Donchian Breakout
- Keltner, Chester. "How to Make Money in Commodities" (1960) - Keltner Channels

## License

Part of kimsfinance - High-performance GPU-accelerated trading library.
