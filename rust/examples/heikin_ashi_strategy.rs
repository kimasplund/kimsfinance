//! Heikin-Ashi Strategy Example
//!
//! Demonstrates a simple trend-following strategy comparing regular OHLC vs Heikin-Ashi candles.
//! Heikin-Ashi candles smooth price action and make trends more visible.
//!
//! # Strategy
//!
//! - Enter Long: Heikin-Ashi candle is bullish (close > open) AND no lower wick
//! - Exit Long: Heikin-Ashi candle is bearish (close < open)
//! - Enter Short: Heikin-Ashi candle is bearish AND no upper wick
//! - Exit Short: Heikin-Ashi candle is bullish
//!
//! # Usage
//!
//! ```bash
//! # Compile with GPU support
//! cargo build --release --features gpu --example heikin_ashi_strategy
//!
//! # Run with CSV data (OHLCV format)
//! ./target/release/examples/heikin_ashi_strategy ohlcv.csv
//!
//! # Or run with demo data
//! ./target/release/examples/heikin_ashi_strategy --demo
//! ```

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::candles::*;
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  Heikin-Ashi Trend Following Strategy");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Load OHLCV data
    let ohlcv = if std::env::args().any(|arg| arg == "--demo") {
        println!("1️⃣  Generating demo OHLCV data...");
        generate_demo_ohlcv()
    } else {
        println!("1️⃣  Loading OHLCV data from CSV...");
        let csv_path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "ohlcv.csv".to_string());
        load_ohlcv_csv(&csv_path)?
    };

    println!("   ✅ Loaded {} candles", ohlcv.len());
    println!();

    // Initialize GPU device
    println!("2️⃣  Initializing GPU device...");
    let device = GpuDevice::new()?;
    println!("   ✅ GPU initialized");
    println!();

    // Transform to Heikin-Ashi
    println!("3️⃣  Transforming to Heikin-Ashi candles...");
    let mut batch = HeikinAshiBatch::new();
    batch.add_task(ohlcv.clone(), ());

    let ha_candles = execute_batch(&device, &batch)?;
    let ha = &ha_candles[0];

    println!("   ✅ Generated {} Heikin-Ashi candles", ha.len());
    println!();

    // Display comparison: Regular vs Heikin-Ashi
    println!("4️⃣  Candle Comparison (First 5):");
    println!();
    println!("   Regular OHLC:");
    println!("   Index │   Open    │   High    │    Low    │  Close");
    println!("   ──────┼───────────┼───────────┼───────────┼──────────");
    for i in 0..5.min(ohlcv.len()) {
        let c = &ohlcv[i];
        println!("   {:>5} │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>9.2}",
            i, c.open, c.high, c.low, c.close
        );
    }

    println!();
    println!("   Heikin-Ashi (Smoothed):");
    println!("   Index │ HA-Open   │ HA-High   │  HA-Low   │ HA-Close");
    println!("   ──────┼───────────┼───────────┼───────────┼──────────");
    for i in 0..5.min(ha.len()) {
        let c = &ha[i];
        println!("   {:>5} │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>9.2}",
            i, c.open, c.high, c.low, c.close
        );
    }
    println!();

    // Run strategy on both regular and HA candles
    println!("5️⃣  Running Trend Following Strategy...");
    println!();

    let regular_signals = generate_signals(&ohlcv, false);
    let ha_signals = generate_signals(&ha, true);

    println!("   Strategy Results:");
    println!("   ┌────────────────────────────┬──────────┬──────────┐");
    println!("   │ Metric                     │ Regular  │   HA     │");
    println!("   ├────────────────────────────┼──────────┼──────────┤");

    let regular_trades = count_trades(&regular_signals);
    let ha_trades = count_trades(&ha_signals);
    println!("   │ Total Trades               │ {:>8} │ {:>8} │", regular_trades, ha_trades);

    let regular_profit = calculate_pnl(&ohlcv, &regular_signals);
    let ha_profit = calculate_pnl(&ha, &ha_signals);
    println!("   │ Total P&L                  │ {:>7.2}% │ {:>7.2}% │", regular_profit, ha_profit);

    let regular_winrate = calculate_winrate(&ohlcv, &regular_signals);
    let ha_winrate = calculate_winrate(&ha, &ha_signals);
    println!("   │ Win Rate                   │ {:>7.1}% │ {:>7.1}% │", regular_winrate, ha_winrate);

    let regular_max_dd = calculate_max_drawdown(&ohlcv, &regular_signals);
    let ha_max_dd = calculate_max_drawdown(&ha, &ha_signals);
    println!("   │ Max Drawdown               │ {:>7.2}% │ {:>7.2}% │", regular_max_dd, ha_max_dd);

    println!("   └────────────────────────────┴──────────┴──────────┘");
    println!();

    // Show recent signals
    println!("6️⃣  Recent Signals (Last 10):");
    println!();
    println!("   Index │  Type   │  Price    │ Candle Type");
    println!("   ──────┼─────────┼───────────┼────────────");

    let start = (ha_signals.len() - 10).max(0);
    for i in start..ha_signals.len() {
        match ha_signals[i] {
            Signal::Long => {
                let price = ha[i].close;
                let candle_type = if ha[i].close > ha[i].open { "Bullish" } else { "Bearish" };
                println!("   {:>5} │  LONG   │ {:>9.2} │ {}", i, price, candle_type);
            }
            Signal::Short => {
                let price = ha[i].close;
                let candle_type = if ha[i].close > ha[i].open { "Bullish" } else { "Bearish" };
                println!("   {:>5} │  SHORT  │ {:>9.2} │ {}", i, price, candle_type);
            }
            Signal::Exit => {
                let price = ha[i].close;
                println!("   {:>5} │  EXIT   │ {:>9.2} │", i, price);
            }
            Signal::None => {}
        }
    }
    println!();

    println!("✅ Strategy backtest complete!");
    println!();

    println!("📊 Key Insights:");
    println!();
    println!("   Heikin-Ashi Benefits:");
    println!("   • Smooths price action → clearer trends");
    println!("   • Reduces false signals → fewer whipsaws");
    println!("   • No wicks = strong trend confirmation");
    println!("   • Better for trend-following strategies");
    println!();
    println!("   When to Use Regular OHLC:");
    println!("   • Need exact entry/exit prices");
    println!("   • Scalping strategies (precise timing)");
    println!("   • Support/resistance levels (exact prices)");
    println!();
    println!("   When to Use Heikin-Ashi:");
    println!("   • Trend identification");
    println!("   • Swing trading");
    println!("   • Reducing noise in choppy markets");
    println!("   • Confirming breakouts");
    println!();

    println!("💡 Pro Tips:");
    println!("   • Combine HA with RSI/MACD for confirmation");
    println!("   • Use regular OHLC for precise stop-loss placement");
    println!("   • HA works best in trending markets");
    println!("   • Avoid HA in sideways/ranging markets");
    println!();

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Error: This example requires --features gpu");
    eprintln!();
    eprintln!("Build with:");
    eprintln!("  cargo build --release --features gpu --example heikin_ashi_strategy");
    std::process::exit(1);
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy)]
enum Signal {
    None,
    Long,
    Short,
    Exit,
}

#[cfg(feature = "gpu")]
fn generate_signals(candles: &[Candle], use_ha_rules: bool) -> Vec<Signal> {
    let mut signals = vec![Signal::None; candles.len()];
    let mut position: Option<bool> = None; // None = flat, Some(true) = long, Some(false) = short

    for i in 1..candles.len() {
        let c = &candles[i];
        let is_bullish = c.close > c.open;
        let is_bearish = c.close < c.open;

        if use_ha_rules {
            // Heikin-Ashi specific rules
            let no_lower_wick = (c.low - c.open.min(c.close)).abs() < 0.001;
            let no_upper_wick = (c.high - c.open.max(c.close)).abs() < 0.001;

            match position {
                None => {
                    // Enter long: bullish + no lower wick
                    if is_bullish && no_lower_wick {
                        signals[i] = Signal::Long;
                        position = Some(true);
                    }
                    // Enter short: bearish + no upper wick
                    else if is_bearish && no_upper_wick {
                        signals[i] = Signal::Short;
                        position = Some(false);
                    }
                }
                Some(true) => {
                    // Exit long if bearish
                    if is_bearish {
                        signals[i] = Signal::Exit;
                        position = None;
                    }
                }
                Some(false) => {
                    // Exit short if bullish
                    if is_bullish {
                        signals[i] = Signal::Exit;
                        position = None;
                    }
                }
            }
        } else {
            // Simple moving average crossover for regular OHLC
            let ma_short = simple_ma(candles, i, 5);
            let ma_long = simple_ma(candles, i, 20);

            if i < 20 {
                continue; // Not enough data
            }

            match position {
                None => {
                    if ma_short > ma_long && is_bullish {
                        signals[i] = Signal::Long;
                        position = Some(true);
                    } else if ma_short < ma_long && is_bearish {
                        signals[i] = Signal::Short;
                        position = Some(false);
                    }
                }
                Some(true) => {
                    if ma_short < ma_long {
                        signals[i] = Signal::Exit;
                        position = None;
                    }
                }
                Some(false) => {
                    if ma_short > ma_long {
                        signals[i] = Signal::Exit;
                        position = None;
                    }
                }
            }
        }
    }

    signals
}

#[cfg(feature = "gpu")]
fn simple_ma(candles: &[Candle], idx: usize, period: usize) -> f64 {
    if idx < period {
        return 0.0;
    }
    let sum: f64 = candles[idx - period + 1..=idx]
        .iter()
        .map(|c| c.close)
        .sum();
    sum / period as f64
}

#[cfg(feature = "gpu")]
fn count_trades(signals: &[Signal]) -> usize {
    signals.iter().filter(|s| matches!(s, Signal::Long | Signal::Short)).count()
}

#[cfg(feature = "gpu")]
fn calculate_pnl(candles: &[Candle], signals: &[Signal]) -> f64 {
    let mut total_pnl = 0.0;
    let mut entry_price = 0.0;
    let mut is_long = false;

    for i in 0..signals.len() {
        match signals[i] {
            Signal::Long => {
                entry_price = candles[i].close;
                is_long = true;
            }
            Signal::Short => {
                entry_price = candles[i].close;
                is_long = false;
            }
            Signal::Exit => {
                let exit_price = candles[i].close;
                let pnl = if is_long {
                    (exit_price - entry_price) / entry_price
                } else {
                    (entry_price - exit_price) / entry_price
                };
                total_pnl += pnl;
            }
            Signal::None => {}
        }
    }

    total_pnl * 100.0 // Convert to percentage
}

#[cfg(feature = "gpu")]
fn calculate_winrate(candles: &[Candle], signals: &[Signal]) -> f64 {
    let mut wins = 0;
    let mut total_trades = 0;
    let mut entry_price = 0.0;
    let mut is_long = false;

    for i in 0..signals.len() {
        match signals[i] {
            Signal::Long => {
                entry_price = candles[i].close;
                is_long = true;
            }
            Signal::Short => {
                entry_price = candles[i].close;
                is_long = false;
            }
            Signal::Exit => {
                let exit_price = candles[i].close;
                let pnl = if is_long {
                    exit_price - entry_price
                } else {
                    entry_price - exit_price
                };
                total_trades += 1;
                if pnl > 0.0 {
                    wins += 1;
                }
            }
            Signal::None => {}
        }
    }

    if total_trades == 0 {
        0.0
    } else {
        (wins as f64 / total_trades as f64) * 100.0
    }
}

#[cfg(feature = "gpu")]
fn calculate_max_drawdown(candles: &[Candle], signals: &[Signal]) -> f64 {
    let mut equity = 10000.0;
    let mut peak = equity;
    let mut max_dd = 0.0;
    let mut entry_price = 0.0;
    let mut is_long = false;
    let mut position_size = 0.0;

    for i in 0..signals.len() {
        match signals[i] {
            Signal::Long | Signal::Short => {
                entry_price = candles[i].close;
                is_long = matches!(signals[i], Signal::Long);
                position_size = equity / entry_price;
            }
            Signal::Exit => {
                let exit_price = candles[i].close;
                let pnl = if is_long {
                    (exit_price - entry_price) * position_size
                } else {
                    (entry_price - exit_price) * position_size
                };
                equity += pnl;

                if equity > peak {
                    peak = equity;
                }

                let dd = (peak - equity) / peak * 100.0;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
            Signal::None => {}
        }
    }

    max_dd
}

#[cfg(feature = "gpu")]
fn generate_demo_ohlcv() -> Vec<Candle> {
    let mut candles = Vec::new();
    let mut price = 100.0;
    let start_time = 1609459200; // 2021-01-01 00:00:00

    for i in 0..500 {
        // Simulate realistic price action with trends
        let trend = (i as f64 / 50.0).sin() * 2.0;
        let volatility = (i as f64 * 0.1).cos() * 0.5;

        price += trend + volatility;

        let open = price;
        let high = price + (i as f64 * 0.234).sin().abs() * 2.0;
        let low = price - (i as f64 * 0.456).cos().abs() * 2.0;
        let close = price + volatility;
        let volume = 100.0 + (i as f64 * 0.123).sin().abs() * 50.0;

        candles.push(Candle {
            timestamp: start_time + (i * 60) as i64,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    candles
}

#[cfg(feature = "gpu")]
fn load_ohlcv_csv(path: &str) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut candles = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue; // Skip header
        }

        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 6 {
            continue;
        }

        let candle = Candle {
            timestamp: parts[0].parse()?,
            open: parts[1].parse()?,
            high: parts[2].parse()?,
            low: parts[3].parse()?,
            close: parts[4].parse()?,
            volume: parts[5].parse()?,
        };

        candles.push(candle);
    }

    Ok(candles)
}
