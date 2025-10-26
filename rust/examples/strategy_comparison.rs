//! Strategy Comparison Example
//!
//! Compare all 20+ trading strategies on the same dataset to find the best performer.
//!
//! Run with:
//! ```bash
//! cargo run --example strategy_comparison --release
//! ```

use kimsfinance_core::backtest::{BacktestEngine, Strategy};
use kimsfinance_core::strategies::*;
use ndarray::Array1;
use std::time::Instant;

struct StrategyResult {
    name: String,
    sharpe_ratio: f64,
    total_return: f64,
    max_drawdown: f64,
    win_rate: f64,
    num_trades: usize,
    profit_factor: f64,
    execution_time_ms: u128,
}

impl StrategyResult {
    fn fitness_score(&self) -> f64 {
        let drawdown_penalty = 1.0 - (self.max_drawdown / 100.0).min(1.0);
        let trade_count_bonus = if self.num_trades < 5 {
            0.5
        } else if self.num_trades < 10 {
            0.75
        } else {
            1.0
        };

        self.sharpe_ratio * drawdown_penalty * trade_count_bonus
    }
}

fn generate_realistic_data(n: usize, trend: f64, volatility: f64) -> (Vec<i64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let mut price = 100.0;
    for i in 0..n {
        timestamps.push((i * 300) as i64);

        let random_walk = rng.gen_range(-volatility..volatility);
        let cyclical = ((i as f64 * 0.05).sin() * volatility * 0.5);
        price += trend + random_walk + cyclical;
        price = price.max(10.0);

        let o = price + rng.gen_range(-0.5..0.5);
        let c = price + rng.gen_range(-0.5..0.5);
        let h = o.max(c) + rng.gen_range(0.0..1.0);
        let l = o.min(c) - rng.gen_range(0.0..1.0);

        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        volume.push(rng.gen_range(1000.0..10000.0));
    }

    (
        timestamps,
        Array1::from_vec(open),
        Array1::from_vec(high),
        Array1::from_vec(low),
        Array1::from_vec(close),
        Array1::from_vec(volume),
    )
}

fn run_strategy_comparison(
    name: &str,
    strategy: &mut dyn Strategy,
    timestamps: &[i64],
    open: &Array1<f64>,
    high: &Array1<f64>,
    low: &Array1<f64>,
    close: &Array1<f64>,
    volume: &Array1<f64>,
) -> StrategyResult {
    let engine = BacktestEngine::new();
    let start = Instant::now();

    let result = engine.run(strategy, timestamps, open, high, low, close, volume);

    let execution_time_ms = start.elapsed().as_millis();

    match result {
        Ok(res) => StrategyResult {
            name: name.to_string(),
            sharpe_ratio: res.sharpe_ratio,
            total_return: res.total_return,
            max_drawdown: res.max_drawdown,
            win_rate: res.win_rate,
            num_trades: res.num_trades,
            profit_factor: res.profit_factor,
            execution_time_ms,
        },
        Err(e) => {
            eprintln!("Strategy '{}' failed: {}", name, e);
            StrategyResult {
                name: name.to_string(),
                sharpe_ratio: 0.0,
                total_return: 0.0,
                max_drawdown: 100.0,
                win_rate: 0.0,
                num_trades: 0,
                profit_factor: 0.0,
                execution_time_ms,
            }
        }
    }
}

fn main() {
    println!("=================================================================");
    println!("TRADING STRATEGY COMPARISON");
    println!("=================================================================");
    println!();

    println!("Generating realistic market data (1000 bars)...");
    let (timestamps, open, high, low, close, volume) = generate_realistic_data(1000, 0.05, 1.0);
    let initial_price = close[0];
    let final_price = close[close.len() - 1];
    let buy_hold_return = ((final_price - initial_price) / initial_price) * 100.0;

    println!("Dataset: {} bars", timestamps.len());
    println!("Initial Price: ${:.2}", initial_price);
    println!("Final Price: ${:.2}", final_price);
    println!("Buy & Hold Return: {:.2}%", buy_hold_return);
    println!();

    let mut results = Vec::new();

    println!("=================================================================");
    println!("MOMENTUM STRATEGIES (7 strategies)");
    println!("=================================================================");
    println!();

    let momentum_strategies: Vec<(String, Box<dyn Strategy>)> = vec![
        ("RSI Mean Reversion".to_string(), Box::new(momentum::RSIMeanReversion::default())),
        ("RSI Oversold/Overbought".to_string(), Box::new(momentum::RSIOversoldOverbought::default())),
        ("MACD Trend Following".to_string(), Box::new(momentum::MACDTrendFollowing::default())),
        ("MACD Divergence".to_string(), Box::new(momentum::MACDDivergence::default())),
        ("Stochastic Oscillator".to_string(), Box::new(momentum::StochasticOscillator::default())),
        ("ROC Breakout".to_string(), Box::new(momentum::ROCBreakout::default())),
        ("CCI Reversal".to_string(), Box::new(momentum::CCIReversal::default())),
    ];

    for (name, mut strategy) in momentum_strategies {
        print!("Testing: {:<30} ... ", name);
        let result = run_strategy_comparison(&name, strategy.as_mut(), &timestamps, &open, &high, &low, &close, &volume);
        println!("Sharpe: {:.2} | Return: {:.2}% | Trades: {}", result.sharpe_ratio, result.total_return, result.num_trades);
        results.push(result);
    }

    println!();
    println!("=================================================================");
    println!("TREND STRATEGIES (4 strategies)");
    println!("=================================================================");
    println!();

    let trend_strategies: Vec<(String, Box<dyn Strategy>)> = vec![
        ("EMA Crossover (50/200)".to_string(), Box::new(trend::EMACrossover::default())),
        ("Triple EMA Trend".to_string(), Box::new(trend::TripleEMATrend::default())),
        ("Donchian Breakout".to_string(), Box::new(trend::DonchianBreakout::default())),
        ("Keltner Trend".to_string(), Box::new(trend::KeltnerTrend::default())),
    ];

    for (name, mut strategy) in trend_strategies {
        print!("Testing: {:<30} ... ", name);
        let result = run_strategy_comparison(&name, strategy.as_mut(), &timestamps, &open, &high, &low, &close, &volume);
        println!("Sharpe: {:.2} | Return: {:.2}% | Trades: {}", result.sharpe_ratio, result.total_return, result.num_trades);
        results.push(result);
    }

    println!();
    println!("=================================================================");
    println!("VOLATILITY STRATEGIES (3 strategies)");
    println!("=================================================================");
    println!();

    let volatility_strategies: Vec<(String, Box<dyn Strategy>)> = vec![
        ("Bollinger Squeeze".to_string(), Box::new(volatility::BollingerBandsSqueeze::default())),
        ("Bollinger Expansion".to_string(), Box::new(volatility::BollingerBandsExpansion::default())),
        ("ATR Volatility Breakout".to_string(), Box::new(volatility::ATRVolatilityBreakout::default())),
    ];

    for (name, mut strategy) in volatility_strategies {
        print!("Testing: {:<30} ... ", name);
        let result = run_strategy_comparison(&name, strategy.as_mut(), &timestamps, &open, &high, &low, &close, &volume);
        println!("Sharpe: {:.2} | Return: {:.2}% | Trades: {}", result.sharpe_ratio, result.total_return, result.num_trades);
        results.push(result);
    }

    println!();
    println!("=================================================================");
    println!("COMPOSITE STRATEGIES (5 strategies)");
    println!("=================================================================");
    println!();

    let composite_strategies: Vec<(String, Box<dyn Strategy>)> = vec![
        ("RSI + ATR".to_string(), Box::new(composite::RSIWithATR::default())),
        ("MACD + EMA".to_string(), Box::new(composite::MACDWithEMA::default())),
        ("Bollinger + Stochastic".to_string(), Box::new(composite::BollingerWithStochastic::default())),
        ("Triple Confirmation".to_string(), Box::new(composite::TripleConfirmation::default())),
        ("Volatility + Momentum".to_string(), Box::new(composite::VolatilityMomentum::default())),
    ];

    for (name, mut strategy) in composite_strategies {
        print!("Testing: {:<30} ... ", name);
        let result = run_strategy_comparison(&name, strategy.as_mut(), &timestamps, &open, &high, &low, &close, &volume);
        println!("Sharpe: {:.2} | Return: {:.2}% | Trades: {}", result.sharpe_ratio, result.total_return, result.num_trades);
        results.push(result);
    }

    println!();
    println!("=================================================================");
    println!("STRATEGY RANKINGS");
    println!("=================================================================");
    println!();

    results.sort_by(|a, b| b.fitness_score().partial_cmp(&a.fitness_score()).unwrap());

    println!("{:<35} {:>10} {:>12} {:>12} {:>10} {:>8} {:>10} {:>10}",
        "Strategy", "Fitness", "Sharpe", "Return %", "Drawdown", "Trades", "Win Rate", "Time (ms)");
    println!("{}", "-".repeat(125));

    for (i, result) in results.iter().enumerate() {
        println!("{:<2}. {:<32} {:>10.2} {:>12.2} {:>11.2}% {:>9.2}% {:>8} {:>9.2}% {:>10}",
            i + 1,
            result.name,
            result.fitness_score(),
            result.sharpe_ratio,
            result.total_return,
            result.max_drawdown,
            result.num_trades,
            result.win_rate,
            result.execution_time_ms,
        );
    }

    println!();
    println!("=================================================================");
    println!("TOP 5 STRATEGIES");
    println!("=================================================================");
    println!();

    for (i, result) in results.iter().take(5).enumerate() {
        println!("{}. {}", i + 1, result.name);
        println!("   Sharpe Ratio: {:.2}", result.sharpe_ratio);
        println!("   Total Return: {:.2}%", result.total_return);
        println!("   Max Drawdown: {:.2}%", result.max_drawdown);
        println!("   Win Rate: {:.2}%", result.win_rate);
        println!("   Trades: {}", result.num_trades);
        println!("   Profit Factor: {:.2}", result.profit_factor);
        println!("   Execution Time: {}ms", result.execution_time_ms);
        println!();
    }

    println!("=================================================================");
    println!("Comparison complete! Tested {} strategies.", results.len());
    println!("=================================================================");
}
