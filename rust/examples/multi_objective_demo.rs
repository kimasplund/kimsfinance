//! Multi-Objective Optimization Demo
//!
//! Demonstrates multi-objective optimization using NSGA-II to find Pareto-optimal
//! strategy parameters across multiple conflicting objectives.
//!
//! # Features Demonstrated
//!
//! - NSGA-II algorithm for multi-objective optimization
//! - Pareto frontier discovery
//! - Trade-offs between Sharpe ratio, Sortino ratio, and drawdown
//! - Balanced solution selection
//! - Objective-specific optimization
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example multi_objective_demo
//! ```

use kimsfinance_core::backtest::{
    BacktestEngine, IndicatorConfig, IndicatorValues, MultiObjectiveOptimizer, OHLCVBar, Objective,
    ParameterGrid, ParameterRange, Signal, Strategy,
};
use ndarray::Array1;

/// RSI + ATR combo strategy for demonstration
#[derive(Debug, Clone)]
struct ComboStrategy {
    rsi_period: usize,
    atr_period: usize,
    rsi_buy_threshold: f64,
    rsi_sell_threshold: f64,
}

impl Strategy for ComboStrategy {
    fn on_data(&mut self, bar: &OHLCVBar, indicators: &IndicatorValues) -> Signal {
        let rsi_key = format!("rsi_{}", self.rsi_period);
        let atr_key = format!("atr_{}", self.atr_period);

        let rsi = match indicators.get(&rsi_key) {
            Some(&value) if !value.is_nan() => value,
            _ => return Signal::Hold,
        };

        let atr = match indicators.get(&atr_key) {
            Some(&value) if !value.is_nan() => value,
            _ => return Signal::Hold,
        };

        // Use ATR for volatility filtering (only trade in reasonable volatility)
        let volatility_ratio = atr / bar.close;
        if volatility_ratio > 0.05 {
            // Too volatile, don't trade
            return Signal::Hold;
        }

        if rsi < self.rsi_buy_threshold {
            Signal::Buy
        } else if rsi > self.rsi_sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn indicators(&self) -> Vec<IndicatorConfig> {
        vec![
            IndicatorConfig::RSI {
                period: self.rsi_period,
            },
            IndicatorConfig::ATR {
                period: self.atr_period,
            },
        ]
    }

    fn initial_capital(&self) -> f64 {
        10_000.0
    }
}

/// Generate synthetic OHLCV data
fn generate_test_data(
    n: usize,
) -> (
    Vec<i64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    let mut price = 100.0;
    let start_time = 1_600_000_000i64;

    for i in 0..n {
        timestamps.push(start_time + (i as i64 * 3600));

        // Generate realistic price movement with trends and mean reversion
        let trend = (i as f64 / 100.0).sin() * 5.0;
        let noise = (i as f64 * 0.37).sin() * 2.0;
        let change = trend + noise;
        price += change;

        let volatility = 0.005 + (i as f64 * 0.1).sin().abs() * 0.01;
        let o = price;
        let h = price + (price * volatility);
        let l = price - (price * volatility);
        let c = price + (change * 0.5);

        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        volume.push(1000.0 + (i as f64 * 10.0));

        price = c;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Objective Optimization Demo ===\n");

    // Generate synthetic data
    let n = 180 * 24; // 6 months of hourly data
    let (timestamps, open, high, low, close, volume) = generate_test_data(n);

    println!("Generated {} bars of synthetic data", n);

    // Create backtest engine
    let engine = BacktestEngine::new();

    // Create strategy
    let mut strategy = ComboStrategy {
        rsi_period: 14,
        atr_period: 14,
        rsi_buy_threshold: 30.0,
        rsi_sell_threshold: 70.0,
    };

    // Define parameter grid
    let mut param_grid = ParameterGrid::new();
    param_grid.add_range(
        "rsi_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 5,
        },
    );
    param_grid.add_range(
        "atr_period",
        ParameterRange::Int {
            min: 10,
            max: 20,
            step: 5,
        },
    );
    param_grid.add_range(
        "rsi_buy_threshold",
        ParameterRange::Float {
            min: 25.0,
            max: 35.0,
            step: 5.0,
        },
    );
    param_grid.add_range(
        "rsi_sell_threshold",
        ParameterRange::Float {
            min: 65.0,
            max: 75.0,
            step: 5.0,
        },
    );

    println!("Parameter grid size: {} combinations\n", param_grid.size());

    // Configure multi-objective optimizer
    println!("Optimization Objectives:");
    println!("  1. Maximize Sharpe Ratio (risk-adjusted return)");
    println!("  2. Maximize Sortino Ratio (downside risk-adjusted return)");
    println!("  3. Minimize Maximum Drawdown");
    println!("  4. Maximize Win Rate\n");

    let optimizer = MultiObjectiveOptimizer::new()
        .add_objective(Objective::MaximizeSharpe)
        .add_objective(Objective::MaximizeSortino)
        .add_objective(Objective::MinimizeDrawdown)
        .add_objective(Objective::MaximizeWinRate)
        .population_size(50) // Smaller population for demo
        .generations(20) // Fewer generations for demo
        .mutation_rate(0.15)
        .crossover_rate(0.9);

    println!("Optimizer Configuration:");
    println!("  Population size: 50");
    println!("  Generations: 20");
    println!("  Mutation rate: 0.15");
    println!("  Crossover rate: 0.9\n");

    // Run multi-objective optimization
    println!("Running multi-objective optimization...\n");

    let result = optimizer.optimize(
        &engine,
        &mut strategy,
        &timestamps,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &param_grid,
    )?;

    // Display results
    println!("\n=== Optimization Results ===\n");
    println!(
        "Pareto frontier size: {} solutions",
        result.pareto_front.len()
    );
    println!("Total solutions explored: {}\n", result.all_solutions.len());

    // Display Pareto front
    println!("=== Pareto Frontier ===");
    println!("(Non-dominated solutions - each represents optimal trade-off)\n");

    for (i, solution) in result.pareto_front.iter().enumerate() {
        println!("Solution {}:", i + 1);
        println!("  Parameters:");
        for (key, value) in &solution.parameters {
            println!("    {}: {:.2}", key, value);
        }
        println!("  Objectives:");
        for (j, &value) in solution.objectives.iter().enumerate() {
            let obj_name = match j {
                0 => "Sharpe Ratio",
                1 => "Sortino Ratio",
                2 => "Max Drawdown (inverted)",
                3 => "Win Rate",
                _ => "Unknown",
            };
            // Display drawdown as positive value
            let display_value = if j == 2 { -value } else { value };
            println!("    {}: {:.3}", obj_name, display_value);
        }
        println!("  Performance:");
        println!(
            "    Total Return: {:.2}%",
            solution.backtest_result.total_return
        );
        println!("    Trades: {}", solution.backtest_result.num_trades);
        println!(
            "    Profit Factor: {:.2}",
            solution.backtest_result.profit_factor
        );
        println!();
    }

    // Find best solutions for each objective
    println!("=== Best Solutions by Objective ===\n");

    if let Some(best_sharpe) = result.best_for_objective(Objective::MaximizeSharpe) {
        println!("Best Sharpe Ratio: {:.3}", best_sharpe.objectives[0]);
        println!("  Parameters: {:?}", best_sharpe.parameters);
        println!("  Return: {:.2}%", best_sharpe.backtest_result.total_return);
        println!();
    }

    if let Some(best_sortino) = result.best_for_objective(Objective::MaximizeSortino) {
        println!("Best Sortino Ratio: {:.3}", best_sortino.objectives[1]);
        println!("  Parameters: {:?}", best_sortino.parameters);
        println!(
            "  Return: {:.2}%",
            best_sortino.backtest_result.total_return
        );
        println!();
    }

    if let Some(best_dd) = result.best_for_objective(Objective::MinimizeDrawdown) {
        println!("Minimum Drawdown: {:.2}%", -best_dd.objectives[2]);
        println!("  Parameters: {:?}", best_dd.parameters);
        println!("  Sharpe: {:.3}", best_dd.objectives[0]);
        println!();
    }

    if let Some(best_wr) = result.best_for_objective(Objective::MaximizeWinRate) {
        println!("Maximum Win Rate: {:.1}%", best_wr.objectives[3]);
        println!("  Parameters: {:?}", best_wr.parameters);
        println!("  Sharpe: {:.3}", best_wr.objectives[0]);
        println!();
    }

    // Balanced solution
    if let Some(balanced) = result.balanced_solution() {
        println!("=== Balanced Solution (Recommended) ===");
        println!("(Solution closest to median across all objectives)\n");
        println!("Parameters:");
        for (key, value) in &balanced.parameters {
            println!("  {}: {:.2}", key, value);
        }
        println!("\nObjectives:");
        println!("  Sharpe Ratio: {:.3}", balanced.objectives[0]);
        println!("  Sortino Ratio: {:.3}", balanced.objectives[1]);
        println!("  Max Drawdown: {:.2}%", -balanced.objectives[2]);
        println!("  Win Rate: {:.1}%", balanced.objectives[3]);
        println!("\nPerformance:");
        println!(
            "  Total Return: {:.2}%",
            balanced.backtest_result.total_return
        );
        println!("  Trades: {}", balanced.backtest_result.num_trades);
        println!(
            "  Profit Factor: {:.2}",
            balanced.backtest_result.profit_factor
        );
    }

    println!("\n=== Key Insights ===");
    println!("- The Pareto frontier shows the optimal trade-offs between objectives");
    println!("- No single solution is best in all objectives simultaneously");
    println!("- Choose solution based on your risk tolerance and objectives:");
    println!("  * High Sharpe: Best risk-adjusted returns");
    println!("  * High Sortino: Best downside protection");
    println!("  * Low Drawdown: Most conservative");
    println!("  * High Win Rate: Most consistent");
    println!("  * Balanced: Good overall compromise");

    Ok(())
}
