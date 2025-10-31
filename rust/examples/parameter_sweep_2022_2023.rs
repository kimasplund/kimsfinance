//! Parameter Sweep Optimizer - Clean 2021-2022 Data
//!
//! Focused on post-AAPL-split data (Sept 2020+) with better quality.
//! Tests 576 parameter combinations to find optimal settings.
//!
//! Run with:
//! ```bash
//! cargo run --release --features data-downloaders --example parameter_sweep_2022_2023
//! ```

use chrono::NaiveDate;
use kimsfinance_core::strategy::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParameterCombination {
    dte_min: i32,
    dte_max: i32,
    delta_min: f64,
    delta_max: f64,
    profit_target_pct: f64,
    stop_loss_pct: f64,
    max_hold_days: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizationResult {
    params: ParameterCombination,
    symbol: String,
    num_trades: usize,
    total_pnl: f64,
    win_rate: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    return_on_capital: f64,
    profit_factor: f64,
    avg_days_in_trade: f64,
    fitness_score: f64,
}

fn calculate_fitness(result: &BacktestResult) -> f64 {
    // Fitness formula:
    // (ROC * 0.3) + (Sharpe * 25) + (WinRate * 0.2) - (MaxDD% * 0.5)

    let roc = result.return_on_capital;
    let sharpe = result.sharpe_ratio;
    let win_rate = result.win_rate;
    let max_dd_pct = result.max_drawdown / 10_000.0 * 100.0; // As percentage of initial capital

    (roc * 0.3) + (sharpe * 25.0) + (win_rate * 0.2) - (max_dd_pct * 0.5)
}

fn create_param_combinations() -> Vec<ParameterCombination> {
    let mut combinations = Vec::new();

    // DTE ranges (3 options)
    let dte_ranges = vec![(21, 35), (30, 45), (35, 50)];

    // Delta ranges (4 options)
    let delta_ranges = vec![
        (0.10, 0.25),
        (0.15, 0.30),
        (0.20, 0.35),
        (0.25, 0.40),
    ];

    // Profit targets (4 options)
    let profit_targets = vec![40.0, 50.0, 60.0, 75.0];

    // Stop losses (3 options)
    let stop_losses = vec![150.0, 200.0, 250.0];

    // Max hold days (4 options)
    let max_hold_days = vec![21, 28, 35, 42];

    // Generate all combinations: 3 * 4 * 4 * 3 * 4 = 576
    for (dte_min, dte_max) in dte_ranges {
        for (delta_min, delta_max) in delta_ranges.clone() {
            for profit_target in profit_targets.iter() {
                for stop_loss in stop_losses.iter() {
                    for max_hold in max_hold_days.iter() {
                        combinations.push(ParameterCombination {
                            dte_min,
                            dte_max,
                            delta_min,
                            delta_max,
                            profit_target_pct: *profit_target,
                            stop_loss_pct: *stop_loss,
                            max_hold_days: *max_hold,
                        });
                    }
                }
            }
        }
    }

    combinations
}

fn run_backtest_for_params(
    param_combo: &ParameterCombination,
    symbol: &str,
    data_dir: &str,
    spot_dir: &str,
    start_date: NaiveDate,
    end_date: NaiveDate,
) -> Option<OptimizationResult> {
    // Create strategy params
    let mut params = StrategyParams {
        name: format!("BPS_{}", symbol),
        dte_min: param_combo.dte_min,
        dte_max: param_combo.dte_max,
        delta_min: param_combo.delta_min,
        delta_max: param_combo.delta_max,
        profit_target_pct: Some(param_combo.profit_target_pct),
        stop_loss_pct: Some(param_combo.stop_loss_pct),
        max_hold_days: Some(param_combo.max_hold_days),
        position_size_pct: 5.0, // 5% per trade
        min_credit: Some(0.20),
        commission_per_contract: 0.65,
        slippage_ticks: 1.0,
        apply_bid_ask_spread: true,
        custom_params: std::collections::HashMap::new(),
    };

    // Create loaders
    let loader = match OptionsDataLoader::new(data_dir) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to create options loader: {:?}", e);
            return None;
        }
    };

    let spot_loader = match SpotDataLoader::new(spot_dir) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to create spot loader: {:?}", e);
            return None;
        }
    };

    // Create strategy and engine
    let strategy = BullPutSpread::new(params.clone());
    let mut engine = BacktestEngine::new(loader, spot_loader, 10_000.0);

    // Set risk limits
    engine.max_risk_per_trade_pct = 5.0;
    engine.max_concurrent_positions = 10;
    engine.max_margin_utilization_pct = 50.0;

    // Run backtest
    match engine.run_bull_put_spread(symbol, &strategy, &params, start_date, end_date) {
        Ok(result) => {
            // Only consider results with at least 10 trades
            if result.num_trades < 10 {
                return None;
            }

            let fitness = calculate_fitness(&result);

            Some(OptimizationResult {
                params: param_combo.clone(),
                symbol: symbol.to_string(),
                num_trades: result.num_trades,
                total_pnl: result.total_pnl,
                win_rate: result.win_rate,
                sharpe_ratio: result.sharpe_ratio,
                max_drawdown: result.max_drawdown,
                return_on_capital: result.return_on_capital,
                profit_factor: result.profit_factor,
                avg_days_in_trade: result.avg_days_in_trade,
                fitness_score: fitness,
            })
        }
        Err(e) => {
            // Don't spam errors for each parameter combination
            None
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bull Put Spread Parameter Optimization (2022-2023) ===\n");

    // Configuration - CLEAN DATA PERIOD ONLY
    let data_dir = "data/yfinance/options_historical";
    let spot_dir = "../data/yfinance/ohlcv";
    let symbols = vec!["AAPL", "SPY", "TSLA", "QQQ"];

    // Focus on 2021-2022 (post-AAPL split, clean available data)
    let start_date = NaiveDate::from_ymd_opt(2021, 1, 1).expect("Invalid start date");
    let end_date = NaiveDate::from_ymd_opt(2022, 12, 31).expect("Invalid end date");

    println!("Configuration:");
    println!("  Symbols: {:?}", symbols);
    println!("  Period: {} to {} (post-split clean data)", start_date, end_date);
    println!("  Testing 576 parameter combinations per symbol");
    println!("  Minimum trades: 10");
    println!();

    // Generate parameter combinations
    println!("Generating parameter combinations...");
    let param_combinations = create_param_combinations();
    println!("  Total combinations: {}", param_combinations.len());
    println!();

    // Create tasks for all symbol-parameter combinations
    let mut tasks = Vec::new();
    for symbol in &symbols {
        for param_combo in &param_combinations {
            tasks.push((symbol.to_string(), param_combo.clone()));
        }
    }

    println!("Total tasks: {} (576 params × {} symbols)", tasks.len(), symbols.len());
    println!("Running parallel backtests using {} threads...\n", rayon::current_num_threads());

    // Run all backtests in parallel
    let results: Vec<OptimizationResult> = tasks
        .par_iter()
        .filter_map(|(symbol, param_combo)| {
            run_backtest_for_params(
                param_combo,
                symbol,
                data_dir,
                spot_dir,
                start_date,
                end_date,
            )
        })
        .collect();

    println!("\n=== Optimization Complete ===");
    println!("Valid results: {} / {}", results.len(), tasks.len());
    println!();

    if results.is_empty() {
        eprintln!("ERROR: No valid results found!");
        eprintln!("This likely means:");
        eprintln!("  1. No options data available for 2022-2023 period");
        eprintln!("  2. All backtests failed to find sufficient trades (min 10)");
        eprintln!("  3. Data quality issues persist even in clean period");
        eprintln!();
        eprintln!("Please verify data availability:");
        eprintln!("  ls -lh data/yfinance/options_historical/");
        return Ok(());
    }

    // Sort by fitness score (best first)
    let mut sorted_results = results;
    sorted_results.sort_by(|a, b| {
        b.fitness_score
            .partial_cmp(&a.fitness_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Display top 20 results
    println!("=== Top 20 Parameter Combinations ===\n");
    for (i, result) in sorted_results.iter().take(20).enumerate() {
        println!("Rank #{} - {} (Fitness: {:.2})", i + 1, result.symbol, result.fitness_score);
        println!("  DTE: {}-{} days", result.params.dte_min, result.params.dte_max);
        println!(
            "  Delta: {:.2}-{:.2}",
            result.params.delta_min, result.params.delta_max
        );
        println!("  Profit Target: {:.0}%", result.params.profit_target_pct);
        println!("  Stop Loss: {:.0}%", result.params.stop_loss_pct);
        println!("  Max Hold: {} days", result.params.max_hold_days);
        println!("  ---");
        println!("  Trades: {}", result.num_trades);
        println!("  Total P&L: ${:.2}", result.total_pnl);
        println!("  ROC: {:.1}%", result.return_on_capital);
        println!("  Win Rate: {:.1}%", result.win_rate);
        println!("  Sharpe: {:.2}", result.sharpe_ratio);
        println!("  Profit Factor: {:.2}", result.profit_factor);
        println!("  Max Drawdown: ${:.2}", result.max_drawdown);
        println!("  Avg Days: {:.1}", result.avg_days_in_trade);
        println!();
    }

    // Save results to JSON
    let output_file = "data/optimization_results_2021_2022.json";
    let json = serde_json::to_string_pretty(&sorted_results)?;
    let mut file = File::create(output_file)?;
    file.write_all(json.as_bytes())?;
    println!("Results saved to: {}", output_file);

    // Summary statistics
    println!("\n=== Summary Statistics ===");
    println!("Total valid backtests: {}", sorted_results.len());

    let avg_roc: f64 = sorted_results.iter().map(|r| r.return_on_capital).sum::<f64>()
        / sorted_results.len() as f64;
    let avg_sharpe: f64 = sorted_results.iter().map(|r| r.sharpe_ratio).sum::<f64>()
        / sorted_results.len() as f64;
    let avg_win_rate: f64 = sorted_results.iter().map(|r| r.win_rate).sum::<f64>()
        / sorted_results.len() as f64;

    println!("Average ROC: {:.1}%", avg_roc);
    println!("Average Sharpe: {:.2}", avg_sharpe);
    println!("Average Win Rate: {:.1}%", avg_win_rate);

    let best = &sorted_results[0];
    println!("\nBest Parameters (Rank #1):");
    println!("  Symbol: {}", best.symbol);
    println!("  DTE: {}-{}", best.params.dte_min, best.params.dte_max);
    println!("  Delta: {:.2}-{:.2}", best.params.delta_min, best.params.delta_max);
    println!("  Profit Target: {:.0}%", best.params.profit_target_pct);
    println!("  Stop Loss: {:.0}%", best.params.stop_loss_pct);
    println!("  Max Hold: {}", best.params.max_hold_days);
    println!("  ROC: {:.1}%", best.return_on_capital);
    println!("  Sharpe: {:.2}", best.sharpe_ratio);
    println!("  Win Rate: {:.1}%", best.win_rate);

    // Profitability assessment
    println!("\n=== Profitability Assessment ===");
    if best.return_on_capital > 50.0 && best.sharpe_ratio > 1.5 && best.win_rate > 60.0 {
        println!("✅ HIGHLY PROFITABLE: Exceeds all targets!");
        println!("   - ROC > 50% (2-year period)");
        println!("   - Sharpe > 1.5 (excellent risk-adjusted returns)");
        println!("   - Win Rate > 60% (consistent profitability)");
    } else if best.return_on_capital > 30.0 && best.sharpe_ratio > 1.0 && best.win_rate > 55.0 {
        println!("✅ PROFITABLE: Meets profitability targets");
        println!("   - ROC > 30% (2-year period)");
        println!("   - Sharpe > 1.0 (good risk-adjusted returns)");
        println!("   - Win Rate > 55% (reliable profitability)");
    } else if best.return_on_capital > 15.0 && best.sharpe_ratio > 0.5 {
        println!("⚠️  MARGINALLY PROFITABLE: Needs optimization");
        println!("   - ROC > 15% but below 30% target");
        println!("   - Sharpe > 0.5 but below 1.0 target");
    } else {
        println!("❌ NOT PROFITABLE: Strategy needs significant revision");
        println!("   - ROC < 15% (insufficient returns)");
        println!("   - Consider different strategy or assets");
    }

    Ok(())
}
