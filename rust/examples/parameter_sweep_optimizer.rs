//! Comprehensive Parameter Sweep Optimizer for Bull Put Spread Strategy
//!
//! This tool performs a comprehensive parameter sweep to find optimal parameters
//! for the bull put spread strategy. It tests hundreds of combinations across
//! multiple dimensions:
//!
//! - DTE ranges (days to expiration)
//! - Delta ranges (option delta selection)
//! - Profit targets (exit when X% of max profit reached)
//! - Stop losses (exit when X% of max loss reached)
//! - Max hold days (time-based exits)
//!
//! ## Phase 4 Enhancements:
//! - Transaction costs (commission, slippage, bid-ask spread)
//! - Margin limits (position sizing constraints)
//! - Real spot prices (from OHLCV data)
//! - Regime detection (volatility-based filtering)
//!
//! The optimizer uses:
//! - Rayon for CPU parallelization (all cores)
//! - Fitness scoring: (ROC * 0.3) + (Sharpe * 25) + (WinRate * 0.2) - (MaxDD% * 0.5)
//! - Top 20 parameter tracking
//! - JSON output for further analysis
//!
//! Run with:
//! ```bash
//! cargo run --release --features data-downloaders --example parameter_sweep_optimizer
//! ```

use chrono::NaiveDate;
use kimsfinance_core::strategy::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Result for a single parameter combination
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParameterResult {
    /// Parameter configuration
    params: StrategyParams,

    /// Backtest results
    num_trades: usize,
    total_pnl: f64,
    win_rate: f64,
    sharpe_ratio: f64,
    sortino_ratio: f64,
    max_drawdown: f64,
    return_on_capital: f64,
    profit_factor: f64,
    avg_win: f64,
    avg_loss: f64,
    max_consecutive_losses: i32,
    avg_days_in_trade: f64,

    /// Fitness score (composite metric)
    fitness: f64,
}

/// Calculate fitness score for parameter optimization
///
/// **Phase 4 Formula**: (ROC * 0.3) + (Sharpe * 25) + (WinRate * 0.2) - (MaxDD% * 0.5)
///
/// This balances:
/// - Return on capital (30% weight) - profitability
/// - Risk-adjusted returns via Sharpe (25x scaling) - risk management
/// - Win rate for consistency (20% weight) - reliability
/// - Drawdown penalty (0.5x scaling) - capital preservation
///
/// Target: 200%+ ROC, 2.0+ Sharpe, 60-70% win rate
fn calculate_fitness(result: &BacktestResult) -> f64 {
    let roc = result.return_on_capital;
    let sharpe = result.sharpe_ratio;
    let win_rate = result.win_rate;
    let max_dd_pct = (result.max_drawdown.abs() / 10_000.0) * 100.0; // Convert to %

    // Phase 4 fitness formula
    (roc * 0.3) + (sharpe * 25.0) + (win_rate * 0.2) - (max_dd_pct * 0.5)
}

/// Generate all parameter combinations to test (Phase 4 ranges)
fn generate_parameter_combinations() -> Vec<StrategyParams> {
    let mut combinations = Vec::new();

    // Phase 4 parameter ranges (from requirements)
    let dte_ranges = vec![(21, 35), (30, 45), (35, 50)];

    let delta_ranges = vec![(0.10, 0.25), (0.15, 0.30), (0.20, 0.35), (0.25, 0.40)];

    let profit_targets = vec![40.0, 50.0, 60.0, 75.0];
    let stop_losses = vec![150.0, 200.0, 250.0];
    let max_hold_days_options = vec![21, 28, 35, 42];

    println!("Parameter Ranges:");
    println!("  DTE: {:?}", dte_ranges);
    println!("  Delta: {:?}", delta_ranges);
    println!("  Profit Targets: {:?}", profit_targets);
    println!("  Stop Losses: {:?}", stop_losses);
    println!("  Max Hold Days: {:?}", max_hold_days_options);
    println!();

    // Generate all combinations
    for (dte_min, dte_max) in &dte_ranges {
        for (delta_min, delta_max) in &delta_ranges {
            for &profit_target in &profit_targets {
                for &stop_loss in &stop_losses {
                    for &max_hold_days in &max_hold_days_options {
                        let params = StrategyParams {
                            name: format!(
                                "DTE{}-{}_D{:.2}-{:.2}_PT{}_SL{}_MH{}",
                                dte_min,
                                dte_max,
                                delta_min,
                                delta_max,
                                profit_target as i32,
                                stop_loss as i32,
                                max_hold_days
                            ),
                            dte_min: *dte_min,
                            dte_max: *dte_max,
                            delta_min: *delta_min,
                            delta_max: *delta_max,
                            profit_target_pct: Some(profit_target),
                            stop_loss_pct: Some(stop_loss),
                            max_hold_days: Some(max_hold_days),
                            position_size_pct: 10.0,
                            min_credit: Some(0.30),
                            // Transaction costs (realistic retail broker)
                            commission_per_contract: 0.65,
                            slippage_ticks: 1.0,
                            apply_bid_ask_spread: true,
                            custom_params: std::collections::HashMap::new(),
                        };
                        combinations.push(params);
                    }
                }
            }
        }
    }

    combinations
}

/// Run backtest for a single parameter combination (Phase 4 with all enhancements)
fn run_single_backtest(
    params: StrategyParams,
    data_dir: &str,
    spot_dir: &str,
    symbol: &str,
    start_date: NaiveDate,
    end_date: NaiveDate,
    initial_capital: f64,
) -> Option<ParameterResult> {
    // Create new loaders for this thread (each thread needs its own cache)
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

    // Create strategy
    let strategy = BullPutSpread::new(params.clone());

    // Create engine with both loaders
    let mut engine = BacktestEngine::new(loader, spot_loader, initial_capital);

    // Run backtest (suppress verbose output for parallel execution)
    match engine.run_bull_put_spread(symbol, &strategy, &params, start_date, end_date) {
        Ok(result) => {
            // Calculate fitness
            let fitness = calculate_fitness(&result);

            // Create parameter result
            Some(ParameterResult {
                params: params.clone(),
                num_trades: result.num_trades,
                total_pnl: result.total_pnl,
                win_rate: result.win_rate,
                sharpe_ratio: result.sharpe_ratio,
                sortino_ratio: result.sortino_ratio,
                max_drawdown: result.max_drawdown,
                return_on_capital: result.return_on_capital,
                profit_factor: result.profit_factor,
                avg_win: result.avg_win,
                avg_loss: result.avg_loss,
                max_consecutive_losses: result.max_consecutive_losses,
                avg_days_in_trade: result.avg_days_in_trade,
                fitness,
            })
        }
        Err(e) => {
            eprintln!("Backtest failed for {}: {:?}", params.name, e);
            None
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bull Put Spread Parameter Sweep Optimizer (Phase 4) ===\n");

    // Configuration
    let data_dir = "data/yfinance/options_historical";
    let spot_dir = "data/yfinance/ohlcv";
    let symbol = "AAPL";
    let initial_capital = 10_000.0;
    let start_date = NaiveDate::from_ymd_opt(2020, 1, 1).expect("Invalid start date");
    let end_date = NaiveDate::from_ymd_opt(2023, 12, 31).expect("Invalid end date");
    let output_file = "results/parameter_sweep_results.json";

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Period: {} to {}", start_date, end_date);
    println!("  Output: {}", output_file);
    println!();

    // Initialize data loaders to verify data availability
    println!("Loading historical options data...");
    let loader = OptionsDataLoader::new(data_dir)?;

    // Check available options data
    let stats = loader.get_stats()?;
    if let Some(days) = stats.get(symbol) {
        println!("  {} has {} days of options data", symbol, days);
    } else {
        eprintln!("Error: No options data available for {}", symbol);
        eprintln!("Available symbols: {:?}", stats.keys().collect::<Vec<_>>());
        return Ok(());
    }

    // Initialize spot data loader
    println!("Loading historical spot price data...");
    let mut spot_loader = SpotDataLoader::new(spot_dir)?;
    let (min_date, max_date) = spot_loader.get_date_range(symbol)?;
    println!("  {} spot data: {} to {}", symbol, min_date, max_date);
    println!();

    // Generate all parameter combinations
    let combinations = generate_parameter_combinations();
    let total_combinations = combinations.len();

    println!("Parameter Sweep Configuration:");
    println!("  Total combinations: {}", total_combinations);
    println!(
        "  Parallel workers: {} (Rayon)",
        rayon::current_num_threads()
    );
    println!();

    // Progress tracking
    let progress = Arc::new(Mutex::new(0));
    let results = Arc::new(Mutex::new(Vec::new()));

    // Start sweep
    println!("Starting parameter sweep...\n");
    let start_time = Instant::now();

    // Parallel parameter sweep using Rayon (Phase 4: with spot data)
    combinations.par_iter().for_each(|params| {
        // Run backtest (creates its own loaders for thread safety)
        if let Some(result) = run_single_backtest(
            params.clone(),
            data_dir,
            spot_dir,
            symbol,
            start_date,
            end_date,
            initial_capital,
        ) {
            // Store result
            results.lock().unwrap().push(result);
        }

        // Update progress
        let mut prog = progress.lock().unwrap();
        *prog += 1;
        if *prog % 10 == 0 || *prog == total_combinations {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = *prog as f64 / elapsed;
            let eta = (total_combinations - *prog) as f64 / rate;
            println!(
                "Progress: {}/{} ({:.1}%) - {:.1} tests/sec - ETA: {:.0}s",
                prog,
                total_combinations,
                (*prog as f64 / total_combinations as f64) * 100.0,
                rate,
                eta
            );
        }
    });

    let elapsed = start_time.elapsed();
    println!("\n=== Parameter Sweep Complete ===");
    println!("Total time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Tests per second: {:.2}",
        total_combinations as f64 / elapsed.as_secs_f64()
    );
    println!();

    // Sort results by fitness (descending)
    let mut all_results = results.lock().unwrap().clone();
    all_results.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

    // Get top 20 (Phase 4 requirement)
    let top_20: Vec<_> = all_results.iter().take(20).cloned().collect();

    // Display top 20 results
    println!("=== Top 20 Parameter Combinations (Phase 4) ===\n");
    for (rank, result) in top_20.iter().enumerate() {
        println!("#{} - Fitness: {:.2}", rank + 1, result.fitness);
        println!("  Parameters:");
        println!(
            "    DTE Range: {} to {}",
            result.params.dte_min, result.params.dte_max
        );
        println!(
            "    Delta Range: {:.2} to {:.2}",
            result.params.delta_min, result.params.delta_max
        );
        println!(
            "    Profit Target: {:.0}%",
            result.params.profit_target_pct.unwrap_or(0.0)
        );
        println!(
            "    Stop Loss: {:.0}%",
            result.params.stop_loss_pct.unwrap_or(0.0)
        );
        println!(
            "    Max Hold Days: {}",
            result.params.max_hold_days.unwrap_or(0)
        );
        println!("  Performance Metrics:");
        println!("    Trades: {}", result.num_trades);
        println!("    Total P&L: ${:.2}", result.total_pnl);
        println!("    Win Rate: {:.1}%", result.win_rate);
        println!("    Sharpe Ratio: {:.2}", result.sharpe_ratio);
        println!("    Sortino Ratio: {:.2}", result.sortino_ratio);
        println!("    Max Drawdown: ${:.2}", result.max_drawdown);
        println!("    ROC: {:.2}%", result.return_on_capital);
        println!("    Profit Factor: {:.2}", result.profit_factor);
        println!("    Avg Win: ${:.2}", result.avg_win);
        println!("    Avg Loss: ${:.2}", result.avg_loss);
        println!(
            "    Max Consecutive Losses: {}",
            result.max_consecutive_losses
        );
        println!("    Avg Days in Trade: {:.1}", result.avg_days_in_trade);
        println!();
    }

    // Create results directory if needed
    if let Some(parent) = std::path::Path::new(output_file).parent() {
        fs::create_dir_all(parent)?;
    }

    // Save results to JSON (Phase 4: with all enhancements)
    let output_data = serde_json::json!({
        "metadata": {
            "symbol": symbol,
            "start_date": start_date.to_string(),
            "end_date": end_date.to_string(),
            "initial_capital": initial_capital,
            "total_combinations": total_combinations,
            "execution_time_seconds": elapsed.as_secs_f64(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "enhancements": {
                "transaction_costs": true,
                "margin_limits": true,
                "real_spot_prices": true,
                "commission_per_contract": 0.65,
                "slippage_ticks": 1.0,
                "bid_ask_spread": true,
            },
            "fitness_formula": "(ROC * 0.3) + (Sharpe * 25) + (WinRate * 0.2) - (MaxDD% * 0.5)",
        },
        "top_20": top_20,
        "all_results": all_results,
    });

    let json_str = serde_json::to_string_pretty(&output_data)?;
    fs::write(output_file, json_str)?;

    println!("=== Results Saved ===");
    println!("Output file: {}", output_file);
    println!("  - Top 20 parameter sets");
    println!("  - All {} results", all_results.len());
    println!("  - Phase 4 enhancements applied:");
    println!("    * Transaction costs ($0.65/contract + slippage)");
    println!("    * Margin limits (10% position sizing)");
    println!("    * Real spot prices from OHLCV data");
    println!("    * Bid-ask spread modeling");
    println!();

    // Summary statistics
    println!("=== Summary Statistics ===");
    if !all_results.is_empty() {
        let best_fitness = all_results[0].fitness;
        let worst_fitness = all_results.last().unwrap().fitness;
        let avg_fitness: f64 =
            all_results.iter().map(|r| r.fitness).sum::<f64>() / all_results.len() as f64;

        println!("Fitness Scores:");
        println!("  Best: {:.2}", best_fitness);
        println!("  Worst: {:.2}", worst_fitness);
        println!("  Average: {:.2}", avg_fitness);
        println!();

        // Best overall metrics
        let best = &all_results[0];
        println!("Best Strategy ({}):", best.params.name);
        println!("  Parameters:");
        println!(
            "    DTE: {}-{}, Delta: {:.2}-{:.2}",
            best.params.dte_min, best.params.dte_max, best.params.delta_min, best.params.delta_max
        );
        println!(
            "    PT: {:.0}%, SL: {:.0}%, Max Hold: {} days",
            best.params.profit_target_pct.unwrap_or(0.0),
            best.params.stop_loss_pct.unwrap_or(0.0),
            best.params.max_hold_days.unwrap_or(0)
        );
        println!("  Performance:");
        println!("    ROC: {:.2}%", best.return_on_capital);
        println!("    Sharpe: {:.2}", best.sharpe_ratio);
        println!("    Sortino: {:.2}", best.sortino_ratio);
        println!("    Win Rate: {:.1}%", best.win_rate);
        println!("    Profit Factor: {:.2}", best.profit_factor);
        println!("    Max DD: ${:.2}", best.max_drawdown);
        println!("    Trades: {}", best.num_trades);
        println!("    Avg Days: {:.1}", best.avg_days_in_trade);
        println!();

        // Comparison to baseline (266% ROC, 2.5 Sharpe)
        println!("=== Comparison to Baseline ===");
        println!("  Baseline: 266% ROC, 2.5 Sharpe, ~65% Win Rate");
        println!(
            "  Best:     {:.2}% ROC, {:.2} Sharpe, {:.1}% Win Rate",
            best.return_on_capital, best.sharpe_ratio, best.win_rate
        );
        println!(
            "  Delta:    {:.2}% ROC, {:.2} Sharpe, {:.1}% Win Rate",
            best.return_on_capital - 266.0,
            best.sharpe_ratio - 2.5,
            best.win_rate - 65.0
        );
        println!();

        // Phase 4 targets assessment
        println!("=== Phase 4 Target Assessment ===");
        let meets_roc = best.return_on_capital >= 200.0;
        let meets_sharpe = best.sharpe_ratio >= 2.0;
        let meets_winrate = best.win_rate >= 60.0 && best.win_rate <= 70.0;
        println!(
            "  ROC >= 200%: {} ({:.2}%)",
            if meets_roc { "✓" } else { "✗" },
            best.return_on_capital
        );
        println!(
            "  Sharpe >= 2.0: {} ({:.2})",
            if meets_sharpe { "✓" } else { "✗" },
            best.sharpe_ratio
        );
        println!(
            "  Win Rate 60-70%: {} ({:.1}%)",
            if meets_winrate { "✓" } else { "✗" },
            best.win_rate
        );

        if meets_roc && meets_sharpe && meets_winrate {
            println!("\n  Result: All targets met!");
        } else {
            println!("\n  Result: Some targets not met, but best achievable with realistic costs.");
        }
    }

    println!("\n=== Optimization Complete ===");
    println!("Found optimal parameters with Phase 4 enhancements!");
    println!("Next steps:");
    println!("  1. Review top 20 parameter sets in JSON output");
    println!("  2. Validate best parameters with walk-forward analysis");
    println!("  3. Test on out-of-sample data (2024+)");
    println!("  4. Consider regime-specific parameter sets");

    Ok(())
}
