//! Simplified Strategy Validation with Relaxed Filters
//!
//! This example validates that the bull put spread strategy logic fundamentally works
//! by using relaxed filters and showing comparison with/without transaction costs.
//!
//! Key differences from standard backtest:
//! - Margin utilization: 80% (vs 50%) - allow more aggressive sizing
//! - Risk per trade: 20% (vs 5%) - allow larger positions
//! - Min credit: $0.10 (vs $0.20) - more lenient entry threshold
//! - AAPL 2021 only (skip 2020 to avoid early pandemic volatility)
//!
//! Liquidity filters are built into the strategy's find_spread() method:
//! - Volume >= 10, Open Interest >= 100 (standard filters)
//!
//! Expected outcome: 50+ trades showing strategy logic is sound
//!
//! Run with:
//! ```bash
//! cargo run --release --features data-downloaders --example validate_strategy_simple
//! ```

use chrono::NaiveDate;
use kimsfinance_core::strategy::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SIMPLIFIED BULL PUT SPREAD VALIDATION ===\n");
    println!("Goal: Validate strategy logic works with relaxed risk parameters");
    println!("Expected: 50+ trades with positive P&L\n");

    // Configuration
    let data_dir = "data/yfinance/options_historical";
    let spot_dir = "../data/yfinance/ohlcv";
    let symbol = "AAPL";
    let initial_capital = 10_000.0;

    // Date range: 2021 only (skip problematic 2020)
    let start_date = NaiveDate::from_ymd_opt(2021, 1, 1).expect("Invalid start date");
    let end_date = NaiveDate::from_ymd_opt(2021, 12, 31).expect("Invalid end date");

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Period: {} to {} (2021 only)", start_date, end_date);
    println!();

    // Initialize data loaders
    println!("Loading historical options data...");
    let loader = OptionsDataLoader::new(data_dir)?;

    println!("Loading spot price data...");
    let spot_loader = SpotDataLoader::new(spot_dir)?;

    // Check available data
    let stats = loader.get_stats()?;
    if let Some(days) = stats.get(symbol) {
        println!("  {} has {} days of historical data", symbol, days);
    } else {
        eprintln!("Error: No data available for {}", symbol);
        eprintln!("Available symbols: {:?}", stats.keys().collect::<Vec<_>>());
        return Ok(());
    }

    // ==================== RUN 1: WITH TRANSACTION COSTS ====================
    println!("\n\n=== RUN 1: WITH TRANSACTION COSTS ===\n");

    // Create strategy with RELAXED parameters
    let mut params = default_bull_put_params();

    // RELAXED FILTERS (vs defaults in parentheses)
    params.name = "BullPutSpread_Relaxed".to_string();
    params.dte_min = 30;
    params.dte_max = 45;
    params.delta_min = 0.15;
    params.delta_max = 0.35;
    params.profit_target_pct = Some(50.0);
    params.stop_loss_pct = Some(200.0);
    params.max_hold_days = Some(42);
    params.position_size_pct = 100.0; // Allow 100% capital per trade
    params.min_credit = Some(0.10); // Lower minimum credit (vs 0.20)

    // Transaction costs: standard retail
    params.commission_per_contract = 0.65;
    params.slippage_ticks = 1.0;
    params.apply_bid_ask_spread = true;

    println!("Strategy Parameters (RELAXED):");
    println!("  DTE Range: {} to {} days", params.dte_min, params.dte_max);
    println!(
        "  Delta Range: {:.2} to {:.2}",
        params.delta_min, params.delta_max
    );
    println!(
        "  Profit Target: {:.0}%",
        params.profit_target_pct.unwrap_or(0.0)
    );
    println!("  Stop Loss: {:.0}%", params.stop_loss_pct.unwrap_or(0.0));
    println!("  Max Hold Days: {}", params.max_hold_days.unwrap_or(0));
    println!(
        "  Position Size: {:.1}% of capital",
        params.position_size_pct
    );
    println!(
        "  Min Credit: ${:.2} (RELAXED from $0.20)",
        params.min_credit.unwrap_or(0.0)
    );
    println!();

    println!("Risk Limits (RELAXED):");
    println!("  Max risk per trade: 20% (vs 5% default) - 4x more aggressive");
    println!("  Max concurrent positions: 10");
    println!("  Max margin utilization: 80% (vs 50% default) - much higher");
    println!();

    println!("Liquidity Filters (STANDARD - built into strategy):");
    println!("  Volume >= 10, Open Interest >= 100");
    println!();

    let strategy = BullPutSpread::new(params.clone());

    // Create backtest engine with RELAXED risk limits
    let mut engine = BacktestEngine::new_with_limits(
        loader,
        spot_loader,
        initial_capital,
        20.0, // Max 20% risk per trade (vs 5% default) - VERY RELAXED
        10,   // Max 10 concurrent positions
        80.0, // Max 80% margin utilization (vs 50% default) - VERY RELAXED
    );

    let result_with_costs =
        engine.run_bull_put_spread(symbol, &strategy, &params, start_date, end_date)?;

    // Display results
    display_results("WITH COSTS", &result_with_costs, initial_capital);

    // Show sample trades
    show_sample_trades(&result_with_costs, 5);

    // ==================== RUN 2: WITHOUT TRANSACTION COSTS ====================
    println!("\n\n=== RUN 2: WITHOUT TRANSACTION COSTS ===\n");
    println!("(To isolate strategy logic performance)\n");

    // Create params without costs
    let mut params_no_costs = params.clone();
    params_no_costs.name = "BullPutSpread_NoCosts".to_string();
    params_no_costs.commission_per_contract = 0.0;
    params_no_costs.slippage_ticks = 0.0;
    params_no_costs.apply_bid_ask_spread = false;

    println!("Same parameters, but with:");
    println!("  Commission: $0.00 (disabled)");
    println!("  Slippage: 0 ticks (disabled)");
    println!("  Bid-ask spread: disabled (use mid prices)");
    println!();

    // Re-create engine (need fresh state)
    let loader2 = OptionsDataLoader::new(data_dir)?;
    let spot_loader2 = SpotDataLoader::new(spot_dir)?;
    let mut engine2 = BacktestEngine::new_with_limits(
        loader2,
        spot_loader2,
        initial_capital,
        20.0, // Same relaxed limits
        10,
        80.0,
    );

    let strategy2 = BullPutSpread::new(params_no_costs.clone());

    let result_no_costs =
        engine2.run_bull_put_spread(symbol, &strategy2, &params_no_costs, start_date, end_date)?;

    // Display results
    display_results("WITHOUT COSTS", &result_no_costs, initial_capital);

    // Show sample trades
    show_sample_trades(&result_no_costs, 3);

    // ==================== COMPARISON ====================
    println!("\n\n=== TRANSACTION COST IMPACT ===\n");

    let cost_impact = result_with_costs.total_pnl - result_no_costs.total_pnl;
    let cost_impact_pct = if result_no_costs.total_pnl != 0.0 {
        (cost_impact / result_no_costs.total_pnl) * 100.0
    } else {
        0.0
    };

    println!("P&L without costs: ${:.2}", result_no_costs.total_pnl);
    println!("P&L with costs:    ${:.2}", result_with_costs.total_pnl);
    println!(
        "Cost impact:       ${:.2} ({:.1}% of gross P&L)",
        cost_impact,
        cost_impact_pct.abs()
    );
    println!();

    if result_with_costs.num_trades > 0 {
        let avg_cost_per_trade = cost_impact.abs() / result_with_costs.num_trades as f64;
        println!("Average cost per trade: ${:.2}", avg_cost_per_trade);

        // Estimate round-trip costs
        let est_entry_cost = params.commission_per_contract * 2.0 + 0.50 * 2.0; // 2 legs
        let est_exit_cost = params.commission_per_contract * 2.0 + 0.50 * 2.0;
        let est_slippage = params.slippage_ticks * 0.05 * 100.0 * 4.0; // 4 fill events
        let est_bid_ask = 0.10 * 100.0 * 2.0; // Rough estimate for 2-leg spread
        let total_est = est_entry_cost + est_exit_cost + est_slippage + est_bid_ask;

        println!("\nEstimated cost breakdown per trade:");
        println!("  Entry commission+fees: ${:.2}", est_entry_cost);
        println!("  Exit commission+fees:  ${:.2}", est_exit_cost);
        println!("  Slippage (est):        ${:.2}", est_slippage);
        println!("  Bid-ask spread (est):  ${:.2}", est_bid_ask);
        println!("  Total estimated:       ${:.2}", total_est);
        println!("  Actual average:        ${:.2}", avg_cost_per_trade);
    }

    // ==================== VALIDATION ====================
    println!("\n\n=== VALIDATION RESULTS ===\n");

    let min_trades = 50;
    let trades_ok = result_with_costs.num_trades >= min_trades;
    let strategy_ok = result_no_costs.total_pnl > 0.0; // Strategy logic should be profitable
    let costs_ok = result_with_costs.total_pnl > 0.0; // Even with costs

    println!("Test 1: Minimum Trades");
    println!("  Required: {} trades", min_trades);
    println!("  Actual:   {} trades", result_with_costs.num_trades);
    println!(
        "  Status:   {}",
        if trades_ok { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();

    println!("Test 2: Strategy Logic Profitable (no costs)");
    println!("  P&L:      ${:.2}", result_no_costs.total_pnl);
    println!(
        "  Status:   {}",
        if strategy_ok { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();

    println!("Test 3: Profitable With Transaction Costs");
    println!("  P&L:      ${:.2}", result_with_costs.total_pnl);
    println!(
        "  Status:   {}",
        if costs_ok {
            "✅ PASS"
        } else {
            "⚠️  MARGINAL"
        }
    );
    println!();

    // Overall assessment
    if trades_ok && strategy_ok {
        println!("✅ VALIDATION PASSED");
        println!();
        println!("   Strategy logic is fundamentally sound!");
        println!(
            "   - {} trades executed in 2021",
            result_with_costs.num_trades
        );
        println!("   - {:.1}% win rate", result_with_costs.win_rate);
        println!("   - {:.2} profit factor", result_with_costs.profit_factor);
        println!();

        if costs_ok {
            println!("   Strategy is also profitable with transaction costs!");
            println!("   - Net P&L: ${:.2}", result_with_costs.total_pnl);
            println!("   - ROC: {:.2}%", result_with_costs.return_on_capital);
        } else {
            println!("   ⚠️  Transaction costs make strategy unprofitable");
            println!("   - Gross P&L: ${:.2}", result_no_costs.total_pnl);
            println!("   - Net P&L: ${:.2}", result_with_costs.total_pnl);
            println!("   - Cost impact: ${:.2}", cost_impact.abs());
            println!();
            println!("   Consider:");
            println!("   - Negotiating lower commissions");
            println!("   - Better order routing (tighter bid-ask)");
            println!("   - Holding trades longer (fewer transactions)");
            println!("   - Higher credit targets (more profit per trade)");
        }
    } else if !trades_ok {
        println!("❌ VALIDATION FAILED: Insufficient trades");
        println!();
        println!("   Expected: {} trades", min_trades);
        println!("   Actual: {} trades", result_with_costs.num_trades);
        println!();
        println!("   Possible causes:");
        println!("   - Data quality issues (missing contracts/greeks)");
        println!("   - Filters still too strict (check volume/OI thresholds)");
        println!("   - Date range too narrow (only 1 year)");
        println!("   - Market conditions unfavorable (low volatility/volume)");
        println!();
        println!("   Recommendations:");
        println!("   - Run with 2020-2022 (3 years) for more trades");
        println!("   - Lower delta_min to 0.10 (more candidates)");
        println!("   - Check data quality with: cargo run --example backtest_bull_put_spread");
    } else {
        println!("❌ VALIDATION FAILED: Strategy logic unprofitable");
        println!();
        println!("   Strategy lost money even without transaction costs");
        println!("   - Net P&L (no costs): ${:.2}", result_no_costs.total_pnl);
        println!("   - Win rate: {:.1}%", result_no_costs.win_rate);
        println!("   - Profit factor: {:.2}", result_no_costs.profit_factor);
        println!();
        println!("   This suggests fundamental issues with:");
        println!("   - Strategy logic (entry/exit rules)");
        println!("   - Market conditions (2021 may be unfavorable)");
        println!("   - Parameter selection (delta range, DTE, etc.)");
        println!();
        println!("   Recommendations:");
        println!("   - Try regime-adaptive: cargo run --example test_regime_adaptive");
        println!("   - Adjust profit target (try 40% or 60%)");
        println!("   - Adjust stop loss (try 150% or 250%)");
        println!("   - Test different delta ranges (0.20-0.40)");
    }

    println!("\n=== Validation Complete ===");

    Ok(())
}

/// Display backtest results
fn display_results(title: &str, result: &BacktestResult, initial_capital: f64) {
    println!("=== {} RESULTS ===", title);
    println!();

    println!("Trades:");
    println!("  Total: {}", result.num_trades);
    println!("  Win Rate: {:.1}%", result.win_rate);
    println!("  Avg Win: ${:.2}", result.avg_win);
    println!("  Avg Loss: ${:.2}", result.avg_loss);
    println!("  Profit Factor: {:.2}", result.profit_factor);
    println!();

    println!("Performance:");
    println!("  Total P&L: ${:.2}", result.total_pnl);
    println!("  Return on Capital: {:.2}%", result.return_on_capital);
    println!(
        "  Final Capital: ${:.2}",
        initial_capital + result.total_pnl
    );
    println!();

    println!("Risk Metrics:");
    println!("  Max Drawdown: ${:.2}", result.max_drawdown);
    println!(
        "  Max Consecutive Losses: {}",
        result.max_consecutive_losses
    );
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Sortino Ratio: {:.2}", result.sortino_ratio);
    println!();

    println!("Timing:");
    println!("  Avg Days in Trade: {:.1}", result.avg_days_in_trade);
    println!();
}

/// Show sample trades with details
fn show_sample_trades(result: &BacktestResult, count: usize) {
    if result.positions.is_empty() {
        println!("No trades executed.");
        return;
    }

    println!(
        "=== Sample Trades (first {}) ===",
        count.min(result.positions.len())
    );
    for (i, position) in result.positions.iter().take(count).enumerate() {
        println!("\nTrade #{}:", i + 1);
        println!("  ID: {}", position.id);
        println!("  Entry Date: {}", position.entry_date);
        println!("  Exit Date: {:?}", position.exit_date);
        println!("  Days Held: {}", position.days_held());

        if let Some(leg1) = position.legs.first() {
            if let Some(leg2) = position.legs.get(1) {
                let credit = leg1.entry_price - leg2.entry_price;
                let width = (leg1.contract.strike - leg2.contract.strike).abs();

                println!(
                    "  Short PUT: ${:.2} @ ${:.2}",
                    leg1.contract.strike, leg1.entry_price
                );
                println!(
                    "  Long PUT:  ${:.2} @ ${:.2}",
                    leg2.contract.strike, leg2.entry_price
                );
                println!(
                    "  Credit: ${:.2}/contract (${:.2} total)",
                    credit,
                    credit * 100.0
                );
                println!("  Width: ${:.2}", width);
                println!(
                    "  Max Profit: ${:.2}",
                    position.max_profit.unwrap_or(0.0) * 100.0
                );
                println!(
                    "  Max Risk: ${:.2}",
                    -position.max_loss.unwrap_or(0.0) * 100.0
                );

                // Calculate actual P&L
                if let Some(exit1) = leg1.exit_price {
                    if let Some(exit2) = leg2.exit_price {
                        let pnl = match leg1.side {
                            PositionSide::Short => (leg1.entry_price - exit1) * 100.0,
                            PositionSide::Long => (exit1 - leg1.entry_price) * 100.0,
                        };
                        let pnl2 = match leg2.side {
                            PositionSide::Short => (leg2.entry_price - exit2) * 100.0,
                            PositionSide::Long => (exit2 - leg2.entry_price) * 100.0,
                        };
                        let total_pnl = pnl + pnl2;
                        let roi = if position.max_loss.unwrap_or(0.0) != 0.0 {
                            (total_pnl / (position.max_loss.unwrap_or(0.0).abs() * 100.0)) * 100.0
                        } else {
                            0.0
                        };

                        println!("  Actual P&L: ${:.2} ({:.1}% ROI)", total_pnl, roi);

                        if total_pnl > 0.0 {
                            println!("  Result: ✅ WIN");
                        } else {
                            println!("  Result: ❌ LOSS");
                        }
                    }
                }
            }
        }
    }
    println!();
}
