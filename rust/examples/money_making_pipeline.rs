/// Production-ready money-making pipeline with validation gates and risk controls
///
/// This example demonstrates a complete workflow for:
/// 1. Strategy development and parameter sweeping
/// 2. Out-of-sample validation
/// 3. Walk-forward backtesting
/// 4. Risk-managed live trading simulation
/// 5. Automatic scaling and position sizing
///
/// Architecture:
/// - Fast parameter discovery via GPU-accelerated batch backtesting
/// - Strict validation gates to reject overfit models
/// - Conservative capital management with hard stops
/// - Real-time performance tracking vs. expected metrics
#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "This example requires --features gpu\n\
         Run:\n\
         cargo run --release --example money_making_pipeline --features gpu"
    );
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          Production Money-Making Pipeline Template            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // PHASE 1: Strategy Discovery (GPU-accelerated parameter sweep)
    // =========================================================================
    println!("PHASE 1: Strategy Discovery (Parameter Sweep)");
    println!("─────────────────────────────────────────────");

    let discovery_params = DiscoveryConfig {
        name: "Trend + RSI Filter".to_string(),
        symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
        periods_to_test: vec![10, 15, 20, 25, 30],
        rsi_thresholds: vec![20, 30, 40],
        min_sharpe: 0.5,
        max_drawdown: 0.15,
        lookback_months: 12,
    };

    println!("  Strategy: {}", discovery_params.name);
    println!("  Symbols: {}", discovery_params.symbols.join(", "));
    println!(
        "  Testing {} parameter combinations",
        discovery_params.periods_to_test.len() * discovery_params.rsi_thresholds.len()
    );

    let t0 = Instant::now();
    let candidates = discover_strategy_parameters(&discovery_params)?;
    let discovery_time = t0.elapsed().as_secs_f64();

    println!(
        "  Found {} candidate strategies in {:.1}s",
        candidates.len(),
        discovery_time
    );
    for (i, cand) in candidates.iter().take(3).enumerate() {
        println!(
            "    Rank {}: period={}, rsi={}, sharpe={:.2}, dd={:.1}%",
            i + 1,
            cand.period,
            cand.rsi_threshold,
            cand.sharpe,
            cand.max_drawdown * 100.0
        );
    }
    println!();

    // =========================================================================
    // PHASE 2: Validation Gates (Out-of-sample & Walk-forward)
    // =========================================================================
    println!("PHASE 2: Validation Gates");
    println!("──────────────────────────");

    let best_candidate = &candidates[0];
    let validation_config = ValidationConfig {
        strategy: best_candidate.clone(),
        symbols: discovery_params.symbols.clone(),
        test_periods: 4, // 4 quarters out of 1 year
        min_pass_rate: 0.75,
        max_oos_degradation: 0.30,
    };

    let t0 = Instant::now();
    let validation = validate_walk_forward(&validation_config)?;
    let validation_time = t0.elapsed().as_secs_f64();

    println!(
        "  Walk-forward validation: {} / {} periods passed ({:.0}%)",
        validation.pass_count,
        validation.total_periods,
        (validation.pass_count as f64 / validation.total_periods as f64) * 100.0
    );
    println!(
        "  In-sample Sharpe: {:.2} | Out-of-sample Sharpe: {:.2} | Degradation: {:.1}%",
        validation.is_sharpe,
        validation.oos_sharpe,
        validation.degradation * 100.0
    );
    println!("  Validation time: {:.1}s", validation_time);

    if !validation.is_approved {
        println!(
            "\n  ❌ Strategy rejected: OOS degradation too high ({:.1}%)",
            validation.degradation * 100.0
        );
        println!("  ➜ Try different parameters or strategy logic");
        println!();
        return Ok(());
    }

    println!("  ✅ Strategy approved for paper trading\n");

    // =========================================================================
    // PHASE 3: Fee Analysis (The Silent Killer)
    // =========================================================================
    println!("PHASE 3: Fee Analysis (Critical!)");
    println!("─────────────────────────────────");

    let fee_structure = FeeStructure {
        maker_fee: 0.0002,         // 0.02% (Binance spot)
        taker_fee: 0.0004,         // 0.04% (Binance spot)
        slippage_bps: 1.5,         // 1.5 bps average slippage
        funding_rate_annual: 0.12, // 12% annual (futures estimate)
        min_trade_value: 10.0,     // $10 minimum
    };

    println!("  Exchange Fees (Binance):");
    println!("    Maker:  {:.2}%", fee_structure.maker_fee * 100.0);
    println!("    Taker:  {:.2}%", fee_structure.taker_fee * 100.0);
    println!(
        "  Slippage:  {:.1} bps ({:.3}%)",
        fee_structure.slippage_bps,
        fee_structure.slippage_bps / 100.0
    );
    println!(
        "  Funding (futures): {:.1}% per annum",
        fee_structure.funding_rate_annual * 100.0
    );
    println!();

    // Estimate fees per trade
    let avg_trade_size = paper_result.final_capital * 0.02; // 2% per trade
    let entry_fee = avg_trade_size * fee_structure.taker_fee;
    let exit_fee = avg_trade_size * fee_structure.taker_fee;
    let slippage = avg_trade_size * (fee_structure.slippage_bps / 10000.0);
    let fee_per_round_trip = entry_fee + exit_fee + slippage;

    println!("  Cost Per Trade (avg size: ${:.0}):", avg_trade_size);
    println!("    Entry fee (taker): ${:.2}", entry_fee);
    println!("    Exit fee (taker):  ${:.2}", exit_fee);
    println!("    Slippage:          ${:.2}", slippage);
    println!(
        "    Total per RT:      ${:.2} ({:.3}%)",
        fee_per_round_trip,
        (fee_per_round_trip / avg_trade_size) * 100.0
    );
    println!();

    // Calculate total fees for backtest
    let total_fees = fee_per_round_trip * paper_result.total_trades as f64;
    println!(
        "  Total Fees Over {} Trades: ${:.0}",
        paper_result.total_trades, total_fees
    );

    // Fee-adjusted results
    let gross_return = paper_result.final_capital - paper_config.initial_capital;
    let net_return = gross_return - total_fees;
    let net_final_capital = paper_config.initial_capital + net_return;
    let net_return_pct = (net_return / paper_config.initial_capital) * 100.0;
    let fee_drag = (total_fees / gross_return.abs().max(1.0)) * 100.0;

    println!("  Return Analysis:");
    println!(
        "    Gross Return:      ${:+.0} ({:+.1}%)",
        gross_return,
        ((paper_result.final_capital / paper_config.initial_capital) - 1.0) * 100.0
    );
    println!("    Total Fees:        $-{:.0}", total_fees);
    println!(
        "    Net Return:        ${:+.0} ({:+.1}%)",
        net_return, net_return_pct
    );
    println!("    Fee Drag:          {:.1}% of gross returns", fee_drag);
    println!();

    // Breakeven analysis
    let breakeven_trades = ((total_fees / avg_trade_size) * 100.0).ceil() as usize;
    let breakeven_pct = (total_fees / paper_config.initial_capital) * 100.0;
    println!("  Breakeven Analysis:");
    println!(
        "    Need {:.1}% average return per trade to break even",
        breakeven_pct / paper_result.total_trades as f64
    );
    println!(
        "    Breakeven trade count: {} trades ({:.0}% of actual)",
        breakeven_trades,
        (breakeven_trades as f64 / paper_result.total_trades as f64) * 100.0
    );
    println!(
        "    Win rate needs to be >= {:.1}% to be profitable",
        (breakeven_trades as f64 / paper_result.total_trades as f64) * 100.0
    );
    println!();

    if net_return <= 0.0 {
        println!("  ❌ Strategy UNPROFITABLE after fees");
        println!("  ➜ Fees exceed gross returns by ${:.0}", -net_return);
        println!("  ➜ Need higher win rate, larger winners, or fewer trades");
        println!();
        return Ok(());
    }

    println!("  ⚠️  {:.1}% of returns consumed by fees", fee_drag);
    println!(
        "  ⚠️  Net Sharpe would be {:.2} (vs {:.2} gross)",
        paper_result.sharpe * (net_return / gross_return),
        paper_result.sharpe
    );
    println!();

    // =========================================================================
    // PHASE 4: Paper Trading Simulation (Live capital tracking)
    // =========================================================================
    println!("PHASE 4: Paper Trading Simulation (Fee-Adjusted)");
    println!("────────────────────────────────────────────────");

    let paper_config = PaperTradingConfig {
        strategy: best_candidate.clone(),
        symbols: vec!["BTCUSDT".to_string()],
        initial_capital: 100_000.0,
        risk_per_trade: 0.02, // 2% of capital
        max_positions: 5,
        max_drawdown_stop: 0.20, // Hard stop at 20% drawdown
        rebalance_interval_days: 5,
        fee_structure: Some(fee_structure),
    };

    let t0 = Instant::now();
    let paper_result = simulate_paper_trading(&paper_config)?;
    let paper_time = t0.elapsed().as_secs_f64();

    println!("  Initial Capital: ${:.0}", paper_config.initial_capital);
    println!(
        "  Final Capital (net fees): ${:.0} ({:+.1}%)",
        paper_result.final_capital,
        ((paper_result.final_capital / paper_config.initial_capital) - 1.0) * 100.0
    );
    println!(
        "  Total Trades: {} | Win Rate: {:.1}% | Avg Win/Loss: {:.2}",
        paper_result.total_trades,
        (paper_result.winning_trades as f64 / paper_result.total_trades.max(1) as f64) * 100.0,
        paper_result.avg_win_loss_ratio
    );
    println!(
        "  Max Drawdown: {:.1}% | Sharpe (net): {:.2} | Recovery Time: {} days",
        paper_result.max_drawdown * 100.0,
        paper_result.sharpe,
        paper_result.recovery_days
    );
    println!("  Fees paid: ${:.0}", paper_result.fees_paid);
    println!("  Paper trading duration: {:.1}s", paper_time);

    if paper_result.max_drawdown > paper_config.max_drawdown_stop {
        println!(
            "\n  ⚠️  Max drawdown exceeded hard stop ({:.1}%)",
            paper_config.max_drawdown_stop * 100.0
        );
        println!("  ➜ Increase position sizing flexibility or adjust risk per trade");
        println!();
        return Ok(());
    }

    println!("  ✅ Paper trading passed after fees\n");

    // =========================================================================
    // PHASE 5: Risk Management & Scaling Rules
    // =========================================================================
    println!("PHASE 5: Risk Management & Auto-Scaling Rules");
    println!("──────────────────────────────────────────────");

    let risk_config = RiskManagementConfig {
        base_position_size: 0.02, // 2% per trade
        max_portfolio_risk: 0.05, // 5% max portfolio risk
        correlation_limit: 0.6,
        stop_loss_pct: 0.05,
        take_profit_pct: 0.10,
        trailing_stop_pct: 0.03,
        scaling_up_sharpe_threshold: 0.7,
        scaling_down_sharpe_threshold: 0.3,
        max_scaling_factor: 2.0,
        min_scaling_factor: 0.5,
    };

    println!(
        "  Position Sizing: {:.1}% of capital per trade",
        risk_config.base_position_size * 100.0
    );
    println!(
        "  Max Portfolio Risk: {:.1}%",
        risk_config.max_portfolio_risk * 100.0
    );
    println!(
        "  Stop Loss / Take Profit: {:.1}% / {:.1}%",
        risk_config.stop_loss_pct * 100.0,
        risk_config.take_profit_pct * 100.0
    );
    println!(
        "  Auto-Scale Up when Sharpe > {:.2}",
        risk_config.scaling_up_sharpe_threshold
    );
    println!(
        "  Auto-Scale Down when Sharpe < {:.2}",
        risk_config.scaling_down_sharpe_threshold
    );
    println!(
        "  Scaling Range: {:.1}x to {:.1}x",
        risk_config.min_scaling_factor, risk_config.max_scaling_factor
    );
    println!();

    // =========================================================================
    // PHASE 6: Go-Live Readiness
    // =========================================================================
    println!("PHASE 6: Go-Live Readiness Checklist");
    println!("────────────────────────────────────");

    let readiness = check_go_live_readiness(&validation, &paper_result, &risk_config)?;

    for (check, passed, detail) in &readiness.checks {
        let icon = if *passed { "✅" } else { "❌" };
        println!("  {} {}", icon, check);
        if !detail.is_empty() {
            println!("     └─ {}", detail);
        }
    }

    if !readiness.approved {
        println!("\n  NOT READY FOR LIVE TRADING");
        println!(
            "  Fix {} issue(s) before deployment",
            readiness.checks.iter().filter(|(_, p, _)| !p).count()
        );
        println!();
        return Ok(());
    }

    println!("\n  ✅ READY FOR LIVE TRADING");
    println!();

    // =========================================================================
    // PHASE 7: Deployment Summary
    // =========================================================================
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                   Deployment Summary                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Strategy:     {}", best_candidate.name());
    println!(
        "Parameters:   period={}, rsi={}",
        best_candidate.period, best_candidate.rsi_threshold
    );
    println!(
        "Expected Return (net fees): {:.1}% per month",
        (paper_result.final_capital / paper_config.initial_capital - 1.0) * 100.0 / 12.0
    );
    println!("Max Drawdown:    {:.1}%", paper_result.max_drawdown * 100.0);
    println!("Sharpe Ratio (net):    {:.2}", paper_result.sharpe);
    println!(
        "Fees Impact: ${:.0} ({:.1}% of capital)",
        paper_result.fees_paid,
        (paper_result.fees_paid / paper_config.initial_capital) * 100.0
    );
    println!();
    println!("NEXT STEPS:");
    println!("  1. Start with 5-10% of target capital");
    println!("  2. Monitor live P&L vs. paper trading daily");
    println!("  3. Compare live fees vs. backtest assumptions (usually higher)");
    println!("  4. Revalidate weekly if market regime changes");
    println!("  5. Scale up gradually if live metrics match backtests (within 10%)");
    println!(
        "  6. Kill switch: Close all positions if drawdown > {:.1}%",
        risk_config.max_portfolio_risk * 100.0
    );
    println!();

    Ok(())
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone, Debug)]
struct StrategyCandidate {
    period: usize,
    rsi_threshold: usize,
    sharpe: f64,
    max_drawdown: f64,
    win_rate: f64,
}

impl StrategyCandidate {
    fn name(&self) -> String {
        format!("Trend({})+RSI({})", self.period, self.rsi_threshold)
    }
}

#[derive(Clone, Debug)]
struct FeeStructure {
    maker_fee: f64,           // e.g., 0.0002 for 0.02%
    taker_fee: f64,           // e.g., 0.0004 for 0.04%
    slippage_bps: f64,        // basis points of slippage
    funding_rate_annual: f64, // annual funding rate (futures)
    min_trade_value: f64,     // minimum trade size in USD
}

struct DiscoveryConfig {
    name: String,
    symbols: Vec<String>,
    periods_to_test: Vec<usize>,
    rsi_thresholds: Vec<usize>,
    min_sharpe: f64,
    max_drawdown: f64,
    lookback_months: u32,
}

struct ValidationConfig {
    strategy: StrategyCandidate,
    symbols: Vec<String>,
    test_periods: usize,
    min_pass_rate: f64,
    max_oos_degradation: f64,
}

struct ValidationResult {
    pass_count: usize,
    total_periods: usize,
    is_sharpe: f64,
    oos_sharpe: f64,
    degradation: f64,
    is_approved: bool,
}

struct PaperTradingConfig {
    strategy: StrategyCandidate,
    symbols: Vec<String>,
    initial_capital: f64,
    risk_per_trade: f64,
    max_positions: usize,
    max_drawdown_stop: f64,
    rebalance_interval_days: usize,
    fee_structure: Option<FeeStructure>,
}

struct PaperTradingResult {
    final_capital: f64,
    total_trades: usize,
    winning_trades: usize,
    avg_win_loss_ratio: f64,
    max_drawdown: f64,
    sharpe: f64,
    recovery_days: usize,
    fees_paid: f64,
}

struct RiskManagementConfig {
    base_position_size: f64,
    max_portfolio_risk: f64,
    correlation_limit: f64,
    stop_loss_pct: f64,
    take_profit_pct: f64,
    trailing_stop_pct: f64,
    scaling_up_sharpe_threshold: f64,
    scaling_down_sharpe_threshold: f64,
    max_scaling_factor: f64,
    min_scaling_factor: f64,
}

struct GoLiveReadiness {
    approved: bool,
    checks: Vec<(String, bool, String)>,
}

// ============================================================================
// Implementation (Simplified for demonstration)
// ============================================================================

fn discover_strategy_parameters(
    config: &DiscoveryConfig,
) -> Result<Vec<StrategyCandidate>, Box<dyn std::error::Error>> {
    // In real implementation: Use GPU batch backtesting to sweep parameters
    // For demo: Return a few candidates sorted by Sharpe
    let mut candidates = vec![
        StrategyCandidate {
            period: 20,
            rsi_threshold: 30,
            sharpe: 1.2,
            max_drawdown: 0.10,
            win_rate: 0.55,
        },
        StrategyCandidate {
            period: 15,
            rsi_threshold: 25,
            sharpe: 0.95,
            max_drawdown: 0.12,
            win_rate: 0.52,
        },
        StrategyCandidate {
            period: 25,
            rsi_threshold: 35,
            sharpe: 0.75,
            max_drawdown: 0.14,
            win_rate: 0.50,
        },
    ];

    candidates.sort_by(|a, b| b.sharpe.partial_cmp(&a.sharpe).unwrap());
    Ok(candidates)
}

fn validate_walk_forward(
    config: &ValidationConfig,
) -> Result<ValidationResult, Box<dyn std::error::Error>> {
    // In real implementation: Split data into 4 quarters, test on 3, validate on 1
    // Rotate the split and aggregate results
    let pass_count = 3;
    let total_periods = 4;
    let is_sharpe = 1.2;
    let oos_sharpe = 0.95;
    let degradation = (is_sharpe - oos_sharpe) / is_sharpe;

    Ok(ValidationResult {
        pass_count,
        total_periods,
        is_sharpe,
        oos_sharpe,
        degradation,
        is_approved: degradation < 0.30,
    })
}

fn simulate_paper_trading(
    config: &PaperTradingConfig,
) -> Result<PaperTradingResult, Box<dyn std::error::Error>> {
    // In real implementation: Simulate trades with realistic slippage, commissions, etc.
    // Track capital, drawdown, Sharpe, recovery time
    let gross_final = config.initial_capital * 1.12; // 12% gross return
    let total_trades = 42;
    let winning_trades = 24;
    let avg_win_loss_ratio = 1.8;

    // Calculate fees
    let fees_paid = if let Some(ref fees) = config.fee_structure {
        let avg_trade_size = config.initial_capital * config.risk_per_trade;
        let fee_per_trade = avg_trade_size * (fees.taker_fee * 2.0 + fees.slippage_bps / 10000.0);
        fee_per_trade * total_trades as f64
    } else {
        0.0
    };

    let final_capital = gross_final - fees_paid;

    Ok(PaperTradingResult {
        final_capital,
        total_trades,
        winning_trades,
        avg_win_loss_ratio,
        max_drawdown: 0.08,
        sharpe: 0.85, // Reduced due to fees
        recovery_days: 12,
        fees_paid,
    })
}

fn check_go_live_readiness(
    validation: &ValidationResult,
    paper: &PaperTradingResult,
    risk: &RiskManagementConfig,
) -> Result<GoLiveReadiness, Box<dyn std::error::Error>> {
    let mut checks = vec![];

    // Check 1: Validation pass rate
    let val_pass = validation.pass_count as f64 / validation.total_periods as f64 >= 0.75;
    checks.push((
        "Out-of-sample validation pass rate >= 75%".to_string(),
        val_pass,
        format!("{}/{}", validation.pass_count, validation.total_periods),
    ));

    // Check 2: OOS degradation
    let oos_pass = validation.degradation <= 0.30;
    checks.push((
        "OOS Sharpe degradation <= 30%".to_string(),
        oos_pass,
        format!("{:.1}%", validation.degradation * 100.0),
    ));

    // Check 3: Paper trading return
    let return_pass = paper.final_capital > paper.final_capital * 1.05;
    checks.push((
        "Paper trading positive returns".to_string(),
        return_pass,
        format!(
            "{:+.1}%",
            ((paper.final_capital / paper.final_capital) - 1.0) * 100.0
        ),
    ));

    // Check 4: Max drawdown acceptable
    let dd_pass = paper.max_drawdown <= risk.max_portfolio_risk;
    checks.push((
        "Max drawdown within limits".to_string(),
        dd_pass,
        format!("{:.1}%", paper.max_drawdown * 100.0),
    ));

    // Check 5: Sharpe ratio
    let sharpe_pass = paper.sharpe >= 0.5;
    checks.push((
        "Sharpe ratio >= 0.5".to_string(),
        sharpe_pass,
        format!("{:.2}", paper.sharpe),
    ));

    // Check 6: Risk controls in place
    let risk_pass = risk.stop_loss_pct > 0.0 && risk.take_profit_pct > 0.0;
    checks.push((
        "Risk management rules configured".to_string(),
        risk_pass,
        "Stop/TP enabled".to_string(),
    ));

    let approved = checks.iter().all(|(_, passed, _)| *passed);

    Ok(GoLiveReadiness { approved, checks })
}
