/// Realistic fee impact scenarios: Why paper trading doesn't match live results
///
/// This shows three fee scenarios:
/// 1. Optimistic (what backtesters assume)
/// 2. Realistic (what traders actually experience)
/// 3. Pessimistic (what happens in volatile market conditions)
///
/// Most traders learn about the gap between #1 and #2 the hard way: losing money.
#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "This example requires --features gpu\n\
         Run:\n\
         cargo run --release --example fee_scenarios --features gpu"
    );
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              Fee Impact: 3 Realistic Scenarios                ║");
    println!("║         (Why Paper Trading Never Matches Live Results)        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Hypothetical strategy (used across all scenarios)
    let gross_return_pct = 0.12; // 12% backtest return
    let initial_capital = 100_000.0;
    let gross_final = initial_capital * (1.0 + gross_return_pct);
    let gross_return = gross_final - initial_capital;
    let trade_count = 42;
    let position_size = initial_capital * 0.02; // 2% per trade

    println!("Base Strategy Assumption:");
    println!("  Backtest gross return: {:.1}%", gross_return_pct * 100.0);
    println!("  Initial capital: ${:.0}", initial_capital);
    println!("  Gross final: ${:.0}", gross_final);
    println!("  Trades per period: {}", trade_count);
    println!("  Position size: ${:.0}", position_size);
    println!();

    // =========================================================================
    // SCENARIO 1: Optimistic (Backtester's Dream)
    // =========================================================================
    println!("─────────────────────────────────────────────────────────────────");
    println!("SCENARIO 1: Optimistic (Backtester Assumption)");
    println!("─────────────────────────────────────────────────────────────────");
    println!();

    let opt = FeeScenario {
        name: "Optimistic".to_string(),
        taker_fee_pct: 0.0004,
        slippage_bps: 1.0, // Ideal slippage
        liquidity_factor: 1.0, // Perfect fills
        execution_delay_sec: 0.0, // No delay
    };

    println!("Assumptions:");
    println!("  Taker fee: {:.2}%", opt.taker_fee_pct * 100.0);
    println!("  Slippage: {:.1} bps", opt.slippage_bps);
    println!("  Liquidity: Perfect (ideal fills)");
    println!("  Execution: Instant fills");
    println!();

    let (opt_net, opt_fees) = calculate_net_return(&opt, position_size, gross_return, trade_count);
    display_results(&opt, opt_net, opt_fees, gross_return, initial_capital);

    // =========================================================================
    // SCENARIO 2: Realistic (What Usually Happens)
    // =========================================================================
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("SCENARIO 2: Realistic (Typical Live Trading)");
    println!("─────────────────────────────────────────────────────────────────");
    println!();

    let real = FeeScenario {
        name: "Realistic".to_string(),
        taker_fee_pct: 0.0004,
        slippage_bps: 2.5, // More realistic slippage
        liquidity_factor: 1.0,
        execution_delay_sec: 0.5, // Typical execution delay
    };

    println!("Assumptions:");
    println!("  Taker fee: {:.2}%", real.taker_fee_pct * 100.0);
    println!("  Slippage: {:.1} bps (bid-ask + movement)", real.slippage_bps);
    println!("  Liquidity: Normal (occasional slippage on large orders)");
    println!("  Execution: Typical latency (0.5 sec)");
    println!();

    let (real_net, real_fees) = calculate_net_return(&real, position_size, gross_return, trade_count);
    display_results(&real, real_net, real_fees, gross_return, initial_capital);

    // =========================================================================
    // SCENARIO 3: Pessimistic (Volatile Market / Large Position)
    // =========================================================================
    println!("\n─────────────────────────────────────────────────────────────────");
    println!("SCENARIO 3: Pessimistic (Volatile Market / Bigger Position)");
    println!("─────────────────────────────────────────────────────────────────");
    println!();

    let pess = FeeScenario {
        name: "Pessimistic".to_string(),
        taker_fee_pct: 0.0004,
        slippage_bps: 5.0, // High slippage in volatile markets
        liquidity_factor: 0.7, // Poor fills on market impact
        execution_delay_sec: 2.0, // Network delay, queue position
    };

    println!("Assumptions:");
    println!("  Taker fee: {:.2}%", pess.taker_fee_pct * 100.0);
    println!("  Slippage: {:.1} bps (volatile market)", pess.slippage_bps);
    println!("  Liquidity: Poor (large position size relative to book)");
    println!("  Execution: Delay and market impact (2.0 sec)");
    println!();

    let (pess_net, pess_fees) = calculate_net_return(&pess, position_size, gross_return, trade_count);
    display_results(&pess, pess_net, pess_fees, gross_return, initial_capital);

    // =========================================================================
    // COMPARISON TABLE
    // =========================================================================
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Comparison Summary                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    println!("┌─────────────┬──────────┬──────────┬──────────┬────────────┐");
    println!("│ Scenario    │ Fees     │ Net $    │ Net %    │ vs Backtest│");
    println!("├─────────────┼──────────┼──────────┼──────────┼────────────┤");

    let (_, opt_fees_paid) = calculate_net_return(&opt, position_size, gross_return, trade_count);
    let opt_net_pct = ((gross_final - opt_fees_paid) / initial_capital - 1.0) * 100.0;
    println!(
        "│ Optimistic  │ ${:<7.0}│ ${:<8.0}│ {:<7.1}%│ +0.0%      │",
        opt_fees_paid,
        gross_final - opt_fees_paid,
        opt_net_pct
    );

    let real_net_pct = ((gross_final - real_fees) / initial_capital - 1.0) * 100.0;
    let real_degradation = (opt_net_pct - real_net_pct) / (opt_net_pct / 100.0).max(0.01);
    println!(
        "│ Realistic   │ ${:<7.0}│ ${:<8.0}│ {:<7.1}%│ {:.1}% ⚠️ │",
        real_fees,
        gross_final - real_fees,
        real_net_pct,
        (1.0 - real_net_pct / opt_net_pct) * 100.0
    );

    let pess_net_pct = ((gross_final - pess_fees) / initial_capital - 1.0) * 100.0;
    let pess_degradation = (opt_net_pct - pess_net_pct) / (opt_net_pct / 100.0).max(0.01);
    println!(
        "│ Pessimistic │ ${:<7.0}│ ${:<8.0}│ {:<7.1}%│ {:.1}% ❌│",
        pess_fees,
        gross_final - pess_fees,
        pess_net_pct,
        (1.0 - pess_net_pct / opt_net_pct) * 100.0
    );

    println!("└─────────────┴──────────┴──────────┴──────────┴────────────┘");
    println!();

    // =========================================================================
    // KEY INSIGHTS
    // =========================================================================
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                      Key Insights                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    println!("1. OPTIMISTIC vs REALISTIC");
    println!("   {:.1}% return degradation (0.02% → 0.07% effective fee)", (1.0 - real_net_pct / opt_net_pct) * 100.0);
    println!("   • This is NORMAL in live trading");
    println!("   • Slippage is worse than assumed");
    println!("   • Network latency causes worse fills");
    println!();

    println!("2. REALISTIC vs PESSIMISTIC");
    println!("   {:.1}% additional degradation (volatile conditions)", (1.0 - pess_net_pct / real_net_pct) * 100.0);
    println!("   • Happens regularly (not once per year)");
    println!("   • Larger positions = worse execution");
    println!("   • High volatility = larger spreads");
    println!();

    println!("3. CUMULATIVE IMPACT");
    let total_degradation = (opt_net_pct - pess_net_pct) / opt_net_pct * 100.0;
    println!("   From optimistic to pessimistic: {:.1}% degradation", total_degradation);
    println!("   {:.1}% return becomes {:.1}% return", opt_net_pct, pess_net_pct);
    if pess_net_pct < 3.0 {
        println!("   ❌ Strategy becomes UNPROFITABLE after risk adjustment");
    } else if pess_net_pct < 5.0 {
        println!("   ⚠️  Strategy BARELY VIABLE (too thin margin)");
    }
    println!();

    // =========================================================================
    // RECOMMENDATIONS
    // =========================================================================
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                   Deployment Recommendations                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    println!("DO NOT deploy a strategy if:");
    println!("  ❌ It doesn't survive pessimistic scenario");
    println!("  ❌ Margin between realistic and pessimistic < 2%");
    println!("  ❌ You haven't tested in live market conditions");
    println!("  ❌ Your position size significantly impacts liquidity");
    println!();

    println!("DO deploy a strategy if:");
    println!("  ✅ Realistic scenario return > 5% annually");
    println!("  ✅ Pessimistic scenario return > 2% annually");
    println!("  ✅ Win rate > 60% (margin of safety)");
    println!("  ✅ Avg winner > 3x avg loser");
    println!("  ✅ Paper trading matches backtest within ±5%");
    println!();

    println!("LIVE TRADING CHECKLIST:");
    println!("  1. Start with 10% of target capital");
    println!("  2. Monitor fees paid vs backtest assumptions");
    println!("  3. If realistic fees 30%+ higher → investigate (usually means slippage is worse)");
    println!("  4. If live return 15%+ below backtest → adjust position sizing or strategy");
    println!("  5. Kill switch: Close all if any trade loses 10% of capital");
    println!();

    Ok(())
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug)]
struct FeeScenario {
    name: String,
    taker_fee_pct: f64,
    slippage_bps: f64,
    liquidity_factor: f64,
    execution_delay_sec: f64,
}

// ============================================================================
// Functions
// ============================================================================

fn calculate_net_return(
    scenario: &FeeScenario,
    position_size: f64,
    gross_return: f64,
    trade_count: usize,
) -> (f64, f64) {
    // Fee per round-trip
    let taker_cost = position_size * scenario.taker_fee_pct * 2.0; // Entry + exit
    let slippage_cost = position_size * (scenario.slippage_bps / 10000.0);
    let liquidity_slippage = position_size * (scenario.slippage_bps / 10000.0) * (1.0 - scenario.liquidity_factor);
    
    let cost_per_rt = taker_cost + slippage_cost + liquidity_slippage;
    let total_fees = cost_per_rt * (trade_count as f64);

    (gross_return - total_fees, total_fees)
}

fn display_results(
    scenario: &FeeScenario,
    net_return: f64,
    total_fees: f64,
    gross_return: f64,
    initial_capital: f64,
) {
    let net_pct = (net_return / initial_capital) * 100.0;
    let gross_pct = (gross_return / initial_capital) * 100.0;
    let fee_drag = ((total_fees / gross_return) * 100.0).abs();
    let final_capital = initial_capital + net_return;

    println!("Results:");
    println!("  Gross return:  ${:+.0} ({:+.1}%)", gross_return, gross_pct);
    println!("  Total fees:    ${:+.0}", total_fees);
    println!("  Net return:    ${:+.0} ({:+.1}%)", net_return, net_pct);
    println!("  Final capital: ${:.0}", final_capital);
    println!("  Fee drag:      {:.1}% of returns", fee_drag);
    println!();

    // Profitability assessment
    if net_return <= 0.0 {
        println!("  ❌ UNPROFITABLE (losses exceed gains)");
    } else if net_pct < 2.0 {
        println!("  ⚠️  MARGINAL (after risk adjustment, barely viable)");
    } else if net_pct < 5.0 {
        println!("  ✅ VIABLE (acceptable return, thin margin)");
    } else {
        println!("  ✅ STRONG (good return, reasonable margin)");
    }
}
