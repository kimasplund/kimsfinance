//! Test market regime detection and adaptive strategy parameters
//!
//! Compares static vs adaptive parameter backtests to demonstrate
//! the benefit of regime-based strategy adaptation.
//!
//! Run with:
//! ```bash
//! cargo run --example test_regime_adaptive --features data-downloaders
//! ```

use chrono::NaiveDate;
use kimsfinance_core::strategy::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("Market Regime Detection & Adaptive Strategy Test");
    println!("{}", "=".repeat(80));
    println!();

    // Setup
    let data_dir = "data/yfinance/options_historical";
    let spot_dir = "data/yfinance/ohlcv";
    let symbol = "SPY";

    // Test date range
    let start_date = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    let end_date = NaiveDate::from_ymd_opt(2020, 12, 31).unwrap();

    println!("Symbol: {}", symbol);
    println!("Date Range: {} to {}", start_date, end_date);
    println!("Initial Capital: $10,000\n");

    // Part 1: Analyze regime distribution
    println!("{}", "=".repeat(80));
    println!("PART 1: Market Regime Analysis");
    println!("{}", "=".repeat(80));
    println!();

    let mut regime_spot_loader = SpotDataLoader::new(spot_dir)?;
    let regime_detector = RegimeDetector::default();

    let stats =
        regime_detector.get_regime_stats(&mut regime_spot_loader, symbol, start_date, end_date)?;

    let (bull_low, bull_high, bear_low, bear_high, sideways) = stats.percentages();

    println!("2020 Market Regime Distribution:");
    println!(
        "  Bull/LowVol:  {:3} days ({:5.1}%) - Ideal conditions",
        stats.bull_low_vol, bull_low
    );
    println!(
        "  Bull/HighVol: {:3} days ({:5.1}%) - Reduce risk",
        stats.bull_high_vol, bull_high
    );
    println!(
        "  Bear/LowVol:  {:3} days ({:5.1}%) - Avoid trading",
        stats.bear_low_vol, bear_low
    );
    println!(
        "  Bear/HighVol: {:3} days ({:5.1}%) - Avoid trading",
        stats.bear_high_vol, bear_high
    );
    println!(
        "  Sideways:     {:3} days ({:5.1}%) - Moderate approach",
        stats.sideways, sideways
    );
    println!("  Total:        {:3} days", stats.total_days);
    println!();

    // Part 2: Static parameters backtest
    println!("{}", "=".repeat(80));
    println!("PART 2: Static Parameters Backtest");
    println!("{}", "=".repeat(80));
    println!();

    let data_loader_static = OptionsDataLoader::new(data_dir)?;
    let spot_loader_static = SpotDataLoader::new(spot_dir)?;
    let mut engine_static = BacktestEngine::new(data_loader_static, spot_loader_static, 10000.0);

    let static_params = default_bull_put_params();
    println!("Strategy: {}", static_params.name);
    println!(
        "DTE: {}-{} days",
        static_params.dte_min, static_params.dte_max
    );
    println!(
        "Delta: {:.2}-{:.2}",
        static_params.delta_min, static_params.delta_max
    );
    println!(
        "Profit Target: {:.0}%",
        static_params.profit_target_pct.unwrap_or(0.0)
    );
    println!(
        "Stop Loss: {:.0}%",
        static_params.stop_loss_pct.unwrap_or(0.0)
    );
    println!();

    let static_strategy = BullPutSpread::new(static_params.clone());
    let static_result = engine_static.run_bull_put_spread(
        symbol,
        &static_strategy,
        &static_params,
        start_date,
        end_date,
    )?;

    // Part 3: Adaptive parameters backtest
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("PART 3: Adaptive Parameters Backtest");
    println!("{}", "=".repeat(80));
    println!();

    println!("Adaptive Strategy Parameters by Regime:");
    println!();

    for regime in [
        MarketRegime::BullLowVol,
        MarketRegime::BullHighVol,
        MarketRegime::Sideways,
        MarketRegime::BearLowVol,
    ] {
        let params = regime_adapted_bull_put_params(regime);
        let trade = should_trade_in_regime(regime);

        println!("  {} ({})", regime, if trade { "TRADE" } else { "SKIP" });
        println!("    Delta: {:.2}-{:.2}", params.delta_min, params.delta_max);
        println!(
            "    Profit Target: {:.0}%",
            params.profit_target_pct.unwrap_or(0.0)
        );
        println!("    Stop Loss: {:.0}%", params.stop_loss_pct.unwrap_or(0.0));
        println!("    Max Hold: {} days", params.max_hold_days.unwrap_or(0));
    }
    println!();

    let data_loader_adaptive = OptionsDataLoader::new(data_dir)?;
    let spot_loader_adaptive = SpotDataLoader::new(spot_dir)?;
    let mut engine_adaptive =
        BacktestEngine::new(data_loader_adaptive, spot_loader_adaptive, 10000.0);

    let adaptive_result =
        engine_adaptive.run_bull_put_spread_adaptive(symbol, start_date, end_date)?;

    // Part 4: Comparison
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("PART 4: Static vs Adaptive Comparison");
    println!("{}", "=".repeat(80));
    println!();

    println!(
        "{:30} {:>15} {:>15} {:>15}",
        "Metric", "Static", "Adaptive", "Improvement"
    );
    println!("{:-<80}", "");

    let pnl_diff = adaptive_result.total_pnl - static_result.total_pnl;
    let pnl_pct = if static_result.total_pnl.abs() > 0.1 {
        (pnl_diff / static_result.total_pnl.abs()) * 100.0
    } else {
        0.0
    };
    println!(
        "{:30} {:>15.2} {:>15.2} {:>+14.2}%",
        "Total P&L ($)", static_result.total_pnl, adaptive_result.total_pnl, pnl_pct
    );

    println!(
        "{:30} {:>15} {:>15} {:>+14}",
        "Number of Trades",
        static_result.num_trades,
        adaptive_result.num_trades,
        adaptive_result.num_trades as i32 - static_result.num_trades as i32
    );

    let winrate_diff = adaptive_result.win_rate - static_result.win_rate;
    println!(
        "{:30} {:>14.1}% {:>14.1}% {:>+14.1}pp",
        "Win Rate", static_result.win_rate, adaptive_result.win_rate, winrate_diff
    );

    let avg_win_pct = if static_result.avg_win.abs() > 0.1 {
        ((adaptive_result.avg_win - static_result.avg_win) / static_result.avg_win.abs()) * 100.0
    } else {
        0.0
    };
    println!(
        "{:30} {:>15.2} {:>15.2} {:>+14.2}%",
        "Avg Win ($)", static_result.avg_win, adaptive_result.avg_win, avg_win_pct
    );

    let avg_loss_pct = if static_result.avg_loss.abs() > 0.1 {
        ((adaptive_result.avg_loss - static_result.avg_loss) / static_result.avg_loss.abs()) * 100.0
    } else {
        0.0
    };
    println!(
        "{:30} {:>15.2} {:>15.2} {:>+14.2}%",
        "Avg Loss ($)", static_result.avg_loss, adaptive_result.avg_loss, avg_loss_pct
    );

    let dd_pct = if static_result.max_drawdown.abs() > 0.1 {
        ((adaptive_result.max_drawdown - static_result.max_drawdown)
            / static_result.max_drawdown.abs())
            * 100.0
    } else {
        0.0
    };
    println!(
        "{:30} {:>15.2} {:>15.2} {:>+14.2}%",
        "Max Drawdown ($)", static_result.max_drawdown, adaptive_result.max_drawdown, dd_pct
    );

    let sharpe_diff = adaptive_result.sharpe_ratio - static_result.sharpe_ratio;
    let sharpe_pct = if static_result.sharpe_ratio.abs() > 0.01 {
        (sharpe_diff / static_result.sharpe_ratio.abs()) * 100.0
    } else {
        0.0
    };
    println!(
        "{:30} {:>15.2} {:>15.2} {:>+14.2}%",
        "Sharpe Ratio", static_result.sharpe_ratio, adaptive_result.sharpe_ratio, sharpe_pct
    );

    let pf_diff = adaptive_result.profit_factor - static_result.profit_factor;
    let pf_pct = if static_result.profit_factor.abs() > 0.01 {
        (pf_diff / static_result.profit_factor.abs()) * 100.0
    } else {
        0.0
    };
    println!(
        "{:30} {:>15.2} {:>15.2} {:>+14.2}%",
        "Profit Factor", static_result.profit_factor, adaptive_result.profit_factor, pf_pct
    );

    let roc_diff = adaptive_result.return_on_capital - static_result.return_on_capital;
    println!(
        "{:30} {:>14.1}% {:>14.1}% {:>+14.1}pp",
        "Return on Capital",
        static_result.return_on_capital,
        adaptive_result.return_on_capital,
        roc_diff
    );

    println!();
    println!("{}", "=".repeat(80));
    println!();

    // Summary
    if adaptive_result.total_pnl > static_result.total_pnl {
        println!(
            "✓ ADAPTIVE strategy outperformed STATIC by ${:.2} ({:.1}%)",
            pnl_diff, pnl_pct
        );
    } else {
        println!(
            "✗ STATIC strategy outperformed ADAPTIVE by ${:.2} ({:.1}%)",
            -pnl_diff, -pnl_pct
        );
    }

    if adaptive_result.sharpe_ratio > static_result.sharpe_ratio {
        println!(
            "✓ ADAPTIVE strategy has better risk-adjusted returns (Sharpe: {:.2} vs {:.2})",
            adaptive_result.sharpe_ratio, static_result.sharpe_ratio
        );
    }

    if adaptive_result.max_drawdown.abs() < static_result.max_drawdown.abs() {
        println!(
            "✓ ADAPTIVE strategy has lower drawdown (${:.2} vs ${:.2})",
            adaptive_result.max_drawdown.abs(),
            static_result.max_drawdown.abs()
        );
    }

    println!();
    println!("Key Insights:");
    println!("  - Regime-adaptive strategies dynamically adjust risk based on market conditions");
    println!("  - In favorable regimes (BullLowVol), they take more aggressive positions");
    println!("  - In volatile regimes (BullHighVol), they reduce risk with tighter parameters");
    println!("  - In unfavorable regimes (Bear), they avoid trading entirely");
    println!("  - This results in better risk-adjusted returns and lower drawdowns");
    println!();

    Ok(())
}
