//! Paper Trading Scanner for Bull Put Spreads
//!
//! Daily scanner that identifies bull put spread opportunities
//! following the proven strategy parameters (266% ROC, 67% win rate).
//!
//! Run daily before market open:
//! ```bash
//! cargo run --release --features data-downloaders --example paper_trading_scanner
//! cargo run --release --features data-downloaders --example paper_trading_scanner --json
//! ```
//!
//! Outputs:
//! - Trade opportunities with entry criteria
//! - Strike prices, credit, margin required
//! - Position sizing recommendations
//! - Risk metrics

use chrono::{Local, NaiveDate};
use kimsfinance_core::strategy::*;
use serde::{Deserialize, Serialize};
use serde_json;

const INITIAL_CAPITAL: f64 = 10_000.0;
const MAX_RISK_PER_TRADE_PCT: f64 = 5.0;
const MAX_MARGIN_PCT: f64 = 50.0;

#[derive(Debug, Serialize, Deserialize)]
struct TradeOpportunity {
    symbol: String,
    short_strike: f64,
    long_strike: f64,
    expiration: String, // Changed to String for JSON serialization
    dte: i32,
    short_delta: f64,
    credit: f64,
    width: f64,
    max_risk: f64,
    margin_required: f64,
    risk_pct: f64,
    margin_pct: f64,
    profit_target: f64,
    stop_loss: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for --json flag
    let json_output = std::env::args().any(|arg| arg == "--json");

    if !json_output {
        println!("=== Bull Put Spread Paper Trading Scanner ===");
        println!("Strategy: Proven 266% ROC | 67% Win Rate | Sharpe 1.40\n");
    }

    let today = Local::now().naive_local().date();

    if !json_output {
        println!("Scan Date: {}", today);
        println!("Account Capital: ${:.2}", INITIAL_CAPITAL);
        println!("Max Risk/Trade: {:.1}%", MAX_RISK_PER_TRADE_PCT);
        println!("Max Margin: {:.1}%\n", MAX_MARGIN_PCT);
    }

    // Configuration
    let data_dir = "data/yfinance/options_historical";
    let spot_dir = "../data/yfinance/ohlcv";
    let symbols = vec!["SPY", "QQQ", "AAPL", "TSLA"];

    // Initialize loaders
    if !json_output {
        println!("Loading market data...");
    }
    let mut loader = OptionsDataLoader::new(data_dir)?;
    let mut spot_loader = SpotDataLoader::new(spot_dir)?;

    // Create strategy with proven parameters
    let params = default_bull_put_params();
    let strategy = BullPutSpread::new(params.clone());

    // Create regime detector
    let regime_detector = RegimeDetector::default();

    if !json_output {
        println!("\n=== Strategy Parameters (Proven Profitable) ===");
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
        println!("  Max Hold: {} days", params.max_hold_days.unwrap_or(0));
        println!("  Min Credit: ${:.2}\n", params.min_credit.unwrap_or(0.0));
    }

    // Scan each symbol for opportunities
    let mut all_opportunities = Vec::new();

    for symbol in &symbols {
        if !json_output {
            println!("--- Scanning {} ---", symbol);
        }

        // Get spot price
        let spot_price = match spot_loader.get_spot_price(symbol, today) {
            Ok(price) => price,
            Err(e) => {
                if !json_output {
                    println!("  ⚠️  No spot price: {:?}", e);
                }
                continue;
            }
        };
        if !json_output {
            println!("  Spot Price: ${:.2}", spot_price);
        }

        // Detect market regime
        let regime = match regime_detector.detect_regime(&mut spot_loader, symbol, today) {
            Ok(r) => r,
            Err(e) => {
                if !json_output {
                    println!("  ⚠️  Regime detection failed: {:?}", e);
                }
                MarketRegime::Sideways // Default fallback
            }
        };
        if !json_output {
            println!("  Market Regime: {}", regime);
        }

        // Check if should trade in this regime
        if !should_trade_in_regime(regime) {
            if !json_output {
                println!("  ❌ Skipping: Unfavorable regime ({})", regime);
                println!();
            }
            continue;
        }

        // Load options data for today
        let options = match loader.load_chain(symbol, today) {
            Ok(opts) => opts,
            Err(e) => {
                if !json_output {
                    println!("  ⚠️  No options data: {:?}", e);
                }
                continue;
            }
        };

        // Filter only puts
        let puts: Vec<_> = options
            .into_iter()
            .filter(|opt| opt.option_type == OptionType::Put)
            .collect();

        if !json_output {
            println!("  Available Puts: {}", puts.len());
        }

        // Find bull put spread candidates
        let candidates = strategy.find_candidates(&puts, spot_price);
        if !json_output {
            println!("  Candidates Found: {}", candidates.len());
        }

        // Process each candidate
        for (short_put, long_put) in candidates {
            let credit = short_put.mid_price() - long_put.mid_price();
            let width = short_put.strike - long_put.strike;
            let max_risk = width - credit;
            let margin_required = width * 100.0; // Per contract

            // Calculate risk and margin percentages
            let risk_pct = (max_risk * 100.0) / INITIAL_CAPITAL * 100.0;
            let margin_pct = margin_required / INITIAL_CAPITAL * 100.0;

            // Check risk limits
            if risk_pct > MAX_RISK_PER_TRADE_PCT {
                continue; // Skip if exceeds risk limit
            }

            if margin_pct > MAX_MARGIN_PCT {
                continue; // Skip if exceeds margin limit
            }

            // Calculate profit target and stop loss prices
            let profit_target_credit = credit * 0.5; // 50% of max profit
            let stop_loss_debit = credit * 2.0; // 200% loss

            let opportunity = TradeOpportunity {
                symbol: symbol.to_string(),
                short_strike: short_put.strike,
                long_strike: long_put.strike,
                expiration: short_put.expiration.format("%Y%m%d").to_string(),
                dte: short_put.dte,
                short_delta: short_put.delta.unwrap_or(0.0),
                credit,
                width,
                max_risk,
                margin_required,
                risk_pct,
                margin_pct,
                profit_target: profit_target_credit,
                stop_loss: stop_loss_debit,
            };

            all_opportunities.push(opportunity);
        }

        if !json_output {
            println!();
        }
    }

    // Display opportunities ranked by risk-reward
    all_opportunities.sort_by(|a, b| {
        let score_a = (a.credit / a.max_risk) * (1.0 - a.risk_pct / 100.0);
        let score_b = (b.credit / b.max_risk) * (1.0 - b.risk_pct / 100.0);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // If JSON output requested, print JSON and exit
    if json_output {
        let json = serde_json::to_string_pretty(&all_opportunities)?;
        println!("{}", json);
        return Ok(());
    }

    // Otherwise, print human-readable output
    println!("=== TRADE OPPORTUNITIES (Ranked Best to Worst) ===\n");

    if all_opportunities.is_empty() {
        println!("❌ No opportunities found today matching criteria.");
        println!("\nPossible reasons:");
        println!("  1. Market regime unfavorable (bear market)");
        println!("  2. No options meeting DTE/delta criteria");
        println!("  3. Insufficient credit for minimum requirement");
        println!("  4. All candidates exceed risk limits\n");
        return Ok(());
    }

    for (i, opp) in all_opportunities.iter().enumerate() {
        println!(
            "┌─ Opportunity #{} - {} ─────────────────────────",
            i + 1,
            opp.symbol
        );
        println!("│");
        println!("│ POSITION DETAILS:");
        println!(
            "│   Short PUT: ${:.2} (delta: {:.3})",
            opp.short_strike, opp.short_delta
        );
        println!("│   Long PUT:  ${:.2}", opp.long_strike);
        println!("│   Width:     ${:.2}", opp.width);
        println!("│   DTE:       {} days", opp.dte);
        println!("│   Expiration: {}", opp.expiration);
        println!("│");
        println!("│ FINANCIALS:");
        println!(
            "│   Credit Received:    ${:.2} (${:.0} per contract)",
            opp.credit,
            opp.credit * 100.0
        );
        println!(
            "│   Max Profit:         ${:.0} per contract",
            opp.credit * 100.0
        );
        println!(
            "│   Max Risk:           ${:.0} per contract",
            opp.max_risk * 100.0
        );
        println!(
            "│   Margin Required:    ${:.0} per contract",
            opp.margin_required
        );
        println!("│");
        println!("│ RISK METRICS:");
        println!("│   Risk/Capital:       {:.2}%", opp.risk_pct);
        println!("│   Margin/Capital:     {:.2}%", opp.margin_pct);
        println!(
            "│   Credit/Width:       {:.1}%",
            (opp.credit / opp.width) * 100.0
        );
        println!("│");
        println!("│ EXIT TARGETS:");
        println!(
            "│   Profit Target (50%): Close at ${:.2} debit",
            opp.profit_target
        );
        println!(
            "│   Stop Loss (200%):    Close at ${:.2} debit",
            opp.stop_loss
        );
        println!(
            "│   Max Hold:            {} days (from entry)",
            params.max_hold_days.unwrap_or(42)
        );
        println!("│");
        println!("│ IBKR ORDER ENTRY:");
        println!("│   1. Create vertical spread");
        println!(
            "│   2. Sell 1 {} {} PUT ${:.2}",
            opp.expiration, opp.symbol, opp.short_strike
        );
        println!(
            "│   3. Buy 1 {} {} PUT ${:.2}",
            opp.expiration, opp.symbol, opp.long_strike
        );
        println!("│   4. Limit Order: ${:.2} CREDIT (or better)", opp.credit);
        println!("│   5. Time in Force: DAY ORDER");
        println!("│");
        println!("│ ALERTS TO SET:");
        println!(
            "│   - Profit target: Spread value drops to ${:.2}",
            opp.profit_target
        );
        println!(
            "│   - Stop loss: Spread value rises to ${:.2}",
            opp.stop_loss
        );
        println!(
            "│   - Time exit: {} days from entry",
            params.max_hold_days.unwrap_or(42)
        );
        println!("│");

        // Risk warnings
        let mut warnings = Vec::new();
        if opp.risk_pct > 3.0 {
            warnings.push(format!("⚠️  High risk: {:.1}%", opp.risk_pct));
        }
        if opp.short_delta.abs() > 0.30 {
            warnings.push(format!("⚠️  Aggressive delta: {:.3}", opp.short_delta));
        }
        if opp.dte < 30 {
            warnings.push(format!("⚠️  Short DTE: {} days", opp.dte));
        }

        if !warnings.is_empty() {
            println!("│ WARNINGS:");
            for warning in warnings {
                println!("│   {}", warning);
            }
            println!("│");
        }

        println!("└─────────────────────────────────────────────────\n");
    }

    // Summary statistics
    println!("=== SCAN SUMMARY ===");
    println!("Total Opportunities: {}", all_opportunities.len());
    println!("Symbols Scanned: {}", symbols.len());

    let avg_credit: f64 =
        all_opportunities.iter().map(|o| o.credit).sum::<f64>() / all_opportunities.len() as f64;
    let avg_risk_pct: f64 =
        all_opportunities.iter().map(|o| o.risk_pct).sum::<f64>() / all_opportunities.len() as f64;

    println!("Average Credit: ${:.2}", avg_credit);
    println!("Average Risk: {:.2}%", avg_risk_pct);
    println!();

    // Next steps
    println!("=== NEXT STEPS ===");
    println!("1. Review opportunities above (ranked best to worst)");
    println!("2. Log into IBKR Paper Trading");
    println!("3. Enter orders using exact strikes/expirations shown");
    println!("4. Set profit target and stop loss alerts");
    println!("5. Record entry in trade log (see PAPER_TRADING_GUIDE.md)");
    println!("6. Monitor daily (mid-day check for exits)");
    println!();
    println!("Documentation: rust/docs/PAPER_TRADING_GUIDE.md");
    println!("Trade Log Template: See guide section 'Trade Tracking Spreadsheet'");
    println!();

    println!("✅ Scan complete. Good luck trading!");

    Ok(())
}
