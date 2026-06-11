//! Multi-Asset Trading System Demo
//!
//! This example demonstrates the comprehensive multi-asset support in kimsfinance,
//! showing how to work with different asset classes (stocks, futures, options, forex, crypto).

use chrono::{DateTime, NaiveDate, Utc};
use kimsfinance_core::assets::*;

fn main() {
    println!("=== kimsfinance Multi-Asset Trading System Demo ===\n");

    // ====================================================================
    // 1. EQUITY (STOCKS)
    // ====================================================================
    println!("1. EQUITY ASSETS");
    println!("{}", "=".repeat(60));

    let mut aapl = EquityAsset::new("AAPL", Exchange::Nasdaq);
    println!(
        "Created: {} on {}",
        aapl.symbol(),
        aapl.specification().exchange
    );
    println!("Tick size: ${}", aapl.tick_size());

    // Add corporate actions
    let split_date = NaiveDate::from_ymd_opt(2024, 6, 10).unwrap();
    aapl.add_corporate_action(CorporateAction::new_split(split_date, 4, 1));
    println!("Added 4-for-1 stock split on {}", split_date);

    // Price validation
    match aapl.validate_price(150.05) {
        Ok(price) => println!("Valid price: ${}", price),
        Err(e) => println!("Invalid price: {}", e),
    }

    // Corporate action adjustment
    let historical_price = 600.0; // Pre-split
    let adjusted_price = aapl.adjust_price(historical_price, split_date);
    println!(
        "Price ${} adjusted to ${} after 4-for-1 split",
        historical_price, adjusted_price
    );

    println!();

    // ====================================================================
    // 2. FUTURES
    // ====================================================================
    println!("2. FUTURES CONTRACTS");
    println!("{}", "=".repeat(60));

    let expiration = DateTime::from_timestamp(1742864400, 0).unwrap(); // March 2025
    let es = StandardFutures::es(FuturesMonthCode::March, 2025, expiration);
    println!("Created: {} (E-mini S&P 500)", es.symbol());
    println!("Contract multiplier: ${}", es.contract_multiplier());
    println!("Tick size: ${}", es.tick_size());
    println!("Tick value: ${}", es.specification().tick_value);

    // Calculate contract value
    let price = 5000.0;
    let value = es.calculate_value(price, 1.0).unwrap();
    println!("\n1 contract at {} = ${:.0}", price, value);

    // Calculate P&L in ticks
    let entry = 5000.0;
    let exit = 5010.0;
    let ticks = (exit - entry) / es.tick_size();
    let pnl = es.tick_pnl(ticks);
    println!(
        "Trade: {} -> {} = {:.0} ticks = ${:.2} P&L",
        entry, exit, ticks, pnl
    );

    // Margin requirements
    let (initial, maintenance) = es.required_margin(2);
    println!(
        "Margin for 2 contracts: ${:.0} initial, ${:.0} maintenance",
        initial, maintenance
    );

    println!();

    // ====================================================================
    // 3. OPTIONS
    // ====================================================================
    println!("3. OPTIONS CONTRACTS");
    println!("{}", "=".repeat(60));

    let option_expiration = DateTime::from_timestamp(1737072000, 0).unwrap();
    let mut call = OptionsContract::new(
        "AAPL",
        OptionType::Call,
        150.0,
        option_expiration,
        Exchange::CBOE,
    );
    println!(
        "Created: {} {} ${} strike",
        call.underlying(),
        call.option_type(),
        call.strike()
    );
    println!("OCC Symbol: {}", call.symbol());

    // Black-Scholes pricing
    let spot = 150.0;
    let volatility = 0.25; // 25%
    let risk_free_rate = 0.05; // 5%
    let time_to_expiry = 0.5; // 6 months
    let option_price = call.black_scholes_price(spot, volatility, risk_free_rate, time_to_expiry);
    println!("\nBlack-Scholes price: ${:.2}", option_price);

    // Calculate Greeks
    let greeks = call.calculate_greeks(spot, volatility, risk_free_rate, time_to_expiry);
    println!("Greeks:");
    println!("  Delta:  {:.4}", greeks.delta);
    println!("  Gamma:  {:.4}", greeks.gamma);
    println!("  Theta:  {:.4} (per day)", greeks.theta);
    println!("  Vega:   {:.4} (per 1%)", greeks.vega);
    println!("  Rho:    {:.4} (per 1%)", greeks.rho);

    // Intrinsic and time value
    let intrinsic = call.intrinsic_value(spot);
    let time_value = call.time_value(spot, option_price);
    println!("\nIntrinsic value: ${:.2}", intrinsic);
    println!("Time value: ${:.2}", time_value);

    // Moneyness
    println!("Is ITM: {}", call.is_itm(spot));
    println!("Is ATM: {}", call.is_atm(spot, 0.01));
    println!("Is OTM: {}", call.is_otm(spot));

    println!();

    // ====================================================================
    // 4. FOREX
    // ====================================================================
    println!("4. FOREX PAIRS");
    println!("{}", "=".repeat(60));

    let eurusd = StandardForexPairs::eurusd();
    println!("Created: {}", eurusd.symbol());
    println!("Pip size: {}", eurusd.pip_size());
    println!("Lot size: {:.0} units", eurusd.lot_size());

    // Calculate pip P&L
    let entry_rate = 1.1000;
    let exit_rate = 1.1050;
    let pips = eurusd.calculate_pips(entry_rate, exit_rate, true);
    println!(
        "\nLong trade: {} -> {} = {:.1} pips profit",
        entry_rate, exit_rate, pips
    );

    // Position value
    let lots = 1.0;
    let position_value = eurusd.calculate_position_value(entry_rate, lots);
    println!(
        "Position value: ${:.0} ({} lot at {})",
        position_value, lots, entry_rate
    );

    // Pip value calculation
    let pip_value = eurusd.calculate_pip_value(1.1000, 1.0);
    println!("Pip value: ${:.2} per standard lot", pip_value);

    println!();

    // ====================================================================
    // 5. CRYPTOCURRENCY
    // ====================================================================
    println!("5. CRYPTOCURRENCY");
    println!("{}", "=".repeat(60));

    let btcusd = StandardCryptoPairs::btcusd(Exchange::Binance);
    println!(
        "Created: {} on {}",
        btcusd.symbol(),
        btcusd.specification().exchange
    );
    println!("Precision: {} decimals", btcusd.precision());

    // Satoshi conversion
    let btc_amount = 0.12345678;
    let satoshis = btcusd.to_satoshis(btc_amount);
    println!("\n{} BTC = {} satoshis", btc_amount, satoshis);
    println!("Back to BTC: {}", btcusd.from_satoshis(satoshis));

    // Trading fees
    let trade_value = 50_000.0;
    let maker_fee = btcusd.calculate_fee(trade_value, true);
    let taker_fee = btcusd.calculate_fee(trade_value, false);
    println!("\nTrade value: ${:.2}", trade_value);
    println!("Maker fee: ${:.2}", maker_fee);
    println!("Taker fee: ${:.2}", taker_fee);

    // 24/7 trading
    let now = Utc::now();
    println!(
        "Market open now: {} (crypto trades 24/7)",
        btcusd.is_market_open(now)
    );

    println!();

    // ====================================================================
    // 6. CFD (Contract for Difference)
    // ====================================================================
    println!("6. CFD CONTRACTS");
    println!("{}", "=".repeat(60));

    let cfd = CFDContract::new("AAPL", Exchange::OTC, 10.0, 0.01).with_financing_rate(0.05); // 5% annual

    println!("Created: {}", cfd.symbol());
    println!("Max leverage: {}x", cfd.max_leverage());
    println!("Margin requirement: {}%", cfd.margin_requirement() * 100.0);

    // Margin calculation
    let cfd_price = 150.0;
    let cfd_quantity = 100.0;
    let required_margin = cfd.calculate_margin(cfd_price, cfd_quantity);
    println!(
        "\n{} shares at ${} = ${:.0} margin required",
        cfd_quantity, cfd_price, required_margin
    );

    // Overnight financing
    let position_value = cfd_price * cfd_quantity;
    let overnight_charge = cfd.calculate_overnight_financing(position_value, 1.0);
    println!(
        "Overnight financing charge: ${:.2} per day",
        overnight_charge
    );

    // Max position size
    let capital = 10_000.0;
    let max_size = cfd.max_position_size(capital, cfd_price);
    println!(
        "With ${:.0} capital: max {:.0} shares ({}x leverage)",
        capital,
        max_size,
        cfd.max_leverage()
    );

    println!();

    // ====================================================================
    // 7. MARKET INDEX
    // ====================================================================
    println!("7. MARKET INDICES");
    println!("{}", "=".repeat(60));

    let sp500 = StandardIndices::sp500();
    println!("Created: {}", sp500.symbol());
    println!("Methodology: {:?}", sp500.methodology());

    let djia = StandardIndices::djia();
    println!("\nCreated: {}", djia.symbol());
    println!("Methodology: {:?}", djia.methodology());
    println!("Divisor: {:.3}", djia.divisor());

    // Note: Actual index calculation would require constituent prices
    println!("\n(Index calculation requires constituent price data)");

    println!();

    // ====================================================================
    // SUMMARY
    // ====================================================================
    println!("=== SUMMARY ===");
    println!("kimsfinance supports 7 asset classes:");
    println!("  ✓ Equity (stocks) with corporate actions");
    println!("  ✓ Futures with contract specifications");
    println!("  ✓ Options with Black-Scholes pricing and Greeks");
    println!("  ✓ Forex with pip calculations");
    println!("  ✓ Cryptocurrency with satoshi precision");
    println!("  ✓ CFD with leverage and margin");
    println!("  ✓ Market indices with different methodologies");
}
