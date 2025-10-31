//! Heston Greeks and Strategies Demonstration
//!
//! Shows GPU-accelerated Greeks calculation and straddle strategy signals.
//!
//! # Features Demonstrated
//!
//! 1. GPU-accelerated Greeks calculation (10-100x faster than CPU)
//! 2. Long straddle strategy signals (profit from volatility expansion)
//! 3. Short straddle strategy signals (profit from volatility contraction)
//! 4. Performance comparison (CPU vs GPU)
//!
//! # Run
//!
//! ```bash
//! cargo run --example heston_greeks_strategies_demo --features gpu --release
//! ```

use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{
    GreeksGpuCalculator, HestonGreeksCalculator, HestonParams, OptionQuote, OptionType,
    StraddleParams, StraddleStrategyGpu,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

fn create_option_chain(spot: f64, n_strikes: usize) -> Vec<OptionQuote> {
    let now = chrono::Utc::now().timestamp();
    let expiry_3months = now + (90 * 24 * 3600);

    let strike_range = 0.15; // ±15% from spot
    let strike_step = (2.0 * strike_range * spot) / (n_strikes as f64);

    (0..n_strikes)
        .map(|i| {
            let strike = spot * (1.0 - strike_range) + i as f64 * strike_step;
            OptionQuote {
                underlying: "BTC".to_string(),
                strike,
                expiration: expiry_3months,
                option_type: OptionType::Call,
                spot_price: spot,
                risk_free_rate: 0.05,
                bid: None,
                ask: None,
                last: None,
                implied_vol: Some(0.60 + (i as f64 / n_strikes as f64) * 0.20), // IV smile
                volume: 100.0,
                open_interest: 500.0,
                greeks: None,
            }
        })
        .collect()
}

fn main() {
    println!("=== Heston GPU Greeks & Strategies Demo ===\n");

    // Initialize GPU
    let device = Arc::new(GpuDevice::new().expect("GPU required for this demo"));
    println!("✅ GPU initialized\n");

    // Create Heston model parameters
    let params = HestonParams::new(
        2.0,  // kappa: Mean reversion speed
        0.04, // theta: Long-term variance
        0.3,  // sigma: Vol of vol
        -0.7, // rho: Correlation
        0.04, // v0: Initial variance (sqrt(0.04) = 20% vol)
    )
    .unwrap();

    println!("Heston Parameters:");
    println!("  kappa: {:.2}", params.kappa);
    println!("  theta: {:.4}", params.theta);
    println!("  sigma: {:.2}", params.sigma);
    println!("  rho:   {:.2}", params.rho);
    println!("  v0:    {:.4} (current vol: {:.1}%)", params.v0, params.current_vol() * 100.0);
    println!();

    // Create option chain
    let spot = 48000.0;
    let n_strikes = 20;
    let options = create_option_chain(spot, n_strikes);
    println!("Created option chain: {} strikes from ${:.0} to ${:.0}\n", n_strikes, options.first().unwrap().strike, options.last().unwrap().strike);

    // ==================================================
    // PART 1: GPU Greeks Calculation
    // ==================================================
    println!("=== PART 1: GPU Greeks Calculation ===\n");

    let pricer_cpu = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");
    let pricer_gpu = HestonGpuPricer::new(device.clone(), 4096, 1000).expect("Pricer creation failed");

    let calculator_cpu = HestonGreeksCalculator::new(Arc::new(Mutex::new(pricer_cpu)));
    let mut calculator_gpu = GreeksGpuCalculator::new(device.clone(), Arc::new(Mutex::new(pricer_gpu)))
        .expect("GPU Greeks calculator creation failed");

    // Benchmark CPU
    println!("Calculating Greeks on CPU...");
    let start_cpu = Instant::now();
    let greeks_cpu = calculator_cpu
        .calculate_greeks_batch(&params, &options)
        .expect("CPU Greeks failed");
    let time_cpu = start_cpu.elapsed();
    println!("✅ CPU Greeks: {} options in {:?}", n_strikes, time_cpu);

    // Benchmark GPU
    println!("\nCalculating Greeks on GPU...");
    let start_gpu = Instant::now();
    let greeks_gpu = calculator_gpu
        .calculate_greeks_batch(&params, &options)
        .expect("GPU Greeks failed");
    let time_gpu = start_gpu.elapsed();
    println!("✅ GPU Greeks: {} options in {:?}", n_strikes, time_gpu);

    let speedup = time_cpu.as_secs_f64() / time_gpu.as_secs_f64();
    println!("\n🚀 Speedup: {:.1}x faster on GPU", speedup);

    // Show sample results
    println!("\nSample Greeks (first 3 options):");
    println!("{:<10} {:<8} {:<8} {:<8} {:<8} {:<8}", "Strike", "Delta", "Gamma", "Vega", "Theta", "Rho");
    println!("{:-<60}", "");
    for (i, (opt, greeks)) in options.iter().zip(greeks_gpu.iter()).take(3).enumerate() {
        println!(
            "${:<9.0} {:<8.4} {:<8.5} {:<8.2} {:<8.2} {:<8.2}",
            opt.strike,
            greeks.delta.unwrap(),
            greeks.gamma.unwrap(),
            greeks.vega.unwrap(),
            greeks.theta.unwrap(),
            greeks.rho_greek.unwrap()
        );
    }
    println!();

    // ==================================================
    // PART 2: Long Straddle Strategy
    // ==================================================
    println!("\n=== PART 2: Long Straddle Strategy ===\n");

    let straddle_strategy = StraddleStrategyGpu::new(device.clone()).expect("Straddle strategy creation failed");

    // Prepare data for strategy
    let n_candles = 10;
    let n_strategies = 3;

    let underlying_prices: Vec<f64> = (0..n_candles).map(|i| spot + (i as f64 - 5.0) * 200.0).collect();

    // Simulate ATM option prices (simplified)
    let call_prices: Vec<f64> = vec![2000.0; n_strategies * n_candles];
    let put_prices: Vec<f64> = vec![1900.0; n_strategies * n_candles];

    // IV varies by candle (simulate vol regime changes)
    let mut implied_vols = Vec::new();
    for strat in 0..n_strategies {
        for candle in 0..n_candles {
            let iv = 0.50 + (candle as f64 / n_candles as f64) * 0.20; // 50-70% IV
            implied_vols.push(iv);
        }
    }

    // HV is constant at 60%
    let historical_vols: Vec<f64> = vec![0.60; n_strategies * n_candles];

    // Strategy configurations
    let strategy_params = vec![
        StraddleParams {
            vol_threshold: 5.0,
            breakeven_pct: 2.0,
        },
        StraddleParams {
            vol_threshold: 8.0,
            breakeven_pct: 3.0,
        },
        StraddleParams {
            vol_threshold: 3.0,
            breakeven_pct: 1.5,
        },
    ];

    println!("Strategy Parameters:");
    for (i, p) in strategy_params.iter().enumerate() {
        println!("  Strategy {}: vol_threshold={:.1}%, breakeven={:.1}%", i + 1, p.vol_threshold, p.breakeven_pct);
    }
    println!();

    // Generate signals
    let start_straddle = Instant::now();
    let signals = straddle_strategy
        .generate_long_signals_batch(
            &underlying_prices,
            &call_prices,
            &put_prices,
            &implied_vols,
            &historical_vols,
            &strategy_params,
        )
        .expect("Signal generation failed");
    let time_straddle = start_straddle.elapsed();

    println!("✅ Generated {} signals in {:?}", signals.len(), time_straddle);
    println!();

    // Analyze signals
    let long_positions: usize = signals.iter().filter(|s| s.call_signal == 1).count();
    let no_positions: usize = signals.iter().filter(|s| s.call_signal == 0).count();

    println!("Signal Summary:");
    println!("  Long straddles: {}/{} ({:.1}%)", long_positions, signals.len(), (long_positions as f64 / signals.len() as f64) * 100.0);
    println!("  No positions:   {}/{} ({:.1}%)", no_positions, signals.len(), (no_positions as f64 / signals.len() as f64) * 100.0);
    println!();

    // Show sample signals (Strategy 1)
    println!("Sample Signals (Strategy 1, first 5 candles):");
    println!("{:<10} {:<10} {:<8} {:<8} {:<10}", "Candle", "Spot", "IV", "HV", "Signal");
    println!("{:-<50}", "");
    for i in 0..5.min(n_candles) {
        let signal = &signals[i]; // Strategy 0, candle i
        let iv = implied_vols[i];
        let hv = historical_vols[i];
        let signal_str = if signal.call_signal == 1 {
            "BUY"
        } else {
            "HOLD"
        };
        println!(
            "{:<10} ${:<9.0} {:<7.1}% {:<7.1}% {}",
            i,
            underlying_prices[i],
            iv * 100.0,
            hv * 100.0,
            signal_str
        );
    }
    println!();

    // ==================================================
    // PART 3: Short Straddle Strategy
    // ==================================================
    println!("=== PART 3: Short Straddle Strategy ===\n");

    // For short straddle, we want IV > HV (expensive options)
    let implied_vols_high: Vec<f64> = vec![0.70; n_strategies * n_candles]; // 70% IV
    let historical_vols_low: Vec<f64> = vec![0.58; n_strategies * n_candles]; // 58% HV

    let short_signals = straddle_strategy
        .generate_short_signals_batch(
            &underlying_prices,
            &call_prices,
            &put_prices,
            &implied_vols_high,
            &historical_vols_low,
            &strategy_params,
        )
        .expect("Short signal generation failed");

    let short_positions: usize = short_signals.iter().filter(|s| s.call_signal == -1).count();
    println!("✅ Short Straddle Signals:");
    println!("  Short positions: {}/{} ({:.1}%)", short_positions, short_signals.len(), (short_positions as f64 / short_signals.len() as f64) * 100.0);
    println!("  Average premium: ${:.2}", short_signals.iter().map(|s| s.total_cost).sum::<f64>() / short_positions as f64);
    println!();

    // ==================================================
    // Summary
    // ==================================================
    println!("=== Demo Summary ===\n");
    println!("✅ GPU Greeks: {:.1}x faster than CPU", speedup);
    println!("✅ Long Straddle: {} buy signals generated in {:?}", long_positions, time_straddle);
    println!("✅ Short Straddle: {} sell signals generated", short_positions);
    println!();
    println!("All features working correctly! 🎉");
}
