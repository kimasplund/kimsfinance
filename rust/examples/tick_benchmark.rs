//! Comprehensive tick engine performance benchmark

use kimsfinance_core::backtest::{BacktestConfig, IntraCandleMomentum, TickEngine, TickStrategy};
use kimsfinance_core::binance::{Timeframe, Trade};
use std::time::Instant;

fn main() {
    println!("=== Tick Engine Performance Benchmark ===\n");

    // Test different dataset sizes
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for &n in &sizes {
        println!("Dataset size: {} trades", n);

        // Generate trades
        let trades = generate_trades(n);

        // Create engine and strategy
        let config = BacktestConfig::default();
        let engine = TickEngine::new(config);
        let mut strategy = IntraCandleMomentum::new(0.5);
        let timeframe = Timeframe::parse("5m").unwrap();

        // Run benchmark
        let start = Instant::now();
        let result = engine.run(&mut strategy, &trades, timeframe).unwrap();
        let duration = start.elapsed();

        let throughput = n as f64 / duration.as_secs_f64();

        println!("  Duration: {:.3}s", duration.as_secs_f64());
        println!("  Throughput: {:.2} M trades/sec", throughput / 1_000_000.0);
        println!("  Trades executed: {}", result.num_trades);
        println!("  Final equity: ${:.2}", result.final_equity);
        
        // Verify >1M trades/sec target
        if throughput > 1_000_000.0 {
            println!("  ✅ PASS: Exceeds 1M trades/sec target");
        } else {
            println!("  ❌ FAIL: Below 1M trades/sec target");
        }
        println!();
    }
}

fn generate_trades(n: usize) -> Vec<Trade> {
    let mut trades = Vec::with_capacity(n);
    let mut price = 50_000.0;

    for i in 0..n {
        price += ((i % 100) as f64 - 50.0) * 0.1; // Simple oscillation
        
        trades.push(Trade {
            trade_id: i as u64,
            price,
            quantity: 1.0,
            quote_quantity: price,
            timestamp_ms: i as i64 * 100,
            is_buyer_maker: i % 2 == 0,
        });
    }

    trades
}
