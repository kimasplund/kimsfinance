//! Unit Tests: GPU Orderflow Signals (Agent 2)
//!
//! Validates GPU orderflow feature calculation and signal generation accuracy.
//!
//! # Test Coverage
//!
//! - Orderflow imbalance accuracy: < 1e-9 error
//! - Volume delta accuracy: < 1e-6 error
//! - Signal generation: Exact match with CPU
//! - Batch processing: Multiple parameter sets simultaneously
//!
//! # Status
//!
//! - [PLACEHOLDER] GPU kernels not yet implemented (Agent 2 in progress)
//! - Tests will be enabled when `gpu_orderflow_signals_batch()` is available
//!
//! # Usage
//!
//! ```bash
//! cargo test --features gpu gpu_tick_orderflow -- --ignored
//! ```

#[cfg(feature = "gpu")]
mod gpu_tick_orderflow_tests {
    use approx::assert_abs_diff_eq;
    use kimsfinance_core::backtest::core::Signal;
    use kimsfinance_core::binance::Trade;
    use kimsfinance_core::gpu::device::GpuDevice;
    use std::sync::Arc;

    // ========================================================================
    // Test Configuration
    // ========================================================================

    const TOLERANCE: f64 = 1e-9;
    const VOLUME_TOLERANCE: f64 = 1e-6;

    // ========================================================================
    // Test Data Generators
    // ========================================================================

    fn generate_test_trades(n: usize) -> Vec<Trade> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let base_price = 45000.0;
        let base_timestamp = 1704067200000i64;

        (0..n)
            .map(|i| {
                let price_change = (rng.r#gen::<f64>() - 0.5) * 0.002;
                let price = base_price * (1.0 + price_change);
                let quantity = rng.gen_range(0.001..1.0);

                Trade {
                    trade_id: i as u64,
                    price,
                    quantity,
                    quote_quantity: price * quantity,
                    timestamp_ms: base_timestamp + (i as i64),
                    is_buyer_maker: rng.gen_bool(0.5),
                }
            })
            .collect()
    }

    // ========================================================================
    // CPU Reference Implementations
    // ========================================================================

    /// CPU reference: Calculate orderflow imbalance
    #[allow(dead_code)]
    fn cpu_orderflow_imbalance(trades: &[Trade], window_size: usize) -> Vec<f64> {
        let mut imbalances = Vec::with_capacity(trades.len());

        for i in 0..trades.len() {
            let start = i.saturating_sub(window_size);
            let window = &trades[start..=i];

            let mut buy_volume = 0.0;
            let mut sell_volume = 0.0;

            for trade in window {
                if !trade.is_buyer_maker {
                    buy_volume += trade.quantity;
                } else {
                    sell_volume += trade.quantity;
                }
            }

            let total = buy_volume + sell_volume;
            let imbalance = if total > 0.0 { buy_volume / total } else { 0.5 };

            imbalances.push(imbalance);
        }

        imbalances
    }

    /// CPU reference: Calculate volume delta
    #[allow(dead_code)]
    fn cpu_volume_delta(trades: &[Trade], window_size: usize) -> Vec<f64> {
        let mut deltas = Vec::with_capacity(trades.len());

        for i in 0..trades.len() {
            let start = i.saturating_sub(window_size);
            let window = &trades[start..=i];

            let mut buy_volume = 0.0;
            let mut sell_volume = 0.0;

            for trade in window {
                if !trade.is_buyer_maker {
                    buy_volume += trade.quantity;
                } else {
                    sell_volume += trade.quantity;
                }
            }

            deltas.push(buy_volume - sell_volume);
        }

        deltas
    }

    /// CPU reference: Generate signals from orderflow features
    #[allow(dead_code)]
    fn cpu_orderflow_signals(
        trades: &[Trade],
        params: &[f64], // [imbalance_threshold, volume_delta_threshold, ...]
    ) -> Vec<Signal> {
        let window_size = params[0] as usize;
        let imbalance_threshold = params[1];
        let volume_delta_threshold = params[2];

        let imbalances = cpu_orderflow_imbalance(trades, window_size);
        let deltas = cpu_volume_delta(trades, window_size);

        imbalances
            .iter()
            .zip(deltas.iter())
            .map(|(&imb, &delta)| {
                if imb > imbalance_threshold && delta > volume_delta_threshold {
                    Signal::Buy
                } else if imb < (1.0 - imbalance_threshold) && delta < -volume_delta_threshold {
                    Signal::Sell
                } else {
                    Signal::Hold
                }
            })
            .collect()
    }

    // ========================================================================
    // GPU Implementation Placeholders
    // ========================================================================

    #[allow(dead_code)]
    fn gpu_orderflow_signals_batch(
        device: &Arc<GpuDevice>,
        trades: &[Trade],
        params_batch: &[Vec<f64>],
    ) -> Result<Vec<Vec<Signal>>, String> {
        // PLACEHOLDER: Will be implemented by Agent 2
        let _ = (device, trades, params_batch);
        Err("GPU orderflow signals not yet implemented (Agent 2)".to_string())
    }

    // ========================================================================
    // Unit Tests: Feature Accuracy
    // ========================================================================

    #[test]
    #[ignore]
    fn test_gpu_orderflow_imbalance_accuracy() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let trades = generate_test_trades(10_000);
        let window_size = 100;

        // CPU reference
        let cpu_imbalances = cpu_orderflow_imbalance(&trades, window_size);

        // GPU implementation (placeholder)
        // let gpu_imbalances = gpu_orderflow_imbalance(&device, &trades, window_size).unwrap();

        // Validation (when GPU ready)
        // for (i, (&gpu, &cpu)) in gpu_imbalances.iter().zip(cpu_imbalances.iter()).enumerate() {
        //     assert_abs_diff_eq!(gpu, cpu, epsilon = TOLERANCE,
        //         "Imbalance mismatch at index {}", i);
        // }

        println!(
            "✅ CPU orderflow imbalance calculated: {} values",
            cpu_imbalances.len()
        );
    }

    #[test]
    #[ignore]
    fn test_gpu_volume_delta_accuracy() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let trades = generate_test_trades(10_000);
        let window_size = 100;

        let cpu_deltas = cpu_volume_delta(&trades, window_size);

        // GPU validation (when ready)
        // let gpu_deltas = gpu_volume_delta(&device, &trades, window_size).unwrap();
        // for (i, (&gpu, &cpu)) in gpu_deltas.iter().zip(cpu_deltas.iter()).enumerate() {
        //     assert_abs_diff_eq!(gpu, cpu, epsilon = VOLUME_TOLERANCE,
        //         "Volume delta mismatch at index {}", i);
        // }

        println!(
            "✅ CPU volume delta calculated: {} values",
            cpu_deltas.len()
        );
    }

    // ========================================================================
    // Unit Tests: Signal Generation
    // ========================================================================

    #[test]
    #[ignore]
    fn test_gpu_orderflow_signals_single_strategy() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let trades = generate_test_trades(10_000);
        let params = vec![100.0, 0.6, 10.0]; // window_size, imb_threshold, vol_delta_threshold

        // CPU reference
        let cpu_signals = cpu_orderflow_signals(&trades, &params);

        // GPU implementation
        let gpu_signals = gpu_orderflow_signals_batch(&device, &trades, &[params])
            .expect("GPU orderflow signals failed");

        // Validate exact signal match
        assert_eq!(gpu_signals.len(), 1);
        assert_eq!(gpu_signals[0].len(), cpu_signals.len());

        let matches = gpu_signals[0]
            .iter()
            .zip(cpu_signals.iter())
            .filter(|(gpu, cpu)| gpu == cpu)
            .count();

        let match_rate = matches as f64 / cpu_signals.len() as f64;
        println!("Signal match rate: {:.2}%", match_rate * 100.0);

        assert_eq!(matches, cpu_signals.len(), "Signals must match exactly");
    }

    #[test]
    #[ignore]
    fn test_gpu_orderflow_signals_batch() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let trades = generate_test_trades(10_000);

        // Multiple parameter sets
        let params_batch = vec![
            vec![50.0, 0.55, 5.0],
            vec![100.0, 0.60, 10.0],
            vec![150.0, 0.65, 15.0],
        ];

        // CPU reference (sequential)
        let cpu_signals_batch: Vec<Vec<Signal>> = params_batch
            .iter()
            .map(|params| cpu_orderflow_signals(&trades, params))
            .collect();

        // GPU batch (parallel)
        let gpu_signals_batch = gpu_orderflow_signals_batch(&device, &trades, &params_batch)
            .expect("GPU batch signals failed");

        // Validate all strategies
        assert_eq!(gpu_signals_batch.len(), cpu_signals_batch.len());

        for (i, (gpu, cpu)) in gpu_signals_batch
            .iter()
            .zip(cpu_signals_batch.iter())
            .enumerate()
        {
            assert_eq!(gpu.len(), cpu.len(), "Strategy {} length mismatch", i);

            let matches = gpu.iter().zip(cpu.iter()).filter(|(g, c)| g == c).count();
            assert_eq!(matches, cpu.len(), "Strategy {} signals mismatch", i);

            println!("✅ Strategy {} signals match: {}/{}", i, matches, cpu.len());
        }
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    #[ignore]
    fn test_gpu_orderflow_zero_window() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let trades = generate_test_trades(1000);
        let params = vec![0.0, 0.6, 10.0]; // Zero window size

        let cpu_signals = cpu_orderflow_signals(&trades, &params);
        let gpu_signals = gpu_orderflow_signals_batch(&device, &trades, &[params])
            .expect("Should handle zero window");

        assert_eq!(gpu_signals[0].len(), cpu_signals.len());
    }

    #[test]
    #[ignore]
    fn test_gpu_orderflow_large_window() {
        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let trades = generate_test_trades(1000);
        let params = vec![10000.0, 0.6, 10.0]; // Window larger than dataset

        let cpu_signals = cpu_orderflow_signals(&trades, &params);
        let gpu_signals = gpu_orderflow_signals_batch(&device, &trades, &[params])
            .expect("Should handle large window");

        assert_eq!(gpu_signals[0].len(), cpu_signals.len());
    }

    #[test]
    #[ignore]
    fn test_gpu_orderflow_all_buy() {
        // Edge case: All trades are buys
        let trades: Vec<Trade> = (0..1000)
            .map(|i| Trade {
                trade_id: i,
                price: 45000.0,
                quantity: 1.0,
                quote_quantity: 45000.0,
                timestamp_ms: 1704067200000 + i as i64,
                is_buyer_maker: false, // All buys
            })
            .collect();

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let params = vec![100.0, 0.6, 10.0];

        let cpu_signals = cpu_orderflow_signals(&trades, &params);
        let gpu_signals = gpu_orderflow_signals_batch(&device, &trades, &[params])
            .expect("Should handle all-buy scenario");

        // All should be Buy signals (imbalance = 1.0 > 0.6)
        let cpu_buy_count = cpu_signals
            .iter()
            .filter(|s| matches!(s, Signal::Buy))
            .count();
        let gpu_buy_count = gpu_signals[0]
            .iter()
            .filter(|s| matches!(s, Signal::Buy))
            .count();

        assert_eq!(cpu_buy_count, gpu_buy_count);
        println!(
            "All-buy scenario: {}/{} Buy signals",
            cpu_buy_count,
            trades.len()
        );
    }

    // ========================================================================
    // Performance Tests
    // ========================================================================

    #[test]
    #[ignore]
    fn test_gpu_orderflow_throughput() {
        use std::time::Instant;

        let device = Arc::new(GpuDevice::new().expect("GPU required"));
        let trades = generate_test_trades(100_000);
        let params_batch = vec![
            vec![50.0, 0.55, 5.0],
            vec![100.0, 0.60, 10.0],
            vec![150.0, 0.65, 15.0],
        ];

        // Warmup
        for _ in 0..3 {
            let _ = gpu_orderflow_signals_batch(&device, &trades, &params_batch);
        }

        // Measure
        let start = Instant::now();
        let _signals = gpu_orderflow_signals_batch(&device, &trades, &params_batch)
            .expect("GPU orderflow failed");
        let elapsed = start.elapsed();

        let total_features = trades.len() * params_batch.len();
        let throughput = total_features as f64 / elapsed.as_secs_f64();

        println!(
            "GPU orderflow throughput: {:.2} M features/sec",
            throughput / 1e6
        );

        // Target: 200-500M features/sec
        assert!(
            throughput > 50e6,
            "Throughput too low: {:.2} M/sec (target: >50 M/sec)",
            throughput / 1e6
        );
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_gpu_tick_orderflow_requires_gpu_feature() {
    println!("GPU tick orderflow tests require --features gpu");
}
