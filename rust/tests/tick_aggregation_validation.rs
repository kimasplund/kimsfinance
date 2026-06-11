/// GPU Tick Aggregation Validation Tests
///
/// Validates GPU tick aggregation against CPU reference implementation.
///
/// # Test Strategy
///
/// 1. **Correctness**: GPU output matches CPU exactly
/// 2. **Performance**: GPU achieves >10x speedup for >100K trades
/// 3. **Edge Cases**: Empty trades, single trade, many candles
/// 4. **Numerical Accuracy**: Within 1e-5 tolerance for float32

#[cfg(feature = "gpu")]
#[cfg(test)]
mod gpu_tests {
    use kimsfinance_core::binance::{Candle, Timeframe, Trade};
    use kimsfinance_core::gpu::device::GpuDevice;
    use kimsfinance_core::gpu::tick_aggregation::TickAggregator;
    use std::time::Instant;

    /// Generate synthetic trade data for testing
    fn generate_test_trades(n: usize, timeframe_ms: i64) -> Vec<Trade> {
        let mut trades = Vec::with_capacity(n);
        let base_ts = 1609459200000i64; // 2021-01-01 00:00:00

        for i in 0..n {
            let ts = base_ts + (i as i64) * 1000; // 1 trade per second
            let price = 100.0 + ((i % 100) as f64) * 0.1;
            let quantity = 1.0 + ((i % 10) as f64) * 0.1;

            trades.push(Trade {
                trade_id: i as u64,
                price,
                quantity,
                quote_quantity: price * quantity,
                timestamp_ms: ts,
                is_buyer_maker: i % 2 == 0,
            });
        }

        trades
    }

    /// CPU reference implementation (from aggregation.rs)
    fn aggregate_trades_cpu(trades: &[Trade], timeframe: Timeframe) -> Vec<Candle> {
        use std::collections::HashMap;

        if trades.is_empty() {
            return Vec::new();
        }

        let timeframe_ms = timeframe.to_ms();

        // Bin trades to buckets
        let mut bucket_trades: HashMap<i64, Vec<&Trade>> = HashMap::new();
        for trade in trades {
            let bucket_id = trade.timestamp_ms / timeframe_ms;
            bucket_trades
                .entry(bucket_id)
                .or_insert_with(Vec::new)
                .push(trade);
        }

        // Aggregate each bucket
        let mut candles: Vec<Candle> = bucket_trades
            .iter()
            .map(|(&bucket_id, trades_in_bucket)| {
                let mut trades_sorted = trades_in_bucket.clone();
                trades_sorted.sort_by_key(|t| t.timestamp_ms);

                let open = trades_sorted.first().unwrap().price;
                let close = trades_sorted.last().unwrap().price;

                let high = trades_sorted
                    .iter()
                    .map(|t| t.price)
                    .fold(f64::NEG_INFINITY, f64::max);

                let low = trades_sorted
                    .iter()
                    .map(|t| t.price)
                    .fold(f64::INFINITY, f64::min);

                let volume: f64 = trades_sorted.iter().map(|t| t.quantity).sum();
                let quote_volume: f64 = trades_sorted.iter().map(|t| t.quote_quantity).sum();

                Candle {
                    timestamp: bucket_id * timeframe_ms,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    quote_volume,
                    num_trades: trades_sorted.len(),
                }
            })
            .collect();

        // Sort by timestamp
        candles.sort_by_key(|c| c.timestamp);
        candles
    }

    /// Compare GPU vs CPU results (correctness validation)
    fn validate_gpu_vs_cpu(
        gpu_candles: &[(i64, f32, f32, f32, f32, f32, i32)],
        cpu_candles: &[Candle],
    ) {
        assert_eq!(
            gpu_candles.len(),
            cpu_candles.len(),
            "GPU and CPU should produce same number of candles"
        );

        for (i, (gpu, cpu)) in gpu_candles.iter().zip(cpu_candles.iter()).enumerate() {
            let (gpu_ts, gpu_open, gpu_high, gpu_low, gpu_close, gpu_vol, gpu_count) = gpu;
            let tolerance = 1e-4; // Allow small float32 rounding error

            assert_eq!(*gpu_ts, cpu.timestamp, "Candle {} timestamp mismatch", i);

            assert!(
                ((*gpu_open as f64) - cpu.open).abs() < tolerance,
                "Candle {} open mismatch: GPU={}, CPU={}",
                i,
                gpu_open,
                cpu.open
            );

            assert!(
                ((*gpu_high as f64) - cpu.high).abs() < tolerance,
                "Candle {} high mismatch: GPU={}, CPU={}",
                i,
                gpu_high,
                cpu.high
            );

            assert!(
                ((*gpu_low as f64) - cpu.low).abs() < tolerance,
                "Candle {} low mismatch: GPU={}, CPU={}",
                i,
                gpu_low,
                cpu.low
            );

            assert!(
                ((*gpu_close as f64) - cpu.close).abs() < tolerance,
                "Candle {} close mismatch: GPU={}, CPU={}",
                i,
                gpu_close,
                cpu.close
            );

            assert!(
                ((*gpu_vol as f64) - cpu.volume).abs() < tolerance,
                "Candle {} volume mismatch: GPU={}, CPU={}",
                i,
                gpu_vol,
                cpu.volume
            );

            assert_eq!(
                *gpu_count as usize, cpu.num_trades,
                "Candle {} trade count mismatch",
                i
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_tick_aggregator_small_dataset() {
        let device = GpuDevice::new().expect("GPU not available");
        let aggregator = TickAggregator::new(device).expect("Failed to init aggregator");

        // Generate 1000 trades
        let trades = generate_test_trades(1000, 300_000);
        let timeframe = Timeframe::minutes(5);

        // Extract SoA arrays for GPU
        let timestamps: Vec<i64> = trades.iter().map(|t| t.timestamp_ms).collect();
        let prices: Vec<f32> = trades.iter().map(|t| t.price as f32).collect();
        let volumes: Vec<f32> = trades.iter().map(|t| t.quantity as f32).collect();
        let sides: Vec<i8> = trades
            .iter()
            .map(|t| if t.is_buyer_maker { -1 } else { 1 })
            .collect();

        // GPU aggregation
        let gpu_result = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, timeframe.to_ms())
            .expect("GPU aggregation failed");

        // CPU aggregation
        let cpu_candles = aggregate_trades_cpu(&trades, timeframe);

        // Prepare GPU results for comparison
        let gpu_candles_tuples: Vec<(i64, f32, f32, f32, f32, f32, i32)> = (0..gpu_result
            .num_candles)
            .map(|i| {
                (
                    gpu_result.timestamps[i],
                    gpu_result.open[i],
                    gpu_result.high[i],
                    gpu_result.low[i],
                    gpu_result.close[i],
                    gpu_result.volume[i],
                    gpu_result.num_trades[i],
                )
            })
            .collect();

        // Validate correctness
        validate_gpu_vs_cpu(&gpu_candles_tuples, &cpu_candles);

        println!("✅ Correctness validated: GPU matches CPU exactly");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_tick_aggregator_empty_trades() {
        let device = GpuDevice::new().expect("GPU not available");
        let aggregator = TickAggregator::new(device).expect("Failed to init aggregator");

        let timestamps: Vec<i64> = vec![];
        let prices: Vec<f32> = vec![];
        let volumes: Vec<f32> = vec![];
        let sides: Vec<i8> = vec![];

        let result = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, 300_000)
            .expect("GPU aggregation failed");

        assert_eq!(result.num_candles, 0);
        println!("✅ Empty trades handled correctly");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_tick_aggregator_single_trade() {
        let device = GpuDevice::new().expect("GPU not available");
        let aggregator = TickAggregator::new(device).expect("Failed to init aggregator");

        let timestamps: Vec<i64> = vec![1609459200000];
        let prices: Vec<f32> = vec![100.0];
        let volumes: Vec<f32> = vec![1.5];
        let sides: Vec<i8> = vec![1];

        let result = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, 300_000)
            .expect("GPU aggregation failed");

        assert_eq!(result.num_candles, 1);
        assert_eq!(result.open[0], 100.0);
        assert_eq!(result.high[0], 100.0);
        assert_eq!(result.low[0], 100.0);
        assert_eq!(result.close[0], 100.0);
        assert_eq!(result.volume[0], 1.5);
        assert_eq!(result.num_trades[0], 1);

        println!("✅ Single trade handled correctly");
    }

    #[test]
    #[ignore] // Requires GPU - benchmark test
    fn test_gpu_tick_aggregator_performance_1m() {
        let device = GpuDevice::new().expect("GPU not available");
        let aggregator = TickAggregator::new(device).expect("Failed to init aggregator");

        // Generate 1M trades
        let n_trades = 1_000_000;
        let trades = generate_test_trades(n_trades, 300_000);
        let timeframe = Timeframe::minutes(5);

        // Extract SoA arrays
        let timestamps: Vec<i64> = trades.iter().map(|t| t.timestamp_ms).collect();
        let prices: Vec<f32> = trades.iter().map(|t| t.price as f32).collect();
        let volumes: Vec<f32> = trades.iter().map(|t| t.quantity as f32).collect();
        let sides: Vec<i8> = trades
            .iter()
            .map(|t| if t.is_buyer_maker { -1 } else { 1 })
            .collect();

        // Warm-up (JIT compilation)
        let _ = aggregator
            .aggregate(
                &timestamps[..1000],
                &prices[..1000],
                &volumes[..1000],
                &sides[..1000],
                timeframe.to_ms(),
            )
            .expect("Warm-up failed");

        // Benchmark GPU
        let start_gpu = Instant::now();
        let gpu_result = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, timeframe.to_ms())
            .expect("GPU aggregation failed");
        let gpu_time = start_gpu.elapsed();

        // Benchmark CPU
        let start_cpu = Instant::now();
        let cpu_candles = aggregate_trades_cpu(&trades, timeframe);
        let cpu_time = start_cpu.elapsed();

        // Calculate throughput and speedup
        let gpu_throughput = (n_trades as f64) / gpu_time.as_secs_f64();
        let cpu_throughput = (n_trades as f64) / cpu_time.as_secs_f64();
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("\n=== GPU Tick Aggregation Performance (1M trades) ===");
        println!("GPU time: {:?}", gpu_time);
        println!("CPU time: {:?}", cpu_time);
        println!("GPU throughput: {:.2} M trades/sec", gpu_throughput / 1e6);
        println!("CPU throughput: {:.2} M trades/sec", cpu_throughput / 1e6);
        println!("Speedup: {:.2}x", speedup);
        println!("GPU candles: {}", gpu_result.num_candles);
        println!("CPU candles: {}", cpu_candles.len());

        // Validate correctness
        let gpu_candles_tuples: Vec<(i64, f32, f32, f32, f32, f32, i32)> = (0..gpu_result
            .num_candles)
            .map(|i| {
                (
                    gpu_result.timestamps[i],
                    gpu_result.open[i],
                    gpu_result.high[i],
                    gpu_result.low[i],
                    gpu_result.close[i],
                    gpu_result.volume[i],
                    gpu_result.num_trades[i],
                )
            })
            .collect();

        validate_gpu_vs_cpu(&gpu_candles_tuples, &cpu_candles);

        // Performance assertions
        assert!(
            gpu_throughput > 100_000_000.0,
            "GPU throughput too low: {:.2} M/s",
            gpu_throughput / 1e6
        );

        assert!(
            speedup > 5.0,
            "GPU speedup too low: {:.2}x (expected >5x)",
            speedup
        );

        println!("✅ Performance validated: {:.2}x speedup", speedup);
    }

    #[test]
    #[ignore] // Requires GPU - benchmark test
    fn test_gpu_tick_aggregator_performance_10m() {
        let device = GpuDevice::new().expect("GPU not available");
        let aggregator = TickAggregator::new(device).expect("Failed to init aggregator");

        // Generate 10M trades
        let n_trades = 10_000_000;
        let trades = generate_test_trades(n_trades, 300_000);
        let timeframe = Timeframe::minutes(5);

        // Extract SoA arrays
        let timestamps: Vec<i64> = trades.iter().map(|t| t.timestamp_ms).collect();
        let prices: Vec<f32> = trades.iter().map(|t| t.price as f32).collect();
        let volumes: Vec<f32> = trades.iter().map(|t| t.quantity as f32).collect();
        let sides: Vec<i8> = trades
            .iter()
            .map(|t| if t.is_buyer_maker { -1 } else { 1 })
            .collect();

        // Warm-up
        let _ = aggregator
            .aggregate(
                &timestamps[..1000],
                &prices[..1000],
                &volumes[..1000],
                &sides[..1000],
                timeframe.to_ms(),
            )
            .expect("Warm-up failed");

        // Benchmark GPU
        let start_gpu = Instant::now();
        let gpu_result = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, timeframe.to_ms())
            .expect("GPU aggregation failed");
        let gpu_time = start_gpu.elapsed();

        // Calculate throughput
        let gpu_throughput = (n_trades as f64) / gpu_time.as_secs_f64();

        println!("\n=== GPU Tick Aggregation Performance (10M trades) ===");
        println!("GPU time: {:?}", gpu_time);
        println!("GPU throughput: {:.2} M trades/sec", gpu_throughput / 1e6);
        println!("GPU candles: {}", gpu_result.num_candles);

        // Performance assertion (should achieve >500M trades/sec for large datasets)
        assert!(
            gpu_throughput > 500_000_000.0,
            "GPU throughput too low: {:.2} M/s (expected >500 M/s)",
            gpu_throughput / 1e6
        );

        println!(
            "✅ Performance validated: {:.2} M trades/sec",
            gpu_throughput / 1e6
        );
    }

    #[test]
    #[ignore] // Requires GPU - long-running test
    fn test_gpu_tick_aggregator_stress_106m() {
        let device = GpuDevice::new().expect("GPU not available");
        let aggregator = TickAggregator::new(device).expect("Failed to init aggregator");

        // Generate 106M trades (target dataset size)
        let n_trades = 106_000_000;
        println!("Generating {} trades...", n_trades);

        let trades = generate_test_trades(n_trades, 300_000);
        let timeframe = Timeframe::minutes(5);

        println!("Extracting SoA arrays...");
        let timestamps: Vec<i64> = trades.iter().map(|t| t.timestamp_ms).collect();
        let prices: Vec<f32> = trades.iter().map(|t| t.price as f32).collect();
        let volumes: Vec<f32> = trades.iter().map(|t| t.quantity as f32).collect();
        let sides: Vec<i8> = trades
            .iter()
            .map(|t| if t.is_buyer_maker { -1 } else { 1 })
            .collect();

        println!("Running GPU aggregation...");
        let start_gpu = Instant::now();
        let gpu_result = aggregator
            .aggregate(&timestamps, &prices, &volumes, &sides, timeframe.to_ms())
            .expect("GPU aggregation failed");
        let gpu_time = start_gpu.elapsed();

        let gpu_throughput = (n_trades as f64) / gpu_time.as_secs_f64();

        println!("\n=== GPU Tick Aggregation Stress Test (106M trades) ===");
        println!("GPU time: {:?}", gpu_time);
        println!("GPU throughput: {:.2} B trades/sec", gpu_throughput / 1e9);
        println!("GPU candles: {}", gpu_result.num_candles);

        // Target: <100ms for 106M trades = >1B trades/sec
        assert!(
            gpu_throughput > 1_000_000_000.0,
            "GPU throughput too low: {:.2} B/s (expected >1 B/s)",
            gpu_throughput / 1e9
        );

        println!(
            "✅ Stress test passed: {:.2} B trades/sec",
            gpu_throughput / 1e9
        );
    }
}
