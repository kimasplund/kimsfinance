//! Integration tests for GPU-accelerated Euler Search optimizer
//!
//! Tests convergence, speedup vs grid search, and GPU acceleration

#[cfg(feature = "gpu")]
mod euler_search_tests {
    use kimsfinance_core::backtest::{BacktestConfig, EulerSearchOptimizer, StrategyType};
    use kimsfinance_core::gpu::device::GpuDevice;
    use ndarray::Array1;
    use std::sync::Arc;

    /// Generate synthetic OHLCV data for testing
    fn generate_test_data(
        num_candles: usize,
    ) -> (
        Vec<i64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        let timestamps: Vec<i64> = (0..num_candles).map(|i| i as i64 * 60).collect();

        // Synthetic price data with trend + noise
        let base_price = 100.0;
        let close: Array1<f64> = Array1::from_iter((0..num_candles).map(|i| {
            let trend = (i as f64 * 0.1).sin() * 10.0;
            let noise = ((i * 7) % 13) as f64 - 6.5;
            base_price + trend + noise
        }));

        let open = close.mapv(|c| c - 0.5 + ((c * 123.456) as i64 % 10) as f64 / 10.0);
        let high = close.mapv(|c| c + ((c * 234.567) as i64 % 20) as f64 / 10.0);
        let low = close.mapv(|c| c - ((c * 345.678) as i64 % 20) as f64 / 10.0);
        let volume = Array1::from_elem(num_candles, 1_000_000.0);

        (timestamps, open, high, low, close, volume)
    }

    #[test]
    #[ignore] // Run with: cargo test --features gpu euler_search_integration -- --ignored
    fn test_euler_search_convergence() {
        let device = Arc::new(GpuDevice::new().expect("GPU required for this test"));

        // Generate test data
        let (timestamps, open, high, low, close, volume) = generate_test_data(1000);

        // Create optimizer
        let mut optimizer = EulerSearchOptimizer::new(device)
            .segment_amount(4)
            .max_iterations(10)
            .batch_size(500);

        // Add RSI parameters
        optimizer.add_parameter("rsi_period", 5.0, 30.0, 5.0, 1.0);
        optimizer.add_parameter("buy_threshold", 20.0, 40.0, 5.0, 1.0);
        optimizer.add_parameter("sell_threshold", 60.0, 80.0, 5.0, 1.0);

        // Run optimization
        let result = optimizer
            .optimize(
                StrategyType::RsiCrossover,
                &timestamps,
                &open,
                &high,
                &low,
                &close,
                &volume,
                BacktestConfig::default(),
            )
            .expect("Optimization should succeed");

        // Validate convergence
        println!("\n=== Euler Search Results ===");
        println!("Iterations: {}", result.iterations);
        println!("Total evaluations: {}", result.total_evaluations);
        println!("Best fitness: {:.4}", result.best_fitness);
        println!("Best parameters: {:?}", result.best_parameters);
        println!("Total GPU time: {:.2}ms", result.total_gpu_time_ms);
        println!("Total time: {:.2}ms", result.total_time_ms);
        println!("\nConvergence history:");
        for (i, fitness) in result.convergence_history.iter().enumerate() {
            println!("  Iteration {}: {:.4}", i, fitness);
        }

        // Assertions
        assert!(result.iterations > 0, "Should run at least 1 iteration");
        assert!(result.iterations <= 10, "Should not exceed max iterations");
        assert!(
            result.total_evaluations < 1000,
            "Should use fewer evaluations than grid search"
        );
        assert!(result.best_fitness > -10.0, "Fitness should be reasonable");

        // Check parameter bounds
        let rsi_period = result.best_parameters.get("rsi_period").unwrap();
        assert!(
            *rsi_period >= 5.0 && *rsi_period <= 30.0,
            "RSI period out of bounds"
        );

        let buy_threshold = result.best_parameters.get("buy_threshold").unwrap();
        assert!(
            *buy_threshold >= 20.0 && *buy_threshold <= 40.0,
            "Buy threshold out of bounds"
        );

        let sell_threshold = result.best_parameters.get("sell_threshold").unwrap();
        assert!(
            *sell_threshold >= 60.0 && *sell_threshold <= 80.0,
            "Sell threshold out of bounds"
        );
    }

    #[test]
    #[ignore]
    fn test_euler_search_speedup() {
        let device = Arc::new(GpuDevice::new().expect("GPU required for this test"));

        // Generate test data
        let (timestamps, open, high, low, close, volume) = generate_test_data(500);

        // Create optimizer with smaller search space for faster test
        let mut optimizer = EulerSearchOptimizer::new(device)
            .segment_amount(4)
            .max_iterations(8)
            .batch_size(300);

        // 2-parameter optimization (faster)
        optimizer.add_parameter("fast_period", 5.0, 20.0, 5.0, 1.0);
        optimizer.add_parameter("slow_period", 20.0, 50.0, 10.0, 2.0);

        // Run optimization
        let result = optimizer
            .optimize(
                StrategyType::MaCrossover,
                &timestamps,
                &open,
                &high,
                &low,
                &close,
                &volume,
                BacktestConfig::default(),
            )
            .expect("Optimization should succeed");

        // Calculate speedup vs exhaustive grid
        let grid_speedup = result.grid_search_speedup(10);

        println!("\n=== Speedup Analysis ===");
        println!("Euler evaluations: {}", result.total_evaluations);
        println!(
            "Grid search evaluations (10 points/param): {}",
            10_usize.pow(2)
        );
        println!("Speedup: {:.2}x", grid_speedup);
        println!(
            "GPU time per evaluation: {:.3}ms",
            result.total_gpu_time_ms / result.total_evaluations as f64
        );

        // Should achieve significant speedup
        assert!(
            grid_speedup >= 1.5,
            "Euler should be faster than grid search"
        );
        assert!(
            result.total_evaluations < 100,
            "Should use <100 evaluations for 2 params"
        );
    }

    #[test]
    #[ignore]
    fn test_euler_search_early_stopping() {
        let device = Arc::new(GpuDevice::new().expect("GPU required for this test"));

        // Generate test data
        let (timestamps, open, high, low, close, volume) = generate_test_data(500);

        // Create optimizer with early stopping
        let mut optimizer = EulerSearchOptimizer::new(device)
            .segment_amount(4)
            .max_iterations(20) // High max
            .early_stopping_patience(Some(3)) // Should stop after 3 iterations without improvement
            .batch_size(200);

        // Single parameter for faster convergence
        optimizer.add_parameter("rsi_period", 10.0, 20.0, 2.0, 1.0);
        optimizer.add_parameter("buy_threshold", 25.0, 35.0, 2.0, 1.0);

        // Run optimization
        let result = optimizer
            .optimize(
                StrategyType::RsiCrossover,
                &timestamps,
                &open,
                &high,
                &low,
                &close,
                &volume,
                BacktestConfig::default(),
            )
            .expect("Optimization should succeed");

        println!("\n=== Early Stopping Test ===");
        println!("Iterations: {} (max: 20)", result.iterations);
        println!("Total evaluations: {}", result.total_evaluations);

        // Should stop before max iterations due to early stopping
        assert!(
            result.iterations < 20,
            "Early stopping should trigger before max iterations"
        );
    }

    #[test]
    #[ignore]
    fn test_euler_search_gpu_performance() {
        let device = Arc::new(GpuDevice::new().expect("GPU required for this test"));

        // Large dataset to test GPU acceleration
        let (timestamps, open, high, low, close, volume) = generate_test_data(10_000);

        // Create optimizer
        let mut optimizer = EulerSearchOptimizer::new(device)
            .segment_amount(4)
            .max_iterations(5)
            .batch_size(1000); // Large batch for GPU

        // 3 parameters
        optimizer.add_parameter("rsi_period", 10.0, 20.0, 5.0, 1.0);
        optimizer.add_parameter("buy_threshold", 25.0, 35.0, 5.0, 1.0);
        optimizer.add_parameter("sell_threshold", 65.0, 75.0, 5.0, 1.0);

        // Run optimization and measure time
        let start = std::time::Instant::now();
        let result = optimizer
            .optimize(
                StrategyType::RsiCrossover,
                &timestamps,
                &open,
                &high,
                &low,
                &close,
                &volume,
                BacktestConfig::default(),
            )
            .expect("Optimization should succeed");
        let total_time = start.elapsed();

        println!("\n=== GPU Performance Test ===");
        println!("Dataset: {} candles", timestamps.len());
        println!("Total evaluations: {}", result.total_evaluations);
        println!("GPU time: {:.2}ms", result.total_gpu_time_ms);
        println!("Total time: {:.2}ms", result.total_time_ms);
        println!(
            "Wall clock time: {:.2}ms",
            total_time.as_secs_f64() * 1000.0
        );
        println!(
            "GPU utilization: {:.1}%",
            (result.total_gpu_time_ms / result.total_time_ms) * 100.0
        );

        // Performance targets
        let avg_iter_time = result.total_time_ms / result.iterations as f64;
        println!("Average iteration time: {:.2}ms", avg_iter_time);

        // Each iteration with 1000 params should be <250ms
        assert!(
            avg_iter_time < 500.0,
            "Average iteration time {:.2}ms exceeds 500ms target",
            avg_iter_time
        );
    }

    #[test]
    fn test_euler_search_refinement_step() {
        // Test refinement step structure
        let mut step_sizes = std::collections::HashMap::new();
        step_sizes.insert("param1".to_string(), 5.0);
        step_sizes.insert("param2".to_string(), 10.0);

        let mut search_ranges = std::collections::HashMap::new();
        search_ranges.insert("param1".to_string(), (0.0, 100.0));
        search_ranges.insert("param2".to_string(), (0.0, 200.0));

        let step = kimsfinance_core::backtest::RefinementStep {
            iteration: 0,
            step_sizes,
            search_ranges,
            num_evaluations: 100,
            best_fitness: 1.5,
        };

        assert_eq!(step.iteration, 0);
        assert_eq!(step.num_evaluations, 100);
        assert_eq!(step.best_fitness, 1.5);
    }
}
