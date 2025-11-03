//! Integration test for CPU orderflow processing
//!
//! This test validates that the CPU orderflow implementation provides
//! automatic fallback when GPU is unavailable.

use kimsfinance_core::cpu::orderflow::{
    OrderflowBatchProcessor, OrderflowInput, StrategyConfig, NUM_FEATURES,
};

#[test]
fn test_cpu_orderflow_basic_functionality() {
    let processor = OrderflowBatchProcessor::new();

    // Create sample data
    let num_ticks = 100;
    let input = OrderflowInput {
        timestamps: (0..num_ticks).map(|i| (i as i64) * 1000).collect(),
        close_prices: (0..num_ticks).map(|i| 100.0 + (i as f32) * 0.1).collect(),
        volumes: vec![1000.0; num_ticks],
        buy_volumes: vec![600.0; num_ticks],
        sell_volumes: vec![400.0; num_ticks],
    };

    // Configure strategies
    let strategies = vec![
        StrategyConfig::momentum(),
        StrategyConfig::mean_reversion(),
    ];

    // Process batch
    let result = processor.process_batch(&input, &strategies);
    assert!(result.is_ok(), "Processing should succeed");

    let output = result.unwrap();

    // Validate output structure
    assert_eq!(output.signals.len(), 2, "Should have 2 strategies");
    assert_eq!(output.features.len(), 2, "Should have 2 feature sets");
    assert_eq!(output.feature_ranges.len(), 2, "Should have 2 range sets");

    // Validate signal dimensions
    for strategy_signals in &output.signals {
        assert_eq!(
            strategy_signals.len(),
            num_ticks,
            "Each strategy should have signals for all ticks"
        );
    }

    // Validate feature dimensions
    for strategy_features in &output.features {
        assert_eq!(
            strategy_features.len(),
            num_ticks * NUM_FEATURES,
            "Each strategy should have NUM_FEATURES per tick"
        );
    }

    // Validate feature ranges
    for ranges in &output.feature_ranges {
        assert_eq!(
            ranges.len(),
            NUM_FEATURES * 2,
            "Should have min/max pair for each feature"
        );

        // Check that min <= max for each feature
        for i in 0..NUM_FEATURES {
            let min = ranges[i * 2];
            let max = ranges[i * 2 + 1];
            assert!(
                min <= max,
                "Feature {} min ({}) should be <= max ({})",
                i,
                min,
                max
            );
        }
    }
}

#[test]
fn test_cpu_orderflow_calibration() {
    let processor = OrderflowBatchProcessor::new();

    // Create sample data with known characteristics
    let input = OrderflowInput {
        timestamps: vec![1000, 2000, 3000, 4000, 5000],
        close_prices: vec![100.0, 105.0, 110.0, 115.0, 120.0],
        volumes: vec![1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
        buy_volumes: vec![600.0, 900.0, 1200.0, 1500.0, 1800.0],
        sell_volumes: vec![400.0, 600.0, 800.0, 1000.0, 1200.0],
    };

    // Calibrate ranges
    let result = processor.calibrate_ranges(&input);
    assert!(result.is_ok(), "Calibration should succeed");

    let ranges = result.unwrap();
    assert_eq!(
        ranges.len(),
        NUM_FEATURES * 2,
        "Should have min/max for all features"
    );

    // Validate ranges are sensible
    for i in 0..NUM_FEATURES {
        let min = ranges[i * 2];
        let max = ranges[i * 2 + 1];
        assert!(
            min.is_finite() && max.is_finite(),
            "Feature {} ranges should be finite",
            i
        );
        assert!(
            min <= max,
            "Feature {} min ({}) should be <= max ({})",
            i,
            min,
            max
        );
    }
}

#[test]
fn test_cpu_orderflow_all_strategy_types() {
    let processor = OrderflowBatchProcessor::new();

    // Create sample data
    let input = OrderflowInput {
        timestamps: vec![1000, 2000, 3000, 4000, 5000],
        close_prices: vec![100.0, 101.0, 102.0, 103.0, 104.0],
        volumes: vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        buy_volumes: vec![600.0, 700.0, 800.0, 900.0, 1000.0],
        sell_volumes: vec![400.0, 400.0, 400.0, 400.0, 400.0],
    };

    // Test all strategy types
    let strategies = vec![
        StrategyConfig::momentum(),
        StrategyConfig::mean_reversion(),
        StrategyConfig::breakout(),
        StrategyConfig::scalping(),
        StrategyConfig::trend_following(),
    ];

    let result = processor.process_batch(&input, &strategies);
    assert!(result.is_ok(), "Processing all strategies should succeed");

    let output = result.unwrap();
    assert_eq!(
        output.signals.len(),
        5,
        "Should have signals for all 5 strategies"
    );
}

#[test]
fn test_cpu_orderflow_input_validation() {
    let processor = OrderflowBatchProcessor::new();

    // Test empty input
    let empty_input = OrderflowInput {
        timestamps: vec![],
        close_prices: vec![],
        volumes: vec![],
        buy_volumes: vec![],
        sell_volumes: vec![],
    };

    let strategies = vec![StrategyConfig::momentum()];
    let result = processor.process_batch(&empty_input, &strategies);
    assert!(result.is_err(), "Empty input should fail");

    // Test mismatched lengths
    let mismatched_input = OrderflowInput {
        timestamps: vec![1000, 2000],
        close_prices: vec![100.0], // Wrong length!
        volumes: vec![1000.0, 1100.0],
        buy_volumes: vec![600.0, 700.0],
        sell_volumes: vec![400.0, 400.0],
    };

    let result = processor.process_batch(&mismatched_input, &strategies);
    assert!(result.is_err(), "Mismatched input should fail");

    // Test no strategies
    let valid_input = OrderflowInput {
        timestamps: vec![1000, 2000],
        close_prices: vec![100.0, 101.0],
        volumes: vec![1000.0, 1100.0],
        buy_volumes: vec![600.0, 700.0],
        sell_volumes: vec![400.0, 400.0],
    };

    let result = processor.process_batch(&valid_input, &[]);
    assert!(result.is_err(), "No strategies should fail");
}

#[test]
fn test_cpu_orderflow_large_dataset() {
    let processor = OrderflowBatchProcessor::new();

    // Test with 50K ticks (realistic production size)
    let num_ticks = 50_000;
    let input = OrderflowInput {
        timestamps: (0..num_ticks).map(|i| (i as i64) * 100).collect(),
        close_prices: (0..num_ticks)
            .map(|i| 100.0 + ((i as f32) * 0.01).sin() * 10.0)
            .collect(),
        volumes: (0..num_ticks)
            .map(|i| 1000.0 + ((i as f32) * 0.02).cos() * 500.0)
            .collect(),
        buy_volumes: (0..num_ticks)
            .map(|i| 500.0 + ((i as f32) * 0.01).sin() * 300.0)
            .collect(),
        sell_volumes: (0..num_ticks)
            .map(|i| 500.0 + ((i as f32) * 0.01).cos() * 300.0)
            .collect(),
    };

    let strategies = vec![
        StrategyConfig::momentum(),
        StrategyConfig::mean_reversion(),
        StrategyConfig::breakout(),
    ];

    let start = std::time::Instant::now();
    let result = processor.process_batch(&input, &strategies);
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Large dataset processing should succeed");
    println!(
        "CPU orderflow processing (50K ticks, 3 strategies): {:?}",
        elapsed
    );

    let output = result.unwrap();
    assert_eq!(output.signals.len(), 3);
    assert_eq!(output.signals[0].len(), num_ticks);

    // Should complete in reasonable time (< 1s in release)
    #[cfg(not(debug_assertions))]
    assert!(
        elapsed.as_millis() < 1000,
        "Should process 50K ticks in < 1s, took {:?}",
        elapsed
    );
}
