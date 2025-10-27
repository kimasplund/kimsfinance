//! Heikin-Ashi Transformation Tests
//!
//! Validates smoothed candle transformation algorithm.
//! Tests formula correctness, sequential dependencies, and known-good implementations.

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{execute_batch, GpuDevice, HeikinAshiBatch};

/// Known-good Heikin-Ashi calculation (CPU reference)
#[cfg(feature = "gpu")]
fn calculate_heikin_ashi_reference(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = open.len();
    let mut ha_open = vec![0.0; n];
    let mut ha_high = vec![0.0; n];
    let mut ha_low = vec![0.0; n];
    let mut ha_close = vec![0.0; n];

    for i in 0..n {
        // HA Close = (O + H + L + C) / 4
        ha_close[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;

        // HA Open = (previous HA Open + previous HA Close) / 2
        if i == 0 {
            ha_open[i] = (open[i] + close[i]) / 2.0;
        } else {
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0;
        }

        // HA High = max(H, HA Open, HA Close)
        ha_high[i] = high[i].max(ha_open[i]).max(ha_close[i]);

        // HA Low = min(L, HA Open, HA Close)
        ha_low[i] = low[i].min(ha_open[i]).min(ha_close[i]);
    }

    (ha_open, ha_high, ha_low, ha_close)
}

#[cfg(feature = "gpu")]
#[test]
fn test_heikin_ashi_formula_correctness() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Simple test data
    let open = vec![100.0, 102.0, 104.0, 103.0, 105.0];
    let high = vec![103.0, 105.0, 106.0, 105.0, 107.0];
    let low = vec![99.0, 101.0, 103.0, 102.0, 104.0];
    let close = vec![102.0, 104.0, 105.0, 104.0, 106.0];

    // Calculate reference (CPU)
    let (ref_ha_open, ref_ha_high, ref_ha_low, ref_ha_close) =
        calculate_heikin_ashi_reference(&open, &high, &low, &close);

    // Prepare GPU input: concatenated [open, high, low, close]
    let mut data = Vec::new();
    data.extend(&open);
    data.extend(&high);
    data.extend(&low);
    data.extend(&close);

    let mut batch = HeikinAshiBatch::new();
    batch.add_task(data, ());

    let results = execute_batch(&device, &batch)?;
    let ha_candles = &results[0];

    // Result format: [ha_open, ha_high, ha_low, ha_close] * n
    let n = open.len();
    assert_eq!(ha_candles.len(), n * 4, "Expected 4 values per candle");

    // Compare with reference
    let tolerance = 1e-6;
    for i in 0..n {
        let gpu_open = ha_candles[i];
        let gpu_high = ha_candles[n + i];
        let gpu_low = ha_candles[2 * n + i];
        let gpu_close = ha_candles[3 * n + i];

        assert!(
            (gpu_open - ref_ha_open[i]).abs() < tolerance,
            "HA Open mismatch at {}: GPU={}, CPU={}",
            i,
            gpu_open,
            ref_ha_open[i]
        );
        assert!(
            (gpu_high - ref_ha_high[i]).abs() < tolerance,
            "HA High mismatch at {}: GPU={}, CPU={}",
            i,
            gpu_high,
            ref_ha_high[i]
        );
        assert!(
            (gpu_low - ref_ha_low[i]).abs() < tolerance,
            "HA Low mismatch at {}: GPU={}, CPU={}",
            i,
            gpu_low,
            ref_ha_low[i]
        );
        assert!(
            (gpu_close - ref_ha_close[i]).abs() < tolerance,
            "HA Close mismatch at {}: GPU={}, CPU={}",
            i,
            gpu_close,
            ref_ha_close[i]
        );
    }

    println!("✅ Heikin-Ashi formula correctness test passed");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_heikin_ashi_smoothing_effect() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Create volatile candles
    let open = vec![100.0, 110.0, 95.0, 105.0, 90.0];
    let high = vec![115.0, 120.0, 105.0, 115.0, 100.0];
    let low = vec![95.0, 105.0, 90.0, 95.0, 85.0];
    let close = vec![110.0, 95.0, 105.0, 90.0, 98.0];

    let mut data = Vec::new();
    data.extend(&open);
    data.extend(&high);
    data.extend(&low);
    data.extend(&close);

    let mut batch = HeikinAshiBatch::new();
    batch.add_task(data, ());

    let results = execute_batch(&device, &batch)?;
    let ha_candles = &results[0];

    let n = open.len();

    // HA candles should be smoother (less volatile)
    // Calculate range for original vs HA
    let original_range: f64 = (0..n).map(|i| high[i] - low[i]).sum();
    let ha_range: f64 = (0..n)
        .map(|i| ha_candles[n + i] - ha_candles[2 * n + i])
        .sum();

    // HA should have smaller total range (smoothing effect)
    assert!(
        ha_range < original_range,
        "HA should smooth volatility: HA={:.2}, Original={:.2}",
        ha_range,
        original_range
    );

    println!("✅ Heikin-Ashi smoothing effect validated");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_heikin_ashi_sequential_dependency() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Create data where sequential calculation matters
    let open = vec![100.0; 10];
    let high = vec![105.0; 10];
    let low = vec![95.0; 10];
    let close = vec![102.0; 10];

    let mut data = Vec::new();
    data.extend(&open);
    data.extend(&high);
    data.extend(&low);
    data.extend(&close);

    let mut batch = HeikinAshiBatch::new();
    batch.add_task(data, ());

    let results = execute_batch(&device, &batch)?;
    let ha_candles = &results[0];

    let n = open.len();

    // Verify sequential dependency: each HA Open depends on previous values
    // With constant inputs, HA values should converge
    let ha_open_9 = ha_candles[9];
    let ha_close_9 = ha_candles[3 * n + 9];

    // Last HA Open should reflect accumulated smoothing
    // With constant inputs, HA Open and HA Close should converge toward midpoint
    let midpoint = (open[0] + high[0] + low[0] + close[0]) / 4.0;
    let tolerance = 5.0; // Some divergence allowed due to sequential nature

    assert!(
        (ha_open_9 - midpoint).abs() < tolerance,
        "HA Open should converge: {:.2} vs {:.2}",
        ha_open_9,
        midpoint
    );

    println!("✅ Sequential dependency handling verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_heikin_ashi_first_candle() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    let open = vec![100.0];
    let high = vec![105.0];
    let low = vec![95.0];
    let close = vec![102.0];

    let mut data = Vec::new();
    data.extend(&open);
    data.extend(&high);
    data.extend(&low);
    data.extend(&close);

    let mut batch = HeikinAshiBatch::new();
    batch.add_task(data, ());

    let results = execute_batch(&device, &batch)?;
    let ha_candles = &results[0];

    assert_eq!(ha_candles.len(), 4, "Expected 4 values for single candle");

    let ha_open = ha_candles[0];
    let ha_high = ha_candles[1];
    let ha_low = ha_candles[2];
    let ha_close = ha_candles[3];

    // First HA Open = (Open + Close) / 2
    let expected_open = (open[0] + close[0]) / 2.0;
    assert_eq!(ha_open, expected_open, "First HA Open");

    // First HA Close = (O + H + L + C) / 4
    let expected_close = (open[0] + high[0] + low[0] + close[0]) / 4.0;
    assert_eq!(ha_close, expected_close, "First HA Close");

    // First HA High = max(H, HA Open, HA Close)
    let expected_high = high[0].max(ha_open).max(ha_close);
    assert_eq!(ha_high, expected_high, "First HA High");

    // First HA Low = min(L, HA Open, HA Close)
    let expected_low = low[0].min(ha_open).min(ha_close);
    assert_eq!(ha_low, expected_low, "First HA Low");

    println!("✅ First candle initialization verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_heikin_ashi_trend_detection() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Create uptrend data
    let open = vec![100.0, 102.0, 104.0, 106.0, 108.0];
    let high = vec![103.0, 105.0, 107.0, 109.0, 111.0];
    let low = vec![99.0, 101.0, 103.0, 105.0, 107.0];
    let close = vec![102.0, 104.0, 106.0, 108.0, 110.0];

    let mut data = Vec::new();
    data.extend(&open);
    data.extend(&high);
    data.extend(&low);
    data.extend(&close);

    let mut batch = HeikinAshiBatch::new();
    batch.add_task(data, ());

    let results = execute_batch(&device, &batch)?;
    let ha_candles = &results[0];

    let n = open.len();

    // In uptrend, HA candles should show:
    // 1. HA Close > HA Open (bullish)
    // 2. Small/no lower wicks (HA Low close to HA Open)
    for i in 1..n {
        let ha_open = ha_candles[i];
        let ha_low = ha_candles[2 * n + i];
        let ha_close = ha_candles[3 * n + i];

        assert!(
            ha_close > ha_open,
            "Uptrend: HA Close should be > HA Open at {}",
            i
        );

        // Small lower wick in strong uptrend
        let lower_wick = ha_open - ha_low;
        let body = ha_close - ha_open;
        assert!(
            lower_wick < body * 0.5,
            "Uptrend: Lower wick should be small at {}",
            i
        );
    }

    println!("✅ Trend detection with HA candles verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_heikin_ashi_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Process 3 different symbols in one batch
    let symbols = vec![
        // Symbol 1
        (vec![100.0, 102.0, 104.0], vec![103.0, 105.0, 106.0], vec![99.0, 101.0, 103.0], vec![102.0, 104.0, 105.0]),
        // Symbol 2
        (vec![50.0, 51.0, 52.0], vec![52.0, 53.0, 54.0], vec![49.0, 50.0, 51.0], vec![51.0, 52.0, 53.0]),
        // Symbol 3
        (vec![200.0, 198.0, 196.0], vec![202.0, 200.0, 198.0], vec![198.0, 196.0, 194.0], vec![199.0, 197.0, 195.0]),
    ];

    let mut batch = HeikinAshiBatch::new();

    for (open, high, low, close) in symbols {
        let mut data = Vec::new();
        data.extend(&open);
        data.extend(&high);
        data.extend(&low);
        data.extend(&close);
        batch.add_task(data, ());
    }

    let results = execute_batch(&device, &batch)?;

    assert_eq!(results.len(), 3, "Expected 3 results");

    // Verify each symbol processed correctly
    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.len(), 12, "Symbol {} should have 12 values (3 candles * 4)", i);
        println!("Symbol {} HA candles computed: {:?}", i + 1, &result[..4]);
    }

    println!("✅ Batch processing verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_heikin_ashi_large_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let device = GpuDevice::new()?;

    // Test with larger dataset (1000 candles)
    let n = 1000;
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);

    // Generate synthetic price movement
    let mut price = 100.0;
    for i in 0..n {
        let trend = (i as f64 / 100.0).sin() * 5.0;
        price += trend;

        open.push(price);
        high.push(price + 2.0);
        low.push(price - 2.0);
        close.push(price + trend);
    }

    let mut data = Vec::new();
    data.extend(&open);
    data.extend(&high);
    data.extend(&low);
    data.extend(&close);

    let mut batch = HeikinAshiBatch::new();
    batch.add_task(data, ());

    let results = execute_batch(&device, &batch)?;
    let ha_candles = &results[0];

    assert_eq!(ha_candles.len(), n * 4, "Expected 4000 values");

    // Verify no NaN values
    for (i, &value) in ha_candles.iter().enumerate() {
        assert!(!value.is_nan(), "NaN found at index {}", i);
        assert!(value.is_finite(), "Infinite value at index {}", i);
    }

    println!("✅ Large dataset (1000 candles) processed successfully");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_heikin_ashi_gpu_feature_required() {
    println!("⚠️  Heikin-Ashi tests require 'gpu' feature");
}
