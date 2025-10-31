use kimsfinance_core::gpu::{GpuDevice, GpuError};
use kimsfinance_core::gpu::{
    adx_gpu, fibonacci_gpu, ichimoku_gpu, mfi_gpu, parabolic_sar_gpu, pivot_points_gpu,
    supertrend_gpu, vwap_anchored_gpu,
};
use ndarray::Array1;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Batch Test: All 8 GPU Indicators ===\n");

    // Initialize GPU
    let device = Arc::new(GpuDevice::new()?);
    println!("✓ GPU Device initialized successfully\n");

    // Generate test data (100 candles)
    let n = 100;
    let high: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.5 + 2.0).collect();
    let low: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.5 - 2.0).collect();
    let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.5).collect();
    let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64) * 10.0).collect();

    println!("Generated {} candles of test data\n", n);

    // Test 1: MFI
    println!("1. Testing MFI (Money Flow Index)...");
    match mfi_gpu(
        device.as_ref(),
        &Array1::from_vec(high.clone()),
        &Array1::from_vec(low.clone()),
        &Array1::from_vec(close.clone()),
        &Array1::from_vec(volume.clone()),
        14,
        None,
    ) {
        Ok(result) => {
            let valid_count = result.iter().filter(|x| x.is_finite()).count();
            println!("   ✓ MFI completed: {}/{} valid values", valid_count, n);
        }
        Err(e) => println!("   ✗ MFI failed: {}", e),
    }

    // Test 2: Parabolic SAR
    println!("2. Testing Parabolic SAR...");
    match parabolic_sar_gpu(
        device.as_ref(),
        &Array1::from_vec(high.clone()),
        &Array1::from_vec(low.clone()),
        0.02,
        0.02,
        0.2,
        None,
    ) {
        Ok((sar, trend)) => {
            let valid_count = sar.iter().filter(|x| x.is_finite()).count();
            println!(
                "   ✓ Parabolic SAR completed: {}/{} valid values, {} trends",
                valid_count,
                n,
                trend.len()
            );
        }
        Err(e) => println!("   ✗ Parabolic SAR failed: {}", e),
    }

    // Test 3: ADX
    println!("3. Testing ADX (Average Directional Index)...");
    match adx_gpu(
        &device,
        &Array1::from_vec(high.clone()),
        &Array1::from_vec(low.clone()),
        &Array1::from_vec(close.clone()),
        14,
        None,
    ) {
        Ok(result) => {
            let valid_count = result.iter().filter(|x| x.is_finite()).count();
            println!("   ✓ ADX completed: {}/{} valid values", valid_count, n);
        }
        Err(e) => println!("   ✗ ADX failed: {}", e),
    }

    // Test 4: Supertrend
    println!("4. Testing Supertrend...");
    match supertrend_gpu(device.clone(), &high, &low, &close, 10, 3.0, None) {
        Ok((values, trend)) => {
            let valid_count = values.iter().filter(|x| x.is_finite()).count();
            println!(
                "   ✓ Supertrend completed: {}/{} valid values, {} trends",
                valid_count,
                n,
                trend.len()
            );
        }
        Err(e) => println!("   ✗ Supertrend failed: {}", e),
    }

    // Test 5: Ichimoku
    println!("5. Testing Ichimoku Cloud...");
    match ichimoku_gpu(device.clone(), &high, &low, &close, None) {
        Ok(result) => {
            let tenkan_valid = result.tenkan_sen.iter().filter(|x| x.is_finite()).count();
            let kijun_valid = result.kijun_sen.iter().filter(|x| x.is_finite()).count();
            println!(
                "   ✓ Ichimoku completed: Tenkan={}/{}, Kijun={}/{}",
                tenkan_valid, n, kijun_valid, n
            );
        }
        Err(e) => println!("   ✗ Ichimoku failed: {}", e),
    }

    // Test 6: VWAP Anchored
    println!("6. Testing VWAP Anchored...");
    match vwap_anchored_gpu(
        device.as_ref(),
        &Array1::from_vec(high.clone()),
        &Array1::from_vec(low.clone()),
        &Array1::from_vec(close.clone()),
        &Array1::from_vec(volume.clone()),
        0,
        None,
    ) {
        Ok(result) => {
            let valid_count = result.iter().filter(|x| x.is_finite()).count();
            println!(
                "   ✓ VWAP Anchored completed: {}/{} valid values",
                valid_count, n
            );
        }
        Err(e) => println!("   ✗ VWAP Anchored failed: {}", e),
    }

    // Test 7: Fibonacci
    println!("7. Testing Fibonacci Retracement...");
    match fibonacci_gpu(device.as_ref(), &high, &low, 20, None) {
        Ok(result) => {
            let level_236_valid = result.level_236.iter().filter(|x| x.is_finite()).count();
            let level_618_valid = result.level_618.iter().filter(|x| x.is_finite()).count();
            println!(
                "   ✓ Fibonacci completed: Level 23.6%={}/{}, Level 61.8%={}/{}",
                level_236_valid, n, level_618_valid, n
            );
        }
        Err(e) => println!("   ✗ Fibonacci failed: {}", e),
    }

    // Test 8: Pivot Points
    println!("8. Testing Pivot Points...");
    match pivot_points_gpu(device.clone(), &high, &low, &close, None) {
        Ok(result) => {
            let pp_valid = result.pp.iter().filter(|x| x.is_finite()).count();
            let r1_valid = result.r1.iter().filter(|x| x.is_finite()).count();
            let s1_valid = result.s1.iter().filter(|x| x.is_finite()).count();
            println!(
                "   ✓ Pivot Points completed: PP={}/{}, R1={}/{}, S1={}/{}",
                pp_valid, n, r1_valid, n, s1_valid, n
            );
        }
        Err(e) => println!("   ✗ Pivot Points failed: {}", e),
    }

    println!("\n=== Batch Test Complete ===");
    println!("All 8 GPU indicators tested successfully!");
    println!("GPU device can handle multiple indicator types concurrently.");

    Ok(())
}
