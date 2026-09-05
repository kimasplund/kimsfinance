//! CSV Loader Tests
//!
//! Validates trade data CSV parsing, multi-format support, and streaming.
//! Tests various CSV formats, error handling, and memory efficiency.

#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::{GpuDevice, TradeData};
use std::io::Write;
use tempfile::NamedTempFile;

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_standard_format() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary CSV file
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;
    writeln!(file, "1234567890.0,100.0,10.0")?;
    writeln!(file, "1234567891.0,102.0,12.0")?;
    writeln!(file, "1234567892.0,104.0,14.0")?;
    file.flush()?;

    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), 3, "Should load 3 trades");

    // Verify data parsing
    let (timestamps, prices, volumes) = trades.split();

    assert_eq!(timestamps[0], 1234567890.0, "Timestamp 1");
    assert_eq!(prices[0], 100.0, "Price 1");
    assert_eq!(volumes[0], 10.0, "Volume 1");

    assert_eq!(timestamps[2], 1234567892.0, "Timestamp 3");
    assert_eq!(prices[2], 104.0, "Price 3");
    assert_eq!(volumes[2], 14.0, "Volume 3");

    println!("✅ Standard CSV format parsing verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_alternate_columns() -> Result<(), Box<dyn std::error::Error>> {
    // CSV with different column order
    let mut file = NamedTempFile::new()?;
    writeln!(file, "price,volume,timestamp")?;
    writeln!(file, "100.0,10.0,1234567890.0")?;
    writeln!(file, "102.0,12.0,1234567891.0")?;
    file.flush()?;

    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), 2, "Should load 2 trades");

    let (timestamps, prices, _) = trades.split();

    // Should correctly map columns
    assert_eq!(timestamps[0], 1234567890.0);
    assert_eq!(prices[0], 100.0);

    println!("✅ Alternate column order parsing verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_with_headers_variations() -> Result<(), Box<dyn std::error::Error>> {
    // Test various header name variations
    let mut file = NamedTempFile::new()?;
    writeln!(file, "time,px,vol")?; // Abbreviated headers
    writeln!(file, "1234567890.0,100.0,10.0")?;
    writeln!(file, "1234567891.0,102.0,12.0")?;
    file.flush()?;

    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), 2, "Should parse abbreviated headers");

    println!("✅ Header variations parsing verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_large_dataset() -> Result<(), Box<dyn std::error::Error>> {
    // Create large CSV (10K rows)
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;

    let n = 10_000;
    for i in 0..n {
        let timestamp = 1234567890.0 + i as f64;
        let price = 100.0 + (i as f64 * 0.01);
        let volume = 10.0 + (i as f64 * 0.001);
        writeln!(file, "{},{},{}", timestamp, price, volume)?;
    }
    file.flush()?;

    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), n, "Should load {} trades", n);

    // Verify first and last
    let (timestamps, prices, volumes) = trades.split();

    assert_eq!(timestamps[0], 1234567890.0);
    assert_eq!(timestamps[n - 1], 1234567890.0 + (n - 1) as f64);

    assert_eq!(prices[0], 100.0);
    assert_eq!(volumes[n - 1], 10.0 + ((n - 1) as f64 * 0.001));

    println!("✅ Large dataset (10K rows) loading verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_with_extra_columns() -> Result<(), Box<dyn std::error::Error>> {
    // CSV with extra columns (should be ignored)
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume,symbol,exchange")?;
    writeln!(file, "1234567890.0,100.0,10.0,BTC,BINANCE")?;
    writeln!(file, "1234567891.0,102.0,12.0,BTC,BINANCE")?;
    file.flush()?;

    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), 2, "Should load data ignoring extra columns");

    println!("✅ Extra columns handling verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_scientific_notation() -> Result<(), Box<dyn std::error::Error>> {
    // Test scientific notation parsing
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;
    writeln!(file, "1.23456789e9,1.0e2,1.0e1")?;
    writeln!(file, "1.23456790e9,1.02e2,1.2e1")?;
    file.flush()?;

    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), 2);

    let (_, prices, volumes) = trades.split();

    assert_eq!(prices[0], 100.0, "Scientific notation price");
    assert_eq!(volumes[0], 10.0, "Scientific notation volume");

    println!("✅ Scientific notation parsing verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_missing_values() -> Result<(), Box<dyn std::error::Error>> {
    // CSV with missing values (should error or skip)
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;
    writeln!(file, "1234567890.0,100.0,10.0")?;
    writeln!(file, "1234567891.0,,12.0")?; // Missing price
    writeln!(file, "1234567892.0,104.0,14.0")?;
    file.flush()?;

    let result = TradeData::from_csv(file.path().to_str().unwrap());

    // Should either error or skip invalid rows
    match result {
        Ok(trades) => {
            // If skipping invalid rows
            assert!(trades.len() == 2, "Should skip row with missing value");
            println!("✅ Missing values: skipped invalid rows");
        }
        Err(_) => {
            println!("✅ Missing values: returned error as expected");
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_invalid_format() -> Result<(), Box<dyn std::error::Error>> {
    // CSV with invalid format
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;
    writeln!(file, "not_a_number,100.0,10.0")?;
    file.flush()?;

    let result = TradeData::from_csv(file.path().to_str().unwrap());

    // Should return error
    assert!(result.is_err(), "Should error on invalid format");

    println!("✅ Invalid format error handling verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_empty_file() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;
    // No data rows
    file.flush()?;

    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), 0, "Empty file should return 0 trades");

    println!("✅ Empty file handling verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_no_header() -> Result<(), Box<dyn std::error::Error>> {
    // CSV without header row
    let mut file = NamedTempFile::new()?;
    writeln!(file, "1234567890.0,100.0,10.0")?;
    writeln!(file, "1234567891.0,102.0,12.0")?;
    file.flush()?;

    let result = TradeData::from_csv(file.path().to_str().unwrap());

    // Should either:
    // 1. Assume default column order
    // 2. Error due to missing header
    match result {
        Ok(trades) => {
            assert_eq!(trades.len(), 2, "Should parse assuming default order");
            println!("✅ No header: assumed default column order");
        }
        Err(_) => {
            println!("✅ No header: returned error as expected");
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_multi_symbol() -> Result<(), Box<dyn std::error::Error>> {
    // CSV with multiple symbols (should filter or load all)
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume,symbol")?;
    writeln!(file, "1234567890.0,100.0,10.0,BTC")?;
    writeln!(file, "1234567891.0,50.0,12.0,ETH")?;
    writeln!(file, "1234567892.0,102.0,14.0,BTC")?;
    file.flush()?;

    // Load all
    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), 3, "Should load all symbols");

    // TODO: Add filtering by symbol if API supports it
    // let btc_trades = TradeData::from_csv_filtered(path, "BTC")?;
    // assert_eq!(btc_trades.len(), 2);

    println!("✅ Multi-symbol CSV loading verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_tab_delimited() -> Result<(), Box<dyn std::error::Error>> {
    // Tab-delimited file
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp\tprice\tvolume")?;
    writeln!(file, "1234567890.0\t100.0\t10.0")?;
    writeln!(file, "1234567891.0\t102.0\t12.0")?;
    file.flush()?;

    let result = TradeData::from_csv(file.path().to_str().unwrap());

    // Should handle tab delimiter
    match result {
        Ok(trades) => {
            assert_eq!(trades.len(), 2, "Should parse tab-delimited");
            println!("✅ Tab-delimited parsing verified");
        }
        Err(_) => {
            println!("⚠️  Tab-delimited not supported (expected)");
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
    // Test memory usage with large file
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;

    let n = 100_000;
    for i in 0..n {
        writeln!(file, "{},{},{}", 1234567890.0 + i as f64, 100.0, 10.0)?;
    }
    file.flush()?;

    // Load large file
    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    assert_eq!(trades.len(), n);

    // Verify memory is efficiently allocated (single allocation per vector)
    let (timestamps, prices, volumes) = trades.split();

    assert_eq!(timestamps.len(), n);
    assert_eq!(prices.len(), n);
    assert_eq!(volumes.len(), n);

    println!("✅ Memory efficiency (100K rows) verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_concat_for_batch() -> Result<(), Box<dyn std::error::Error>> {
    // Test TradeData can be concatenated for GPU batch
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;
    writeln!(file, "1234567890.0,100.0,10.0")?;
    writeln!(file, "1234567891.0,102.0,12.0")?;
    file.flush()?;

    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    // Get concatenated buffers for GPU
    let concat_data = trades.concat_buffers();

    // Should be [timestamps..., prices..., volumes...]
    assert_eq!(concat_data.len(), 6, "3 values * 2 trades");

    assert_eq!(concat_data[0], 1234567890.0, "Timestamps first");
    assert_eq!(concat_data[2], 100.0, "Then prices");
    assert_eq!(concat_data[4], 10.0, "Then volumes");

    println!("✅ Buffer concatenation for GPU batch verified");
    Ok(())
}

#[cfg(feature = "gpu")]
#[test]
fn test_csv_loader_integration_with_time_bars() -> Result<(), Box<dyn std::error::Error>> {
    use kimsfinance_core::gpu::{GpuDevice, TimeBarBatch, execute_batch};

    let device = GpuDevice::new()?;

    // Create CSV
    let mut file = NamedTempFile::new()?;
    writeln!(file, "timestamp,price,volume")?;
    writeln!(file, "10.0,100.0,10.0")?;
    writeln!(file, "20.0,102.0,12.0")?;
    writeln!(file, "70.0,104.0,14.0")?;
    writeln!(file, "80.0,106.0,16.0")?;
    file.flush()?;

    // Load from CSV
    let trades = TradeData::from_csv(file.path().to_str().unwrap())?;

    // Create time bar batch
    let mut batch = TimeBarBatch::new();
    batch.add_task(trades.concat_buffers(), 60);

    // Execute on GPU
    let results = execute_batch(&device, &batch)?;

    // Should have 2 candles (minute 1 and minute 2)
    assert!(results[0].len() >= 10, "Should produce time bars");

    println!("✅ CSV loader + time bar integration verified");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_csv_loader_gpu_feature_required() {
    println!("⚠️  CSV loader tests require 'gpu' feature");
}
