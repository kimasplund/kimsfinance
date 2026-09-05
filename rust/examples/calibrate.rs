//! Manual Auto-Tuner Calibration Tool
//!
//! Runs micro-benchmarks to detect optimal CPU vs GPU crossover thresholds.
//!
//! # Usage
//!
//! ```bash
//! # Build with GPU support
//! cargo build --release --features gpu --example calibrate
//!
//! # Run calibration
//! ./target/release/examples/calibrate
//!
//! # View results
//! cat ~/.cache/kimsfinance/autotune.json
//! ```

#[cfg(feature = "gpu")]
use kimsfinance_core::autotuner::AutoTuneProfile;
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  kimsfinance Auto-Tuner Calibration Tool");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Check for existing cache
    if let Some(cached) = AutoTuneProfile::load_from_cache() {
        println!("⚠️  Found existing calibration cache:");
        println!("   Hardware ID: {}", cached.hardware_id);
        println!("   CPU: {:.2} GHz", cached.cpu_clock_ghz);
        println!("   GPU: {:.2} GHz", cached.gpu_clock_ghz);
        println!(
            "   Calibration date: {}",
            format_timestamp(cached.calibration_timestamp)
        );
        println!();

        println!("Do you want to re-calibrate? (y/N): ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Calibration cancelled. Using existing cache.");
            return Ok(());
        }

        println!();
    }

    // Initialize GPU
    println!("Initializing GPU...");
    let device = GpuDevice::new()?;
    println!("✅ GPU initialized");
    println!();

    // Run calibration
    println!("Starting calibration (this may take 2-5 seconds)...");
    println!();

    let profile = AutoTuneProfile::calibrate(&device)?;

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  Calibration Complete");
    println!("═══════════════════════════════════════════════════════");
    println!();

    println!("📊 Hardware Profile:");
    println!("   Hardware ID: {}", profile.hardware_id);
    println!("   CPU: {:.2} GHz", profile.cpu_clock_ghz);
    println!("   GPU: {:.2} GHz (boost)", profile.gpu_clock_ghz);
    println!("   VRAM: {:.0} GB/s", profile.vram_bandwidth_gbs);
    println!("   RAM: {:.0} GB/s", profile.ram_bandwidth_gbs);
    println!();

    println!("🎯 Optimal Thresholds:");
    println!(
        "   Stochastic: {} candles",
        profile.thresholds.stochastic_crossover
    );
    println!("   ROC: {} candles", profile.thresholds.roc_crossover);
    println!(
        "   Williams %R: {} candles",
        profile.thresholds.williams_r_crossover
    );
    println!(
        "   Bollinger: {} candles",
        profile.thresholds.bollinger_crossover
    );
    println!("   MACD: {} candles", profile.thresholds.macd_crossover);
    println!(
        "   Parallel ops: {} elements",
        profile.thresholds.parallel_operations
    );
    println!();

    println!("💾 Cache saved to:");
    // AutoTuneProfile::save_to_cache writes ~/.cache/kimsfinance/autotune.json
    let cache_file = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache")
        .join("kimsfinance")
        .join("autotune.json");
    println!("   {}", cache_file.display());
    println!();

    println!("✅ Calibration successful!");
    println!();
    println!("Next steps:");
    println!(
        "  1. Run auto-tuner demo: cargo run --release --features gpu --example autotuner_demo"
    );
    println!("  2. Use in your code: AutoTuneProfile::get_or_init(&device)");
    println!();

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Error: This tool requires --features gpu");
    eprintln!();
    eprintln!("Build with:");
    eprintln!("  cargo build --release --features gpu --example calibrate");
    std::process::exit(1);
}

#[cfg(feature = "gpu")]
fn format_timestamp(unix_timestamp: u64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = std::time::Duration::from_secs(unix_timestamp);
    let datetime = UNIX_EPOCH + duration;

    // Simple formatting (could use chrono crate for better formatting)
    match datetime.duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let days = d.as_secs() / 86400;
            format!(
                "{} days ago",
                (SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    / 86400)
                    .saturating_sub(days)
            )
        }
        Err(_) => "unknown".to_string(),
    }
}
