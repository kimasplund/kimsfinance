//! Auto-Tuner Demo
//!
//! Demonstrates adaptive CPU vs GPU selection using the auto-tuner.
//!
//! # Usage
//!
//! ```bash
//! # Compile with GPU support
//! cargo build --release --features gpu --example autotuner_demo
//!
//! # Run demo
//! ./target/release/examples/autotuner_demo
//!
//! # Force CPU-only mode
//! KIMSFINANCE_FORCE_CPU=1 ./target/release/examples/autotuner_demo
//!
//! # Clear cache and re-calibrate
//! rm ~/.cache/kimsfinance/autotune.json
//! ./target/release/examples/autotuner_demo
//! ```

#[cfg(feature = "gpu")]
use kimsfinance_core::autotuner::{AutoTuneProfile, ExecutionStrategy};
#[cfg(feature = "gpu")]
use kimsfinance_core::gpu::GpuDevice;
#[cfg(feature = "gpu")]
use ndarray::Array1;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  kimsfinance Auto-Tuner Demo");
    println!("═══════════════════════════════════════════════════════");
    println!();

    // Initialize GPU device
    println!("1️⃣  Initializing GPU device...");
    let device = GpuDevice::new()?;
    println!("   ✅ GPU initialized");
    println!();

    // Get or initialize auto-tune profile
    println!("2️⃣  Loading auto-tune profile...");
    let profile = AutoTuneProfile::get_or_init(&device);
    println!();

    // Display hardware specs
    println!("📊 Hardware Configuration:");
    println!("   CPU: {:.2} GHz", profile.cpu_clock_ghz);
    println!("   GPU: {:.2} GHz (boost)", profile.gpu_clock_ghz);
    println!("   VRAM: {:.0} GB/s", profile.vram_bandwidth_gbs);
    println!("   RAM: {:.0} GB/s", profile.ram_bandwidth_gbs);
    println!(
        "   VRAM/RAM ratio: {:.1}x",
        profile.vram_bandwidth_gbs / profile.ram_bandwidth_gbs
    );
    println!();

    // Display calibrated thresholds
    println!("🎯 Calibrated Crossover Thresholds:");
    println!(
        "   EMA: {} (always CPU)",
        if profile.thresholds.ema_crossover == usize::MAX {
            "N/A".to_string()
        } else {
            profile.thresholds.ema_crossover.to_string()
        }
    );
    println!(
        "   Wilder's (RSI/ATR): {} (always CPU for sequential part)",
        if profile.thresholds.wilders_crossover == usize::MAX {
            "N/A".to_string()
        } else {
            profile.thresholds.wilders_crossover.to_string()
        }
    );
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

    // Demonstrate adaptive selection
    println!("🔀 Adaptive Strategy Selection:");
    println!();

    let test_sizes = vec![100, 1_000, 5_000, 10_000, 50_000, 100_000];

    println!("   Data Size  │  EMA  │  RSI  │ Stoch │  ROC  │ Will%R│  MACD");
    println!("   ───────────┼───────┼───────┼───────┼───────┼───────┼───────");

    for &size in &test_sizes {
        let ema = format_strategy(profile.select_ema_strategy(size));
        let rsi = format_strategy(profile.select_rsi_strategy(size));
        let stoch = format_strategy(profile.select_stochastic_strategy(size));
        let roc = format_strategy(profile.select_roc_strategy(size));
        let williams = format_strategy(profile.select_williams_r_strategy(size));
        let macd = format_strategy(profile.select_macd_strategy(size));

        println!(
            "   {:>10} │ {:^5} │ {:^5} │ {:^5} │ {:^5} │ {:^5} │ {:^5}",
            format_number(size),
            ema,
            rsi,
            stoch,
            roc,
            williams,
            macd
        );
    }

    println!();
    println!("Legend: CPU = Run on CPU, GPU = Run on GPU (or hybrid)");
    println!();

    // Example: Process data with auto-selected strategy
    println!("3️⃣  Example: Processing 100K candles of RSI");
    let data_size = 100_000;
    let close = Array1::from_vec((0..data_size).map(|i| 100.0 + (i as f64 * 0.01)).collect());

    match profile.select_rsi_strategy(data_size) {
        ExecutionStrategy::CPU => {
            println!("   Strategy: CPU-only");
            println!("   Reason: Small dataset or CPU faster for this size");
        }
        ExecutionStrategy::GPU => {
            println!("   Strategy: GPU hybrid (GPU→CPU→GPU pipeline)");
            println!("   Reason: Large dataset, GPU parallel ops dominate");
            println!("   Pipeline: GPU gains/losses → CPU Wilder's → GPU RSI");
        }
        ExecutionStrategy::Hybrid => {
            println!("   Strategy: Custom hybrid");
        }
    }

    println!();
    println!("✅ Demo complete!");
    println!();
    println!("💡 Tips:");
    println!("   • Cache location: ~/.cache/kimsfinance/autotune.json");
    println!("   • Force CPU: export KIMSFINANCE_FORCE_CPU=1");
    println!("   • Re-calibrate: rm ~/.cache/kimsfinance/autotune.json");
    println!();

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Error: This example requires --features gpu");
    eprintln!();
    eprintln!("Build with:");
    eprintln!("  cargo build --release --features gpu --example autotuner_demo");
    std::process::exit(1);
}

#[cfg(feature = "gpu")]
fn format_strategy(strategy: ExecutionStrategy) -> &'static str {
    match strategy {
        ExecutionStrategy::CPU => "CPU",
        ExecutionStrategy::GPU => "GPU",
        ExecutionStrategy::Hybrid => "HYB",
    }
}

#[cfg(feature = "gpu")]
fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}
