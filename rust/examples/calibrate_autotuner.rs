//! Calibrate auto-tuner for this machine
//!
//! Runs hardware detection and micro-benchmarks to find optimal thresholds
//! for GPU vs CPU selection and backtest optimizations.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example calibrate_autotuner --features gpu
//! ```
//!
//! # What It Does
//!
//! 1. **Hardware Detection**:
//!    - CPU clock speed (GHz)
//!    - GPU clock speed (GHz)
//!    - VRAM bandwidth (GB/s)
//!    - RAM bandwidth (GB/s)
//!
//! 2. **Indicator Threshold Calibration**:
//!    - Tests each indicator (RSI, Stochastic, ROC, etc.)
//!    - Finds crossover point where GPU becomes faster
//!    - Validates with 10 iterations per size
//!
//! 3. **Backtest Optimization Calibration** (NEW):
//!    - SIMD Sharpe ratio: Find crossover for AVX2 vs scalar
//!    - Parallel evaluation: Validate threshold for rayon
//!    - HashMap pre-allocation: Disabled (investigation found never beneficial)
//!
//! 4. **Caching**:
//!    - Results saved to `~/.cache/kimsfinance/autotune.json`
//!    - Used automatically by all future runs
//!    - Re-calibrate only if hardware changes
//!
//! # Performance Impact
//!
//! - Calibration time: 30-60 seconds (one-time cost)
//! - Future runs: <1ms to load cached profile
//! - Optimal selection: Always uses fastest implementation
//!
//! # Output
//!
//! ```text
//! 🔧 Calibrating kimsfinance for this machine...
//!
//! 📊 Hardware detected:
//!    CPU: 3.50 GHz
//!    GPU: 3.11 GHz (boost)
//!    VRAM: 288 GB/s
//!    RAM: 77 GB/s
//!
//! ⏱️  Benchmarking crossover points...
//!    Stochastic crossover: 5000 candles (GPU: 150μs, CPU: 200μs)
//!    ROC crossover: 2000 candles (GPU: 80μs, CPU: 120μs)
//!    ...
//!
//! 📊 Calibrating backtest optimization thresholds...
//!    Testing SIMD Sharpe ratio crossover...
//!    ✓ SIMD faster at 10000 points (1.42x speedup: 250μs vs 355μs)
//!
//!    Testing parallel evaluation threshold...
//!    ✓ Using validated threshold: 20 individuals
//!
//! ✅ Backtest thresholds calibrated:
//!    SIMD Sharpe threshold: 10000 points
//!    Parallel eval threshold: 20 individuals
//!
//! ✅ Calibration complete!
//!
//! Backtest Optimization Thresholds:
//!   SIMD Sharpe threshold: 10000 points
//!   Parallel eval threshold: 20 individuals
//!   HashMap pre-alloc: disabled (never beneficial)
//!
//! Cached to: ~/.cache/kimsfinance/autotune.json
//! Re-calibrate if hardware changes or performance degrades.
//! ```

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kimsfinance_core::autotuner::AutoTuneProfile;
    use kimsfinance_core::gpu::GpuDevice;

    println!("🔧 Calibrating kimsfinance for this machine...\n");

    // Initialize GPU device
    let device = GpuDevice::new()?;

    // This will detect hardware, run benchmarks, and cache results
    let profile = AutoTuneProfile::calibrate(&device)?;

    println!("\n✅ Calibration complete!\n");
    println!("Backtest Optimization Thresholds:");
    println!(
        "  SIMD Sharpe threshold: {} points",
        profile.backtest_thresholds.simd_sharpe_threshold
    );
    println!(
        "  Parallel eval threshold: {} individuals",
        profile.backtest_thresholds.parallel_eval_threshold
    );
    println!(
        "  HashMap pre-alloc: {} (investigation found never beneficial)",
        if profile.backtest_thresholds.use_hashmap_prealloc {
            "enabled"
        } else {
            "disabled"
        }
    );

    println!("\nCached to: ~/.cache/kimsfinance/autotune.json");
    println!("Re-calibrate if hardware changes or performance degrades.");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Error: This example requires the 'gpu' feature");
    eprintln!("Run with: cargo run --release --example calibrate_autotuner --features gpu");
    std::process::exit(1);
}
