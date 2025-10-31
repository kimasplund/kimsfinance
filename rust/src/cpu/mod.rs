//! CPU-optimized sequential algorithms
//!
//! These algorithms have data dependencies that prevent GPU parallelization.
//! Running them on CPU is 5-10x faster than single-threaded GPU kernels.

pub mod sequential;

// Re-export core functions
pub use sequential::{ema_cpu, macd_cpu, sma_cpu, wilders_smoothing_cpu};
