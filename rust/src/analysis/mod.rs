//! Analysis modules for trade data
//!
//! This module provides advanced analysis tools for processing
//! tick-level trade data, including volume profile analysis,
//! order flow analysis, and other market microstructure tools.

pub mod volume_profile;
pub mod microstructure;

// Re-export main types
pub use volume_profile::{PriceLevel, VolumeProfile, VolumeProfileBuilder};
pub use microstructure::{MicrostructureAnalyzer, MicrostructureMetrics};
