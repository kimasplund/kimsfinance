//! Core traits and types for technical indicators
//!
//! This module provides the foundational trait system for all technical indicators,
//! enabling code reuse and consistent behavior across different indicator types.

use ndarray::{Array1, ArrayView1};

/// Result type for indicator calculations
pub type IndicatorResult = Result<Array1<f64>, IndicatorError>;

/// Multi-output result (e.g., MACD returns signal, histogram, etc)
pub type MultiResult = Result<IndicatorOutput, IndicatorError>;

/// Error types for indicator calculations
#[derive(Debug, Clone)]
pub enum IndicatorError {
    /// Insufficient data for the requested period
    InsufficientData { required: usize, got: usize },
    /// Invalid parameter value
    InvalidParameter { name: &'static str, value: String },
    /// Mismatched array lengths
    LengthMismatch { expected: usize, got: usize },
    /// Division by zero or invalid computation
    ComputationError(String),
}

impl std::fmt::Display for IndicatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData { required, got } => {
                write!(f, "Insufficient data: need {required}, got {got}")
            }
            Self::InvalidParameter { name, value } => {
                write!(f, "Invalid parameter {name}: {value}")
            }
            Self::LengthMismatch { expected, got } => {
                write!(f, "Length mismatch: expected {expected}, got {got}")
            }
            Self::ComputationError(msg) => write!(f, "Computation error: {msg}"),
        }
    }
}

impl std::error::Error for IndicatorError {}

/// Container for multi-output indicators (MACD, Bollinger Bands, etc)
#[derive(Debug, Clone)]
pub struct IndicatorOutput {
    /// Primary output
    pub primary: Array1<f64>,
    /// Secondary outputs (e.g., upper/lower bands, signal line)
    pub secondary: Vec<Array1<f64>>,
    /// Optional metadata
    pub metadata: Option<OutputMetadata>,
}

/// Metadata for indicator outputs
#[derive(Debug, Clone)]
pub struct OutputMetadata {
    /// Names for each output series
    pub names: Vec<String>,
    /// Custom key-value pairs
    pub extra: std::collections::HashMap<String, String>,
}

/// Core trait for all technical indicators
///
/// Implementors must provide single-output calculation.
/// Multi-output indicators should implement `MultiOutputIndicator`.
pub trait Indicator {
    /// Calculate indicator values
    ///
    /// Returns array of same length as input, with NaN for warmup period.
    fn calculate(&self, prices: ArrayView1<f64>) -> IndicatorResult;

    /// Minimum data points required (warmup period)
    fn min_periods(&self) -> usize;

    /// Indicator name for debugging/logging
    fn name(&self) -> &'static str;
}

/// Trait for indicators with multiple outputs (MACD, Bollinger Bands, etc)
pub trait MultiOutputIndicator {
    /// Calculate indicator with multiple outputs
    fn calculate_multi(&self, prices: ArrayView1<f64>) -> MultiResult;

    /// Minimum data points required
    fn min_periods(&self) -> usize;

    /// Indicator name
    fn name(&self) -> &'static str;
}

/// Trait for indicators requiring OHLCV data (not just close prices)
pub trait OHLCVIndicator {
    /// Calculate indicator from OHLCV data
    fn calculate_ohlcv(
        &self,
        high: ArrayView1<f64>,
        low: ArrayView1<f64>,
        open: ArrayView1<f64>,
        close: ArrayView1<f64>,
        volume: ArrayView1<f64>,
    ) -> IndicatorResult;

    fn min_periods(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Trait for volume-based indicators
pub trait VolumeIndicator {
    /// Calculate indicator from price and volume data
    fn calculate_with_volume(
        &self,
        prices: ArrayView1<f64>,
        volumes: ArrayView1<f64>,
    ) -> IndicatorResult;

    fn min_periods(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Helper to validate array lengths match
#[inline]
pub fn validate_lengths<'a>(arrays: &[ArrayView1<'a, f64>]) -> Result<usize, IndicatorError> {
    if arrays.is_empty() {
        return Err(IndicatorError::ComputationError(
            "No arrays provided".to_string(),
        ));
    }

    let len = arrays[0].len();
    for (_i, arr) in arrays.iter().enumerate().skip(1) {
        if arr.len() != len {
            return Err(IndicatorError::LengthMismatch {
                expected: len,
                got: arr.len(),
            });
        }
    }

    Ok(len)
}

/// Helper to validate sufficient data
#[inline]
pub fn validate_min_periods(data_len: usize, min_periods: usize) -> Result<(), IndicatorError> {
    if data_len < min_periods {
        return Err(IndicatorError::InsufficientData {
            required: min_periods,
            got: data_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_validate_lengths_match() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[4.0, 5.0, 6.0]);
        assert_eq!(validate_lengths(&[a.view(), b.view()]).unwrap(), 3);
    }

    #[test]
    fn test_validate_lengths_mismatch() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[4.0, 5.0]);
        assert!(validate_lengths(&[a.view(), b.view()]).is_err());
    }

    #[test]
    fn test_validate_min_periods_ok() {
        assert!(validate_min_periods(100, 14).is_ok());
    }

    #[test]
    fn test_validate_min_periods_insufficient() {
        assert!(validate_min_periods(10, 14).is_err());
    }
}
