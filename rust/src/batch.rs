//! Batch Indicator Calculation API
//!
//! Minimizes FFI overhead by processing multiple indicators in a single call.
//! Critical for datasets >1,000 rows where FFI crossing dominates performance.
//!
//! # Performance Strategy
//!
//! - **Individual calls**: Good for <1K rows (3-4x faster than Python)
//! - **Batch API**: Essential for >1K rows (avoids repeated FFI crossing)
//!
//! # Example
//!
//! ```python
//! import kimsfinance_core
//! import numpy as np
//!
//! # Single FFI call for multiple indicators
//! results = kimsfinance_core.calculate_indicators_batch({
//!     "high": high_prices,
//!     "low": low_prices,
//!     "open": open_prices,
//!     "close": close_prices,
//!     "volume": volume_data,
//!     "indicators": [
//!         {"name": "rsi", "period": 14},
//!         {"name": "macd", "fast": 12, "slow": 26, "signal": 9},
//!         {"name": "atr", "period": 14},
//!         {"name": "bollinger", "period": 20, "std_dev": 2.0},
//!     ]
//! })
//!
//! # Access results
//! rsi = results["rsi"]
//! macd_line = results["macd"]["line"]
//! macd_signal = results["macd"]["signal"]
//! macd_histogram = results["macd"]["histogram"]
//! ```

use ndarray::ArrayView1;
use std::collections::HashMap;

use crate::indicators::{
    ATR, Aroon, BollingerBands, CCI, CMF, DEMA, DonchianChannels, EMA, ElderRay, HMA, Indicator,
    IndicatorError, KeltnerChannels, MACD, MultiOutputIndicator, OBV, ParabolicSAR, PivotPoints,
    ROC, RSI, SMA, Stochastic, TEMA, TSI, VWAP, VWMA, VolumeProfile, WMA, WilliamsR,
};

/// Indicator request specification
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)] // Technical indicators use standard acronyms
pub enum IndicatorRequest {
    // Moving Averages
    SMA {
        period: usize,
    },
    EMA {
        period: usize,
    },
    WMA {
        period: usize,
    },
    VWMA {
        period: usize,
    },
    DEMA {
        period: usize,
    },
    TEMA {
        period: usize,
    },
    HMA {
        period: usize,
    },

    // Momentum
    RSI {
        period: usize,
    },
    ROC {
        period: usize,
    },
    WilliamsR {
        period: usize,
    },
    Stochastic {
        k_period: usize,
        d_period: usize,
    },
    Aroon {
        period: usize,
    },
    CCI {
        period: usize,
    },
    MACD {
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    },
    TSI {
        long_period: usize,
        short_period: usize,
        signal_period: usize,
    },

    // Volatility
    ATR {
        period: usize,
    },
    BollingerBands {
        period: usize,
        std_dev: f64,
    },
    KeltnerChannels {
        ema_period: usize,
        atr_period: usize,
        atr_multiplier: f64,
    },
    DonchianChannels {
        period: usize,
    },
    ElderRay {
        ema_period: usize,
    },

    // Volume
    OBV,
    VWAP,
    CMF {
        period: usize,
    },
    VolumeProfile {
        num_bins: usize,
    },

    // Trend
    ParabolicSAR {
        af_start: f64,
        af_increment: f64,
        af_max: f64,
    },
    PivotPoints,
}

/// OHLCV data container for batch processing
pub struct OHLCVBatch<'a> {
    pub high: ArrayView1<'a, f64>,
    pub low: ArrayView1<'a, f64>,
    #[allow(dead_code)] // May be used in future indicators
    pub open: ArrayView1<'a, f64>,
    pub close: ArrayView1<'a, f64>,
    pub volume: ArrayView1<'a, f64>,
}

/// Batch calculation result
pub type BatchResult = Result<HashMap<String, IndicatorBatchOutput>, IndicatorError>;

/// Output for a single indicator in batch mode
#[derive(Debug, Clone)]
pub enum IndicatorBatchOutput {
    /// Single array output
    Single(Vec<f64>),
    /// Multiple array outputs (e.g., MACD, Bollinger Bands)
    Multiple {
        primary: Vec<f64>,
        secondary: Vec<Vec<f64>>,
        names: Vec<String>,
    },
}

impl IndicatorBatchOutput {
    /// Create single-output result
    pub fn single(data: Vec<f64>) -> Self {
        Self::Single(data)
    }

    /// Create multi-output result
    pub fn multiple(primary: Vec<f64>, secondary: Vec<Vec<f64>>, names: Vec<String>) -> Self {
        Self::Multiple {
            primary,
            secondary,
            names,
        }
    }
}

/// Calculate multiple indicators in a single batch
///
/// This function minimizes FFI overhead by:
/// 1. Accepting all OHLCV data once
/// 2. Calculating all requested indicators in Rust
/// 3. Returning all results in a single structure
///
/// # Arguments
///
/// * `ohlcv` - OHLCV data container
/// * `requests` - List of indicator requests
///
/// # Returns
///
/// HashMap mapping indicator names to their outputs
///
/// # Example
///
/// ```rust
/// use ndarray::arr1;
/// use batch::{calculate_batch, OHLCVBatch, IndicatorRequest};
///
/// let ohlcv = OHLCVBatch {
///     high: arr1(&[110.0, 115.0]).view(),
///     low: arr1(&[105.0, 110.0]).view(),
///     open: arr1(&[108.0, 112.0]).view(),
///     close: arr1(&[108.0, 112.0]).view(),
///     volume: arr1(&[1000.0, 1500.0]).view(),
/// };
///
/// let requests = vec![
///     ("rsi_14".to_string(), IndicatorRequest::RSI { period: 14 }),
///     ("sma_20".to_string(), IndicatorRequest::SMA { period: 20 }),
/// ];
///
/// let results = calculate_batch(&ohlcv, requests).unwrap();
/// ```
pub fn calculate_batch(
    ohlcv: &OHLCVBatch,
    requests: Vec<(String, IndicatorRequest)>,
) -> BatchResult {
    let mut results = HashMap::new();

    for (name, request) in requests {
        let output = match request {
            // Moving Averages
            IndicatorRequest::SMA { period } => {
                let indicator = SMA::new(period)?;
                let result = indicator.calculate(ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::EMA { period } => {
                let indicator = EMA::new(period)?;
                let result = indicator.calculate(ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::WMA { period } => {
                let indicator = WMA::new(period)?;
                let result = indicator.calculate(ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::VWMA { period } => {
                let indicator = VWMA::new(period)?;
                let result = indicator.calculate_with_volume(ohlcv.close, ohlcv.volume)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::DEMA { period } => {
                let indicator = DEMA::new(period)?;
                let result = indicator.calculate(ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::TEMA { period } => {
                let indicator = TEMA::new(period)?;
                let result = indicator.calculate(ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::HMA { period } => {
                let indicator = HMA::new(period)?;
                let result = indicator.calculate(ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }

            // Momentum
            IndicatorRequest::RSI { period } => {
                let indicator = RSI::new(period)?;
                let result = indicator.calculate(ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::ROC { period } => {
                let indicator = ROC::new(period)?;
                let result = indicator.calculate(ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::WilliamsR { period } => {
                let indicator = WilliamsR::new(period)?;
                let result = indicator.calculate_hlc(ohlcv.high, ohlcv.low, ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::Stochastic { k_period, d_period } => {
                let indicator = Stochastic::new(k_period, d_period)?;
                let output = indicator.calculate_hlc(ohlcv.high, ohlcv.low, ohlcv.close)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec!["k".to_string(), "d".to_string()],
                )
            }
            IndicatorRequest::Aroon { period } => {
                let indicator = Aroon::new(period)?;
                let output = indicator.calculate_hl(ohlcv.high, ohlcv.low)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec!["up".to_string(), "down".to_string()],
                )
            }
            IndicatorRequest::CCI { period } => {
                let indicator = CCI::new(period)?;
                let result = indicator.calculate_hlc(ohlcv.high, ohlcv.low, ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::MACD {
                fast_period,
                slow_period,
                signal_period,
            } => {
                let indicator = MACD::new(fast_period, slow_period, signal_period)?;
                let output = indicator.calculate_multi(ohlcv.close)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec![
                        "line".to_string(),
                        "signal".to_string(),
                        "histogram".to_string(),
                    ],
                )
            }
            IndicatorRequest::TSI {
                long_period,
                short_period,
                signal_period,
            } => {
                let indicator = TSI::new(long_period, short_period, signal_period)?;
                let output = indicator.calculate_multi(ohlcv.close)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec!["tsi".to_string(), "signal".to_string()],
                )
            }

            // Volatility
            IndicatorRequest::ATR { period } => {
                let indicator = ATR::new(period)?;
                let result = indicator.calculate_hlc(ohlcv.high, ohlcv.low, ohlcv.close)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::BollingerBands { period, std_dev } => {
                let indicator = BollingerBands::new(period, std_dev)?;
                let output = indicator.calculate_multi(ohlcv.close)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec![
                        "middle".to_string(),
                        "upper".to_string(),
                        "lower".to_string(),
                    ],
                )
            }
            IndicatorRequest::KeltnerChannels {
                ema_period,
                atr_period,
                atr_multiplier,
            } => {
                let indicator = KeltnerChannels::new(ema_period, atr_period, atr_multiplier)?;
                let output = indicator.calculate_hlc(ohlcv.high, ohlcv.low, ohlcv.close)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec![
                        "middle".to_string(),
                        "upper".to_string(),
                        "lower".to_string(),
                    ],
                )
            }
            IndicatorRequest::DonchianChannels { period } => {
                let indicator = DonchianChannels::new(period)?;
                let output = indicator.calculate_hl(ohlcv.high, ohlcv.low)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec![
                        "middle".to_string(),
                        "upper".to_string(),
                        "lower".to_string(),
                    ],
                )
            }
            IndicatorRequest::ElderRay { ema_period } => {
                let indicator = ElderRay::new(ema_period)?;
                let output = indicator.calculate_hlc(ohlcv.high, ohlcv.low, ohlcv.close)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec!["bull_power".to_string(), "bear_power".to_string()],
                )
            }

            // Volume
            IndicatorRequest::OBV => {
                let indicator = OBV::new();
                let result = indicator.calculate_with_volume(ohlcv.close, ohlcv.volume)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::VWAP => {
                let indicator = VWAP::new();
                let result =
                    indicator.calculate_hlcv(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::CMF { period } => {
                let indicator = CMF::new(period)?;
                let result =
                    indicator.calculate_hlcv(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::VolumeProfile { num_bins } => {
                let indicator = VolumeProfile::new(num_bins)?;
                let result =
                    indicator.calculate_hlcv(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)?;
                IndicatorBatchOutput::single(result.to_vec())
            }

            // Trend
            IndicatorRequest::ParabolicSAR {
                af_start,
                af_increment,
                af_max,
            } => {
                let indicator = ParabolicSAR::new(af_start, af_increment, af_max)?;
                let result = indicator.calculate_hl(ohlcv.high, ohlcv.low)?;
                IndicatorBatchOutput::single(result.to_vec())
            }
            IndicatorRequest::PivotPoints => {
                let indicator = PivotPoints::new();
                let output = indicator.calculate_hlc(ohlcv.high, ohlcv.low, ohlcv.close)?;
                IndicatorBatchOutput::multiple(
                    output.primary.to_vec(),
                    output.secondary.iter().map(|a| a.to_vec()).collect(),
                    vec![
                        "pp".to_string(),
                        "r1".to_string(),
                        "r2".to_string(),
                        "r3".to_string(),
                        "s1".to_string(),
                        "s2".to_string(),
                        "s3".to_string(),
                    ],
                )
            }
        };

        results.insert(name, output);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_batch_single_indicator() {
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0,
        ]);
        let open = arr1(&[
            107.0, 111.0, 116.0, 114.0, 119.0, 122.0, 120.0, 123.0, 127.0, 125.0, 129.0, 132.0,
            130.0, 133.0, 137.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0,
        ]);
        let volume = arr1(&[
            100.0, 150.0, 200.0, 120.0, 180.0, 220.0, 130.0, 190.0, 250.0, 140.0, 200.0, 260.0,
            150.0, 210.0, 270.0,
        ]);

        let ohlcv = OHLCVBatch {
            high: high.view(),
            low: low.view(),
            open: open.view(),
            close: close.view(),
            volume: volume.view(),
        };

        let requests = vec![("rsi".to_string(), IndicatorRequest::RSI { period: 14 })];

        let results = calculate_batch(&ohlcv, requests).unwrap();

        assert!(results.contains_key("rsi"));
        match &results["rsi"] {
            IndicatorBatchOutput::Single(data) => {
                assert_eq!(data.len(), 15);
            }
            _ => panic!("Expected single output"),
        }
    }

    #[test]
    fn test_batch_multiple_indicators() {
        let high = arr1(&[
            110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 123.0, 126.0, 130.0, 128.0, 132.0, 135.0,
            133.0, 136.0, 140.0, 138.0, 142.0, 145.0, 143.0, 146.0, 150.0, 148.0, 152.0, 155.0,
            153.0, 156.0, 160.0, 158.0, 162.0, 165.0,
        ]);
        let low = arr1(&[
            105.0, 110.0, 115.0, 113.0, 117.0, 120.0, 118.0, 121.0, 125.0, 123.0, 127.0, 130.0,
            128.0, 131.0, 135.0, 133.0, 137.0, 140.0, 138.0, 141.0, 145.0, 143.0, 147.0, 150.0,
            148.0, 151.0, 155.0, 153.0, 157.0, 160.0,
        ]);
        let open = arr1(&[
            107.0, 111.0, 116.0, 114.0, 119.0, 122.0, 120.0, 123.0, 127.0, 125.0, 129.0, 132.0,
            130.0, 133.0, 137.0, 135.0, 139.0, 142.0, 140.0, 143.0, 147.0, 145.0, 149.0, 152.0,
            150.0, 153.0, 157.0, 155.0, 159.0, 162.0,
        ]);
        let close = arr1(&[
            108.0, 112.0, 118.0, 115.0, 120.0, 123.0, 121.0, 124.0, 128.0, 126.0, 130.0, 133.0,
            131.0, 134.0, 138.0, 136.0, 140.0, 143.0, 141.0, 144.0, 148.0, 146.0, 150.0, 153.0,
            151.0, 154.0, 158.0, 156.0, 160.0, 163.0,
        ]);
        let volume = arr1(&[
            100.0, 150.0, 200.0, 120.0, 180.0, 220.0, 130.0, 190.0, 250.0, 140.0, 200.0, 260.0,
            150.0, 210.0, 270.0, 160.0, 220.0, 280.0, 170.0, 230.0, 290.0, 180.0, 240.0, 300.0,
            190.0, 250.0, 310.0, 200.0, 260.0, 320.0,
        ]);

        let ohlcv = OHLCVBatch {
            high: high.view(),
            low: low.view(),
            open: open.view(),
            close: close.view(),
            volume: volume.view(),
        };

        let requests = vec![
            ("rsi".to_string(), IndicatorRequest::RSI { period: 14 }),
            ("sma".to_string(), IndicatorRequest::SMA { period: 20 }),
            ("atr".to_string(), IndicatorRequest::ATR { period: 14 }),
            (
                "macd".to_string(),
                IndicatorRequest::MACD {
                    fast_period: 12,
                    slow_period: 26,
                    signal_period: 9,
                },
            ),
        ];

        let results = calculate_batch(&ohlcv, requests).unwrap();

        assert_eq!(results.len(), 4);
        assert!(results.contains_key("rsi"));
        assert!(results.contains_key("sma"));
        assert!(results.contains_key("atr"));
        assert!(results.contains_key("macd"));

        // MACD should have multiple outputs
        match &results["macd"] {
            IndicatorBatchOutput::Multiple {
                primary,
                secondary,
                names,
            } => {
                assert_eq!(primary.len(), 30);
                assert_eq!(secondary.len(), 2); // signal, histogram
                assert_eq!(names.len(), 3); // line, signal, histogram
            }
            _ => panic!("Expected multiple output for MACD"),
        }
    }
}
