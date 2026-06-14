//! Numerical precision policy for GPU compute — the configurable "accuracy limiter".
//!
//! Financial signals are coarse (thresholds / crossovers), so most of FP64's
//! range is decision-irrelevant. But precision is **not** uniform: recursive/IIR
//! and long-cumulative paths compound error and need FP64/FP32, while
//! bounded-window indicators tolerate FP32 (and often FP16). This type makes the
//! choice explicit and configurable instead of hard-coded per kernel.
//!
//! The real promotion gate is backtest trade/P&L equivalence, not an
//! indicator-value tolerance. See
//! `research/gpu-cuda-cores/precision/00-PRECISION-POLICY.md` for the full tier
//! policy and rationale.

/// Numerical class of a GPU compute path — selects a safe default precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalClass {
    /// Bounded-window, non-recursive (SMA, Bollinger middle, CCI). FP32-safe;
    /// FP16-eligible with per-window rebasing.
    BoundedWindow,
    /// Recursive / IIR (EMA, Wilder smoothing in ATR/ADX/RSI, SuperTrend, PSAR).
    /// Error compounds geometrically -> FP32 floor.
    Recursive,
    /// Long cumulative (OBV, VWAP, CVD, equity). O(N) error growth + the FP32
    /// 2^24 (~16.7M) integer cliff -> FP64 by default (or FP32 + compensated sum).
    Cumulative,
    /// Variance / std / z-scores (Bollinger std, CCI). Cancellation-prone; use
    /// the stable two-pass form -> FP32 with care.
    Variance,
}

/// Requested numerical precision for a GPU computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Force FP64 — exact reference; slowest (Ada runs FP64 at 1/64 of FP32).
    F64,
    /// Force FP32 — default for most indicators (~2x bandwidth, full ALU rate).
    F32,
    /// Resolve to the tier-policy default for the path's [`NumericalClass`].
    #[default]
    Auto,
}

impl Precision {
    /// Resolve `Auto` to a concrete precision using the tier policy for `class`.
    ///
    /// FP16/INT8 tiers are introduced per-kernel where a rebasing/quantization
    /// rewrite makes them safe; `Auto` never silently drops below the FP32 floor.
    pub fn resolve(self, class: NumericalClass) -> Precision {
        match self {
            Precision::Auto => match class {
                NumericalClass::Cumulative => Precision::F64,
                _ => Precision::F32,
            },
            other => other,
        }
    }

    /// True if the resolved precision for `class` is FP64.
    pub fn is_f64(self, class: NumericalClass) -> bool {
        matches!(self.resolve(class), Precision::F64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_picks_tier_defaults() {
        // Default is Auto.
        assert_eq!(Precision::default(), Precision::Auto);
        // Bounded/recursive/variance -> f32; cumulative -> f64.
        assert_eq!(
            Precision::Auto.resolve(NumericalClass::BoundedWindow),
            Precision::F32
        );
        assert_eq!(
            Precision::Auto.resolve(NumericalClass::Recursive),
            Precision::F32
        );
        assert_eq!(
            Precision::Auto.resolve(NumericalClass::Variance),
            Precision::F32
        );
        assert_eq!(
            Precision::Auto.resolve(NumericalClass::Cumulative),
            Precision::F64
        );
    }

    #[test]
    fn explicit_overrides_class() {
        // Explicit F64/F32 ignore the class default.
        assert_eq!(
            Precision::F64.resolve(NumericalClass::BoundedWindow),
            Precision::F64
        );
        assert_eq!(
            Precision::F32.resolve(NumericalClass::Cumulative),
            Precision::F32
        );
        assert!(Precision::Auto.is_f64(NumericalClass::Cumulative));
        assert!(!Precision::Auto.is_f64(NumericalClass::BoundedWindow));
    }
}
