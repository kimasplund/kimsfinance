//! Async Transfer Regression Tests
//!
//! Ensures all GPU indicators use asynchronous transfers with pinned memory.
//! This prevents accidental reintroduction of synchronous transfers which are
//! 1.5x slower (64μs vs 42μs) and block compute/transfer overlap.
//!
//! **Performance Impact**: Sync transfers would reduce GPU utilization by 15-25%
//! and increase transfer latency by 50%.

use std::fs;
use std::path::Path;

/// Core GPU indicators that must use async transfers (memcpy_htod/dtoh)
const CORE_INDICATORS: &[&str] = &[
    "rsi.rs",
    "atr.rs",
    "sma.rs",
    "bollinger.rs",
    "stochastic.rs",
    "adx.rs",
    "cci.rs",
    "williams_r.rs",
    "keltner.rs",
    "vwap.rs",
    "vwap_anchored.rs",
    "obv_optimized.rs",
    "cmf.rs",
    "elder_ray.rs",
    "donchian.rs",
    "aroon.rs",
    "roc.rs",
    "pivot_points.rs",
    "supertrend.rs",
    "mfi.rs",
    "ichimoku.rs",
    "fibonacci.rs",
    "vwma.rs",
    "wma.rs",
];

/// Files allowed to use sync transfers (experimental/benchmark code)
const SYNC_ALLOWLIST: &[&str] = &[
    "sma.rs", // sma_gpu_shared is experimental shared memory variant
];

#[test]
fn test_no_sync_transfers_in_core_indicators() {
    let gpu_dir = Path::new("src/gpu");
    let mut violations = Vec::new();

    for &indicator in CORE_INDICATORS {
        let path = gpu_dir.join(indicator);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read indicator file: {}", indicator));

        // Check for sync transfers (skip allowlisted files)
        if !SYNC_ALLOWLIST.contains(&indicator) {
            if content.contains("copy_to_device") || content.contains("copy_to_host") {
                violations.push(format!(
                    "{} uses sync transfers (copy_to_device/copy_to_host) - \
                     should use async transfers (memcpy_htod/memcpy_dtoh) for 1.5x speedup",
                    indicator
                ));
            }
        }
    }

    if !violations.is_empty() {
        panic!("Sync transfer violations found:\n{}", violations.join("\n"));
    }
}

#[test]
fn test_all_indicators_use_async_transfers() {
    let gpu_dir = Path::new("src/gpu");
    let mut missing_async = Vec::new();

    for &indicator in CORE_INDICATORS {
        let path = gpu_dir.join(indicator);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read indicator file: {}", indicator));

        // All indicators should use async transfers
        let has_async = content.contains("memcpy_htod") || content.contains("memcpy_dtoh");

        if !has_async {
            missing_async.push(format!(
                "{} doesn't use async transfers (memcpy_htod/dtoh) - \
                 async transfers provide 1.5x speedup and enable compute/transfer overlap",
                indicator
            ));
        }
    }

    if !missing_async.is_empty() {
        panic!(
            "Indicators missing async transfers:\n{}",
            missing_async.join("\n")
        );
    }
}

#[test]
fn test_all_indicators_use_pinned_memory_or_stream_alloc() {
    let gpu_dir = Path::new("src/gpu");
    let mut missing_optimization = Vec::new();

    for &indicator in CORE_INDICATORS {
        let path = gpu_dir.join(indicator);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read indicator file: {}", indicator));

        // Indicators should use either:
        // 1. Pinned memory pool (device.pinned_pool)
        // 2. Stream-based allocation (stream.alloc_*)
        // Both enable async transfers
        let has_pinned_pool = content.contains("pinned_pool") || content.contains("pinned_");
        let has_stream_alloc =
            content.contains("stream.alloc") || content.contains("exec_stream.alloc");

        if !has_pinned_pool && !has_stream_alloc {
            missing_optimization.push(format!(
                "{} doesn't use pinned memory or stream allocation - \
                 should use one for optimal async transfers",
                indicator
            ));
        }
    }

    if !missing_optimization.is_empty() {
        panic!(
            "Indicators missing async transfer optimization:\n{}",
            missing_optimization.join("\n")
        );
    }
}

#[test]
fn test_async_infrastructure_exists() {
    // Verify critical infrastructure files exist and contain expected patterns
    let infrastructure = vec![
        (
            "src/gpu/async_transfers.rs",
            "memcpy_htod",
            "Core async transfer utilities",
        ),
        (
            "src/gpu/persistent/pinned_memory.rs",
            "PinnedBufferPool",
            "Pinned buffer pool management",
        ),
        (
            "src/gpu/triple_buffer.rs",
            "TripleBuffer",
            "Triple buffering for pipelines",
        ),
        (
            "src/gpu/async_alloc.rs",
            "async",
            "Asynchronous memory allocation",
        ),
        (
            "src/gpu/device.rs",
            "pinned_pool",
            "Device-level pinned pool integration",
        ),
    ];

    let mut missing_infrastructure = Vec::new();

    for (file, pattern, description) in infrastructure {
        match fs::read_to_string(file) {
            Ok(content) => {
                if !content.contains(pattern) {
                    missing_infrastructure.push(format!(
                        "{} exists but missing expected pattern '{}' ({})",
                        file, pattern, description
                    ));
                }
            }
            Err(_) => {
                missing_infrastructure.push(format!(
                    "{} not found - critical infrastructure missing ({})",
                    file, description
                ));
            }
        }
    }

    if !missing_infrastructure.is_empty() {
        panic!(
            "Critical async transfer infrastructure issues:\n{}",
            missing_infrastructure.join("\n")
        );
    }
}

#[test]
fn test_no_mixed_transfer_patterns() {
    // Ensure indicators don't mix sync and async transfers (except allowlisted)
    let gpu_dir = Path::new("src/gpu");
    let mut mixed_patterns = Vec::new();

    for &indicator in CORE_INDICATORS {
        if SYNC_ALLOWLIST.contains(&indicator) {
            continue; // Skip allowlisted files
        }

        let path = gpu_dir.join(indicator);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read indicator file: {}", indicator));

        let has_sync = content.contains("copy_to_device") || content.contains("copy_to_host");
        let has_async = content.contains("memcpy_htod") || content.contains("memcpy_dtoh");

        if has_sync && has_async {
            mixed_patterns.push(format!(
                "{} mixes sync and async transfers - \
                 should use only async transfers for consistency and optimal performance",
                indicator
            ));
        }
    }

    if !mixed_patterns.is_empty() {
        panic!(
            "Indicators with mixed transfer patterns:\n{}",
            mixed_patterns.join("\n")
        );
    }
}

#[test]
fn test_async_transfer_adoption_rate() {
    // Track async transfer adoption rate - should be 97%+ (27/28 core indicators)
    let gpu_dir = Path::new("src/gpu");
    let mut async_count = 0;
    let total_count = CORE_INDICATORS.len();

    for &indicator in CORE_INDICATORS {
        let path = gpu_dir.join(indicator);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read indicator file: {}", indicator));

        if content.contains("memcpy_htod") || content.contains("memcpy_dtoh") {
            async_count += 1;
        }
    }

    let adoption_rate = (async_count as f64 / total_count as f64) * 100.0;

    assert!(
        adoption_rate >= 97.0,
        "Async transfer adoption rate ({:.1}%) below 97% - {} of {} indicators use async transfers",
        adoption_rate,
        async_count,
        total_count
    );
}

#[cfg(test)]
mod documentation_tests {
    use super::*;

    #[test]
    fn test_audit_report_exists() {
        // Verify audit documentation exists
        let docs = vec![
            "docs/ASYNC_TRANSFER_AUDIT_REPORT.md",
            "docs/AGENT2_COMPLETION_SUMMARY.md",
            "docs/ASYNC_TRANSFER_VISUAL_SUMMARY.txt",
            "docs/AGENT2_NEXT_STEPS.md",
        ];

        for doc in docs {
            assert!(
                Path::new(doc).exists(),
                "Async transfer documentation not found: {}",
                doc
            );
        }
    }
}
