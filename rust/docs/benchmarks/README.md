# Benchmark & Validation Quick Reference

**Agent 6 Framework** - Statistical validation for GPU optimizations

---

## Quick Commands

### Run Full Validation

```bash
# Automated workflow (recommended)
./scripts/run_comprehensive_validation.sh

# Output: docs/benchmarks/validation_report_<timestamp>.md
```

### Individual Benchmarks

```bash
# Comprehensive GPU validation (n=100, statistical analysis)
cargo bench --bench comprehensive_gpu_validation

# Performance regression test (vs baselines.json)
cargo bench --bench performance_regression

# All indicators timing (quick check)
cargo bench --bench all_indicators_timing

# Accuracy validation (GPU vs CPU)
cargo test --test gpu_accuracy_validation -- --ignored --nocapture
```

### Analyze Results

```bash
# Parse Criterion output and compute statistics
python3 scripts/analyze_benchmark_results.py

# Manual review
ls -lh target/criterion/gpu_validation/
```

---

## Expected Output

### Successful Validation

```
========================================
  GPU Indicator Validation Suite
  Agent 6: Benchmark & Validation
========================================

[INFO] Validating environment...
[SUCCESS] GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU
[INFO] Memory: 12285MB
[INFO] CUDA Driver: 580.82.07
[INFO] Compute Capability: 8.9
[INFO] Rust: rustc 1.90.0

[INFO] Running baseline benchmark...
[SUCCESS] Baseline benchmark complete

[INFO] Running comprehensive validation...
[SUCCESS] Validation benchmark complete

[INFO] Generating report...
[SUCCESS] Report generated: docs/benchmarks/validation_report_20251101_143022.md

✅ ALL TESTS PASSED
```

### Performance Regression Detected

```
❌ PERFORMANCE REGRESSION DETECTED

Failed Tests:
  - rsi: 15.3% slower than baseline (2512 μs -> 2896 μs)

Exit code: 1
```

---

## Interpreting Results

### Speedup Claims

| Speedup | Confidence | Interpretation |
|---------|-----------|----------------|
| > 2.0x | p < 0.01 | **Strong evidence** of improvement |
| 1.5-2.0x | p < 0.05 | **Significant** improvement |
| 1.2-1.5x | p < 0.10 | **Moderate** improvement |
| < 1.2x | p >= 0.10 | **Weak** or no improvement |

### Effect Size (Cohen's d)

| Cohen's d | Effect | Interpretation |
|-----------|--------|----------------|
| > 0.8 | Large | **Major** performance improvement |
| 0.5-0.8 | Medium | **Meaningful** improvement |
| 0.2-0.5 | Small | **Noticeable** improvement |
| < 0.2 | Negligible | **Minimal** improvement |

### Bandwidth Utilization

| Utilization | Status | Action |
|-------------|--------|--------|
| > 75% | Memory-bound | ✅ Good - focus on reducing traffic |
| 50-75% | Well-optimized | ✅ Near optimal |
| 30-50% | Moderate | ⚠️ Try async transfers or pinned memory |
| < 30% | Compute-bound | ⚠️ Consider kernel fusion |

---

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check Rust GPU feature
cargo check --features gpu
```

### Benchmark Fails to Compile

```bash
# Clean build
cargo clean

# Rebuild with full output
cargo bench --bench comprehensive_gpu_validation --features gpu --verbose
```

### Statistical Tests Show No Significance (p >= 0.05)

**Possible causes**:
1. High variance - increase sample size: `--sample-size 200`
2. Small effect size - optimization may not be meaningful
3. System noise - close other GPU processes

**Actions**:
1. Increase warmup rounds
2. Run on dedicated hardware
3. Check for background GPU processes: `nvidia-smi dmon`

### Accuracy Validation Fails

```bash
# Run with detailed output
cargo test --test gpu_accuracy_validation -- --ignored --nocapture

# Check specific indicator
cargo test --test gpu_accuracy_validation -- --ignored test_rsi_accuracy --nocapture
```

**Common issues**:
- Numerical instability (use Kahan summation)
- NaN propagation differences
- Edge case handling

---

## File Structure

```
docs/benchmarks/
├── README.md                          # This file
├── validation_report_<timestamp>.md   # Generated reports
├── baseline_output.txt                # Raw baseline output
└── validation_output.txt              # Raw validation output

target/criterion/
├── gpu_validation/                    # Comprehensive validation
│   ├── ema_1000/                     # Per-indicator, per-size
│   │   ├── base/estimates.json       # Statistical estimates
│   │   └── report/index.html         # HTML visualization
│   └── ...
└── performance_regression/            # Regression tests
    └── ...
```

---

## Baselines (RTX 3500 Ada, 100K candles)

| Indicator | Baseline (μs) | Tolerance | Category |
|-----------|---------------|-----------|----------|
| **Simple** | | | |
| EMA | 200 | ±10% | CPU fallback |
| ROC | 442 | ±10% | GPU |
| SMA | 519 | ±10% | GPU |
| **Medium** | | | |
| Williams %R | 1,079 | ±10% | GPU |
| CCI | 1,152 | ±10% | GPU |
| Stochastic | 1,279 | ±10% | GPU |
| **Complex** | | | |
| ATR | 1,360 | ±10% | GPU (async optimized) |
| RSI | 2,512 | ±10% | GPU |

---

## Configuration Files

### `benches/baselines.json`

```json
{
  "version": "1.0.0",
  "hardware": {
    "gpu": "NVIDIA RTX 3500 Ada",
    "cpu": "Intel i9-13980HX",
    "cuda": "13.0",
    "compute_capability": "8.9"
  },
  "test_config": {
    "candles": 100000,
    "warmup_runs": 5,
    "measurement_runs": 10
  },
  "baselines": {
    "simple_indicators": { ... },
    "medium_indicators": { ... },
    "complex_indicators": { ... }
  }
}
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: GPU Validation
on:
  pull_request:
    paths: ['src/gpu/**', 'benches/**']

jobs:
  validate:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      - name: Run Validation
        run: ./scripts/run_comprehensive_validation.sh
      - name: Check Regressions
        run: cargo bench --bench performance_regression
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: docs/benchmarks/validation_report_*.md
```

---

## Support

- **Documentation**: See `docs/AGENT_6_VALIDATION_FRAMEWORK.md`
- **Completion Summary**: See `docs/AGENT_6_COMPLETION_SUMMARY.md`
- **Issues**: File GitHub issue with validation report attached

---

**Last Updated**: 2025-11-01
**Framework Version**: 1.0.0
**Maintained by**: Agent 6 (Benchmark & Validation Specialist)
