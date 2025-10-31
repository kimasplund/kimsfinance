#!/bin/bash
# Generate comprehensive genetic optimizer performance report
#
# This script runs all genetic optimizer benchmarks and generates
# a comprehensive performance report comparing:
# - Mutex removal impact (1.6-2.4x speedup)
# - Population scaling efficiency
# - Convergence speed improvements
# - Data size impact
# - FP8 precision tradeoffs
#
# Usage:
#   ./rust/scripts/generate_optimizer_perf_report.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$RUST_DIR/results"

echo "==================================================================="
echo "  Genetic Optimizer Performance Report Generator"
echo "==================================================================="
echo ""
echo "Hardware: Intel i9-13980HX (24 cores) + RTX 3500 Ada"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "This will take approximately 15-20 minutes..."
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Change to rust directory
cd "$RUST_DIR"

echo "==================================================================="
echo "  Phase 1: Parallel Performance (No Mutex)"
echo "==================================================================="
echo ""
cargo bench --features gpu --bench genetic_optimizer_comparison -- parallel_no_mutex 2>&1 | tee "$RESULTS_DIR/optimizer_parallel_bench.txt"

echo ""
echo "==================================================================="
echo "  Phase 2: Population Scaling"
echo "==================================================================="
echo ""
cargo bench --features gpu --bench genetic_optimizer_comparison -- scaling 2>&1 | tee "$RESULTS_DIR/optimizer_scaling_bench.txt"

echo ""
echo "==================================================================="
echo "  Phase 3: Convergence Speed"
echo "==================================================================="
echo ""
cargo bench --features gpu --bench genetic_optimizer_comparison -- convergence 2>&1 | tee "$RESULTS_DIR/optimizer_convergence_bench.txt"

echo ""
echo "==================================================================="
echo "  Phase 4: Data Size Impact"
echo "==================================================================="
echo ""
cargo bench --features gpu --bench genetic_optimizer_comparison -- data_size 2>&1 | tee "$RESULTS_DIR/optimizer_datasize_bench.txt"

echo ""
echo "==================================================================="
echo "  Phase 5: FP8 Precision Tradeoffs"
echo "==================================================================="
echo ""
cargo bench --features gpu --bench genetic_optimizer_precision 2>&1 | tee "$RESULTS_DIR/optimizer_precision_bench.txt"

echo ""
echo "==================================================================="
echo "  Benchmark Results Summary"
echo "==================================================================="
echo ""

# Combine all results
cat "$RESULTS_DIR"/optimizer_*_bench.txt > "$RESULTS_DIR/optimizer_full_results.txt"

# Parse and display key metrics
echo "=== Parallel Performance (No Mutex) ==="
grep -A 2 "ParallelNoMutex" "$RESULTS_DIR/optimizer_parallel_bench.txt" | tail -n 5 || echo "No results found"

echo ""
echo "=== Population Scaling Efficiency ==="
grep -A 2 "Scaling/" "$RESULTS_DIR/optimizer_scaling_bench.txt" | head -n 10 || echo "No results found"

echo ""
echo "=== Convergence Speed ==="
grep -A 2 "Convergence/" "$RESULTS_DIR/optimizer_convergence_bench.txt" | head -n 10 || echo "No results found"

echo ""
echo "=== Data Size Impact ==="
grep -A 2 "DataSize/" "$RESULTS_DIR/optimizer_datasize_bench.txt" | head -n 10 || echo "No results found"

echo ""
echo "=== FP8 Precision Quality ==="
grep -E "(FP64|FP8|Hybrid)" "$RESULTS_DIR/optimizer_precision_bench.txt" | head -n 15 || echo "No results found"

echo ""
echo "==================================================================="
echo "  Report Generation Complete"
echo "==================================================================="
echo ""
echo "Full results saved to:"
echo "  - $RESULTS_DIR/optimizer_full_results.txt"
echo "  - Individual phase results in $RESULTS_DIR/optimizer_*_bench.txt"
echo ""
echo "HTML reports available at:"
echo "  $RUST_DIR/target/criterion/"
echo ""
echo "To generate final markdown report, run:"
echo "  python $RUST_DIR/scripts/generate_optimizer_markdown_report.py"
echo ""
