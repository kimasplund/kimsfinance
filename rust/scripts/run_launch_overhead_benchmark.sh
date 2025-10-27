#!/bin/bash
#
# Launch Overhead Benchmark Runner
#
# Purpose: Validate 2-4x speedup claim for persistent kernels
# Location: /home/kim-asplund/projects/kimsfinance/rust/scripts/run_launch_overhead_benchmark.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/kim-asplund/projects/kimsfinance/rust"
BENCHMARK_NAME="launch_overhead"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Launch Overhead Benchmark Runner${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Step 1: Verify GPU availability
echo -e "${YELLOW}[1/6] Verifying GPU availability...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. GPU required for this benchmark.${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader)
echo -e "${GREEN}GPU detected:${NC}"
echo "  $GPU_INFO"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | cut -d. -f1)
if [ "$COMPUTE_CAP" -lt 7 ]; then
    echo -e "${RED}ERROR: Compute Capability $COMPUTE_CAP detected. Need >= 7.0 for cooperative launch.${NC}"
    exit 1
fi
echo -e "${GREEN}Compute Capability: $COMPUTE_CAP (sufficient for cooperative launch)${NC}"
echo ""

# Step 2: Check for GPU contention
echo -e "${YELLOW}[2/6] Checking GPU utilization...${NC}"
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if [ "$GPU_UTIL" -gt 10 ]; then
    echo -e "${YELLOW}WARNING: GPU is currently $GPU_UTIL% utilized. Results may be noisy.${NC}"
    echo -e "${YELLOW}Recommendation: Close other GPU processes for stable benchmarks.${NC}"
else
    echo -e "${GREEN}GPU utilization: $GPU_UTIL% (good for benchmarking)${NC}"
fi
echo ""

# Step 3: Build benchmark
echo -e "${YELLOW}[3/6] Building benchmark with GPU support...${NC}"
cd "$PROJECT_ROOT"
if ! cargo build --bench "$BENCHMARK_NAME" --features gpu --release; then
    echo -e "${RED}ERROR: Benchmark build failed.${NC}"
    exit 1
fi
echo -e "${GREEN}Build successful.${NC}"
echo ""

# Step 4: Run baseline (traditional approach)
echo -e "${YELLOW}[4/6] Running baseline (traditional multi-launch)...${NC}"
echo -e "${BLUE}This measures N separate kernel launches (current approach)${NC}"
echo ""

cargo bench --bench "$BENCHMARK_NAME" --features gpu -- traditional --save-baseline before

echo ""
echo -e "${GREEN}Baseline complete.${NC}"
echo ""

# Step 5: Run persistent kernel benchmark
echo -e "${YELLOW}[5/6] Running persistent kernel benchmark...${NC}"
echo -e "${BLUE}This measures single kernel launch for N tasks (new approach)${NC}"
echo ""

cargo bench --bench "$BENCHMARK_NAME" --features gpu -- persistent

echo ""
echo -e "${GREEN}Persistent kernel benchmark complete.${NC}"
echo ""

# Step 6: Run comparison benchmarks
echo -e "${YELLOW}[6/6] Running comparison benchmarks...${NC}"
echo ""

# Direct comparison at 10 tasks (critical operating point)
echo -e "${BLUE}Testing overhead reduction at 10 tasks (typical backtest scenario)...${NC}"
cargo bench --bench "$BENCHMARK_NAME" --features gpu -- overhead_reduction_10_tasks

echo ""
echo -e "${BLUE}Testing dataset size scaling (1K, 10K, 100K candles)...${NC}"
cargo bench --bench "$BENCHMARK_NAME" --features gpu -- dataset_size_scaling

echo ""
echo -e "${BLUE}Testing throughput (100 tasks)...${NC}"
cargo bench --bench "$BENCHMARK_NAME" --features gpu -- throughput

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Step 7: Display results summary
REPORT_DIR="$PROJECT_ROOT/target/criterion"
if [ -d "$REPORT_DIR" ]; then
    echo -e "${BLUE}Results location:${NC}"
    echo "  HTML reports: $REPORT_DIR/"
    echo "  Open in browser: file://$REPORT_DIR/report/index.html"
    echo ""
fi

# Step 8: Extract key metrics (if available)
echo -e "${YELLOW}Key Metrics Summary:${NC}"
echo ""

# Function to extract benchmark result
extract_result() {
    local bench_name=$1
    local result_file="$REPORT_DIR/$bench_name/new/estimates.json"

    if [ -f "$result_file" ]; then
        # Use jq if available, otherwise use grep/sed
        if command -v jq &> /dev/null; then
            local mean=$(jq -r '.mean.point_estimate' "$result_file")
            local std=$(jq -r '.std_dev.point_estimate' "$result_file")
            echo "  Mean: ${mean}ns, Std: ${std}ns"
        else
            echo "  (Install jq for detailed metrics)"
        fi
    else
        echo "  (Run benchmark first)"
    fi
}

echo -e "${BLUE}Traditional (10 tasks):${NC}"
extract_result "overhead_traditional_10"

echo -e "${BLUE}Persistent (10 tasks):${NC}"
extract_result "overhead_persistent_10"

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review HTML reports in: $REPORT_DIR/report/index.html"
echo "  2. Check for statistical significance (p < 0.05)"
echo "  3. Verify speedup >= 2.0x (target: 2-4x)"
echo "  4. Compare against baseline: cargo bench --bench $BENCHMARK_NAME --features gpu -- --baseline before"
echo ""

# Step 9: Validation checklist
echo -e "${YELLOW}Success Criteria (2-4x speedup validation):${NC}"
echo "  [ ] Launch overhead reduced by ≥80% (10 tasks)"
echo "  [ ] Throughput improved by ≥2x (100 tasks)"
echo "  [ ] Speedup ≥2.0x for small datasets (1K-10K)"
echo "  [ ] Statistical significance: p < 0.05"
echo "  [ ] Confidence intervals: ≤±10% of mean"
echo "  [ ] Coefficient of variation: <10%"
echo ""

echo -e "${GREEN}For detailed analysis, see:${NC}"
echo "  $PROJECT_ROOT/benches/LAUNCH_OVERHEAD_BENCHMARK.md"
echo ""

exit 0
