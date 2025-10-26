#!/bin/bash
# Run CUDA A/B Testing Suite
#
# This script orchestrates comprehensive CUDA optimization validation:
# - Statistical analysis with n >= 100 iterations
# - Criterion benchmarks with HTML reports
# - Performance regression detection
# - Results summary generation
#
# Usage:
#   ./scripts/run_cuda_ab_test.sh [options]
#
# Options:
#   --quick       Run quick smoke test (n=10, single size)
#   --full        Run full suite (all phases, all sizes)
#   --phase1      Test Phase 1 only (compute_89)
#   --phase2      Test Phase 2 only (L2 + fusion)
#   --phase3      Test Phase 3 only (2D/3D kernels)
#   --indicator   Test specific indicator (rsi|atr|stochastic|sma|macd|bollinger)
#   --baseline    Override baseline architecture (default: compute_80)
#   --help        Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
MODE="full"
INDICATOR=""
BASELINE_ARCH="compute_80"
OPTIMIZED_ARCH="compute_89"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --phase1)
            MODE="phase1"
            shift
            ;;
        --phase2)
            MODE="phase2"
            shift
            ;;
        --phase3)
            MODE="phase3"
            shift
            ;;
        --indicator)
            INDICATOR="$2"
            shift 2
            ;;
        --baseline)
            BASELINE_ARCH="$2"
            shift 2
            ;;
        --help)
            grep '^#' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Verify GPU availability
echo -e "${BLUE}=== GPU Verification ===${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. GPU required for A/B testing.${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
echo ""

# Check CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}WARNING: nvcc not found. CUDA compilation may fail.${NC}"
else
    echo "CUDA Toolkit: $(nvcc --version | grep release | awk '{print $5}')"
fi
echo ""

# Create results directory
mkdir -p ../docs
mkdir -p target/ab_test_results

# Run tests based on mode
case $MODE in
    quick)
        echo -e "${BLUE}=== Quick Smoke Test ===${NC}"
        echo "Running single indicator, single size..."
        echo ""

        cargo bench --features gpu --bench ab_test_cuda -- rsi_100 --quick
        ;;

    phase1)
        echo -e "${BLUE}=== Phase 1: compute_89 Targeting ===${NC}"
        echo "Expected speedup: +15-30%"
        echo "Baseline: $BASELINE_ARCH"
        echo "Optimized: $OPTIMIZED_ARCH"
        echo ""

        # Run statistical analysis
        echo -e "${GREEN}Running statistical analysis (n=100 iterations)...${NC}"
        KIMSFINANCE_GPU_ARCH=$BASELINE_ARCH cargo test --features gpu --release test_statistical_analysis -- --nocapture \
            | tee target/ab_test_results/phase1_statistical.log

        # Run criterion benchmarks
        echo -e "${GREEN}Running Criterion benchmarks...${NC}"
        KIMSFINANCE_GPU_ARCH=$OPTIMIZED_ARCH cargo bench --features gpu --bench ab_test_cuda

        echo -e "${GREEN}✓ Phase 1 testing complete${NC}"
        echo "Results: docs/CUDA_AB_TEST_RESULTS.md"
        ;;

    phase2)
        echo -e "${BLUE}=== Phase 2: L2 Cache + Kernel Fusion ===${NC}"
        echo -e "${YELLOW}Status: Not yet implemented${NC}"
        echo "Expected cumulative speedup: +20-40%"
        echo ""
        echo "Implementation plan:"
        echo "  1. Add L2 cache persistence hints"
        echo "  2. Fuse kernels to reduce memory transfers"
        echo "  3. Optimize shared memory usage"
        echo ""
        echo "Run this script again after Phase 2 implementation."
        ;;

    phase3)
        echo -e "${BLUE}=== Phase 3: 2D/3D Kernels ===${NC}"
        echo -e "${YELLOW}Status: Not yet implemented${NC}"
        echo "Expected cumulative speedup: +30-50%"
        echo ""
        echo "Implementation plan:"
        echo "  1. Refactor to 2D thread blocks"
        echo "  2. Implement 3D grid batching"
        echo "  3. Optimize memory coalescing"
        echo ""
        echo "Run this script again after Phase 3 implementation."
        ;;

    full)
        echo -e "${BLUE}=== Full A/B Test Suite ===${NC}"
        echo "Testing all phases, all indicators, all dataset sizes"
        echo "This will take ~30-60 minutes..."
        echo ""

        # Phase 1
        echo -e "${GREEN}Phase 1: compute_89 targeting${NC}"
        KIMSFINANCE_GPU_ARCH=$BASELINE_ARCH cargo test --features gpu --release test_statistical_analysis -- --nocapture \
            | tee target/ab_test_results/full_statistical.log

        cargo bench --features gpu --bench ab_test_cuda

        # Phase 2 and 3 (when implemented)
        echo -e "${YELLOW}Phase 2 and 3: Pending implementation${NC}"

        echo -e "${GREEN}✓ Full suite complete${NC}"
        echo "Results: docs/CUDA_AB_TEST_RESULTS.md"
        ;;
esac

# Generate summary
echo ""
echo -e "${BLUE}=== Results Summary ===${NC}"

if [ -f ../docs/CUDA_AB_TEST_RESULTS.md ]; then
    echo -e "${GREEN}✓ Results generated successfully${NC}"
    echo ""
    echo "View full report:"
    echo "  cat ../docs/CUDA_AB_TEST_RESULTS.md"
    echo ""
    echo "View Criterion HTML reports:"
    echo "  open target/criterion/index.html"
    echo ""

    # Check for regressions
    if grep -q "⚠" ../docs/CUDA_AB_TEST_RESULTS.md; then
        echo -e "${YELLOW}⚠ WARNING: Some optimizations below expected performance${NC}"
    fi

    if grep -q "✗" ../docs/CUDA_AB_TEST_RESULTS.md; then
        echo -e "${RED}❌ ERROR: Some optimizations not statistically significant${NC}"
        exit 1
    fi

    # Extract key metrics
    echo "Key Metrics:"
    echo "-----------"
    grep -A 5 "## Executive Summary" ../docs/CUDA_AB_TEST_RESULTS.md | tail -n +2
else
    echo -e "${RED}❌ ERROR: Results file not generated${NC}"
    echo "Check logs in target/ab_test_results/"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ A/B testing complete${NC}"
