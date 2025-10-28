#!/bin/bash
#
# Benchmark Infrastructure Verification Script
#
# Purpose: Verify all benchmark files are in place and compile correctly
#
# Usage:
#   bash scripts/verify_benchmark_infrastructure.sh
#
# Exit codes:
#   0 = All checks passed
#   1 = Some checks failed

set -e

echo "=================================================="
echo "GPU Batch Backtesting Benchmark Infrastructure"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Helper function
check_file() {
    local file=$1
    local description=$2

    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description: $file"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description: $file (NOT FOUND)"
        ((FAILED++))
        return 1
    fi
}

echo "Phase 1: Checking Files"
echo "------------------------"

# Benchmark files
check_file "rust/benches/batch_backtest_benchmark.rs" "Main benchmark suite"
check_file "rust/benches/test_data_generator.rs" "Test data generator"
check_file "scripts/validate_batch_accuracy.py" "Accuracy validation script"
check_file "benchmarks/BATCH_BACKTEST_RESULTS.md" "Performance report template"
check_file "benchmarks/BATCH_BACKTEST_README.md" "Benchmark README"

echo ""
echo "Phase 2: Checking Cargo.toml"
echo "-----------------------------"

if grep -q "batch_backtest_benchmark" rust/Cargo.toml; then
    echo -e "${GREEN}✓${NC} Benchmark registered in Cargo.toml"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Benchmark NOT registered in Cargo.toml"
    ((FAILED++))
fi

echo ""
echo "Phase 3: Checking Compilation"
echo "------------------------------"

echo -n "Compiling benchmark (placeholders OK)... "
if cargo bench --bench batch_backtest_benchmark --no-run --features gpu 2>&1 | grep -q "Finished"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "  Run manually: cargo bench --bench batch_backtest_benchmark --no-run --features gpu"
    ((FAILED++))
fi

echo ""
echo "Phase 4: Checking Python Dependencies"
echo "--------------------------------------"

# Check Python dependencies
for pkg in numpy scipy pandas matplotlib; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python package: $pkg"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠${NC} Python package: $pkg (MISSING - install with: pip install $pkg)"
        ((FAILED++))
    fi
done

echo ""
echo "Phase 5: Checking GPU Availability"
echo "-----------------------------------"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} nvidia-smi available"
    ((PASSED++))

    # Get GPU info
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    echo "  GPU: $GPU_NAME"
    echo "  Memory: $GPU_MEM"
    echo "  Driver: $CUDA_VERSION"

    # Check if it's RTX 3500 Ada
    if echo "$GPU_NAME" | grep -q "3500"; then
        echo -e "${GREEN}✓${NC} Target GPU (RTX 3500 Ada) detected"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠${NC} Different GPU detected (benchmarks calibrated for RTX 3500 Ada)"
        ((PASSED++))  # Still pass, just different hardware
    fi
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not available (GPU benchmarks may fail)"
    echo "  This is OK if running on CPU-only machine"
    ((PASSED++))
fi

echo ""
echo "=================================================="
echo "Summary"
echo "=================================================="
echo ""
echo -e "Tests Passed: ${GREEN}$PASSED${NC}"
echo -e "Tests Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Wait for CUDA kernels (Task 1) to complete"
    echo "  2. Uncomment GPU/CPU implementations in batch_backtest_benchmark.rs"
    echo "  3. Run benchmarks: cargo bench --bench batch_backtest_benchmark"
    echo "  4. Validate accuracy: python scripts/validate_batch_accuracy.py"
    echo "  5. Fill in results: benchmarks/BATCH_BACKTEST_RESULTS.md"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed!${NC}"
    echo ""
    echo "Please fix the issues above before proceeding."
    echo ""
    exit 1
fi
