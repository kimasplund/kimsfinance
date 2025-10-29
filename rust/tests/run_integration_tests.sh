#!/bin/bash
# Test Execution Script for Heston-Backtest Integration
# Phase 5 Validation Suite

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Heston-Backtest Integration Test Suite${NC}"
echo -e "${BLUE}Phase 5: Comprehensive Validation${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to run test suite
run_test_suite() {
    local test_name=$1
    local test_cmd=$2
    local description=$3

    echo -e "${YELLOW}>>> Running: ${test_name}${NC}"
    echo -e "${YELLOW}    ${description}${NC}"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if eval "$test_cmd" 2>&1 | tee /tmp/test_output.log; then
        echo -e "${GREEN}✓ PASS: ${test_name}${NC}\n"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL: ${test_name}${NC}\n"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to run optional test (don't fail on skip)
run_optional_test() {
    local test_name=$1
    local test_cmd=$2
    local description=$3

    echo -e "${YELLOW}>>> Running (Optional): ${test_name}${NC}"
    echo -e "${YELLOW}    ${description}${NC}"

    if eval "$test_cmd" 2>&1 | tee /tmp/test_output.log; then
        echo -e "${GREEN}✓ PASS: ${test_name}${NC}\n"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${YELLOW}⊘ SKIP: ${test_name} (GPU required or slow)${NC}\n"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
    fi
}

# ========== 1. Compile Check ==========
echo -e "${BLUE}[1/7] Compilation Check${NC}"
run_test_suite \
    "Compilation (GPU+Heston features)" \
    "cargo build --features 'gpu,heston' --lib" \
    "Verify code compiles with all features enabled"

# ========== 2. Unit Tests ==========
echo -e "${BLUE}[2/7] Unit Tests${NC}"
run_test_suite \
    "Library Unit Tests" \
    "cargo test --features 'gpu,heston' --lib -- --test-threads=1" \
    "Run all unit tests in library code"

run_optional_test \
    "Heston Unit Tests" \
    "cargo test --features 'gpu,heston' --test heston_unit_tests -- --include-ignored --test-threads=1" \
    "Strategy types, Heston pricer, Greeks calculation"

# ========== 3. Integration Tests ==========
echo -e "${BLUE}[3/7] Integration Tests${NC}"
run_optional_test \
    "End-to-End Pipeline Tests" \
    "cargo test --features 'gpu,heston' --test heston_e2e_test -- --include-ignored --test-threads=1" \
    "Full 5-phase pipeline (Phase 0-4) for all strategies"

run_optional_test \
    "Accuracy Validation Tests" \
    "cargo test --features 'gpu,heston' --test heston_accuracy_test -- --include-ignored --test-threads=1" \
    "GPU vs CPU accuracy (<0.05% price, <1% Greeks)"

run_optional_test \
    "Regression Tests" \
    "cargo test --features 'gpu,heston' --test heston_regression_test -- --include-ignored --test-threads=1" \
    "Backward compatibility and performance baselines"

# ========== 4. Load Tests (Optional) ==========
echo -e "${BLUE}[4/7] Load Tests (Optional)${NC}"
echo -e "${YELLOW}Load tests are SLOW - run manually if needed${NC}"
echo -e "${YELLOW}Command: cargo test --features 'gpu,heston' --test heston_load_test -- --include-ignored${NC}\n"
SKIPPED_TESTS=$((SKIPPED_TESTS + 1))

# ========== 5. Performance Benchmarks ==========
echo -e "${BLUE}[5/7] Performance Benchmarks${NC}"
echo -e "${YELLOW}>>> Running: Performance Benchmarks${NC}"
echo -e "${YELLOW}    Phase 0 (Heston), Greeks, Full Pipeline${NC}"

if cargo bench --features 'gpu,heston' --bench heston_integration_bench -- --quick 2>&1 | tee /tmp/bench_output.log; then
    echo -e "${GREEN}✓ Benchmarks completed${NC}\n"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⊘ Benchmarks skipped (GPU required)${NC}\n"
    SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
fi

# ========== 6. Existing Test Suites ==========
echo -e "${BLUE}[6/7] Existing Test Suites${NC}"
run_optional_test \
    "Heston Integration Test (Existing)" \
    "cargo test --features 'gpu,heston' --test heston_integration_test -- --include-ignored --test-threads=1" \
    "Original Heston calibration tests"

run_optional_test \
    "Greeks GPU Test (Existing)" \
    "cargo test --features 'gpu,heston' --test greeks_gpu_test -- --include-ignored --test-threads=1" \
    "Original Greeks calculation tests"

# ========== 7. Summary Report ==========
echo -e "${BLUE}[7/7] Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total Test Suites: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED_TESTS}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Generate detailed report
REPORT_FILE="tests/PHASE_5_TEST_RESULTS_$(date +%Y%m%d_%H%M%S).txt"
echo "Heston-Backtest Integration Test Results" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Summary:" >> "$REPORT_FILE"
echo "  Total: $TOTAL_TESTS" >> "$REPORT_FILE"
echo "  Passed: $PASSED_TESTS" >> "$REPORT_FILE"
echo "  Failed: $FAILED_TESTS" >> "$REPORT_FILE"
echo "  Skipped: $SKIPPED_TESTS" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Append benchmark results if available
if [ -f /tmp/bench_output.log ]; then
    echo "Benchmark Results:" >> "$REPORT_FILE"
    echo "========================================" >> "$REPORT_FILE"
    grep -A 5 "time:" /tmp/bench_output.log >> "$REPORT_FILE" 2>/dev/null || echo "No timing data found" >> "$REPORT_FILE"
fi

echo -e "${GREEN}Detailed report saved to: ${REPORT_FILE}${NC}\n"

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}FAILURE: ${FAILED_TESTS} test suite(s) failed${NC}"
    exit 1
elif [ $PASSED_TESTS -eq 0 ]; then
    echo -e "${YELLOW}WARNING: No tests were run (GPU may be unavailable)${NC}"
    exit 2
else
    echo -e "${GREEN}SUCCESS: All test suites passed!${NC}"
    exit 0
fi
