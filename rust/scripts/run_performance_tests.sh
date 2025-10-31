#!/bin/bash
# Performance Regression Test Runner
#
# Runs the full performance regression test suite and generates a report.
#
# Usage:
#   ./scripts/run_performance_tests.sh              # Run all tests
#   ./scripts/run_performance_tests.sh --verbose    # Verbose output
#   ./scripts/run_performance_tests.sh --save       # Save report to file

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$(dirname "$SCRIPT_DIR")"
REPORT_FILE="$RUST_DIR/performance_report_$(date +%Y%m%d_%H%M%S).txt"

# Parse arguments
VERBOSE=false
SAVE_REPORT=false

for arg in "$@"; do
    case $arg in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --save)
            SAVE_REPORT=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --verbose    Enable verbose output"
            echo "  --save       Save report to file"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}                   ${GREEN}PERFORMANCE REGRESSION TEST SUITE${NC}                     ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check GPU availability
echo -e "${BLUE}[1/4]${NC} Checking GPU availability..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo -e "${RED}ERROR: nvidia-smi not found. GPU required for performance tests.${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo -e "${GREEN}✓${NC} GPU detected: $GPU_NAME"
echo ""

# Check baselines file
echo -e "${BLUE}[2/4]${NC} Checking baselines configuration..."
if [ ! -f "$RUST_DIR/benches/baselines.json" ]; then
    echo -e "${RED}ERROR: baselines.json not found in benches/directory${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Baselines configuration found"
echo ""

# Build release binary
echo -e "${BLUE}[3/4]${NC} Building release binary with GPU features..."
cd "$RUST_DIR"

if [ "$VERBOSE" = true ]; then
    cargo build --release --features gpu --benches
else
    cargo build --release --features gpu --benches > /dev/null 2>&1
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Build complete"
echo ""

# Run performance tests
echo -e "${BLUE}[4/4]${NC} Running performance regression tests..."
echo ""

if [ "$SAVE_REPORT" = true ]; then
    cargo run --release --features gpu --bench performance_regression 2>&1 | tee "$REPORT_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
else
    cargo run --release --features gpu --bench performance_regression
    EXIT_CODE=$?
fi

echo ""

# Report results
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}                          ${GREEN}✓ ALL TESTS PASSED${NC}                              ${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"

    if [ "$SAVE_REPORT" = true ]; then
        echo ""
        echo -e "${BLUE}Report saved to:${NC} $REPORT_FILE"
    fi

    exit 0
elif [ $EXIT_CODE -eq 1 ]; then
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}                     ${RED}✗ PERFORMANCE REGRESSION DETECTED${NC}                 ${RED}║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════╝${NC}"

    if [ "$SAVE_REPORT" = true ]; then
        echo ""
        echo -e "${BLUE}Report saved to:${NC} $REPORT_FILE"
    fi

    echo ""
    echo -e "${YELLOW}Recommendations:${NC}"
    echo "  1. Review the failed tests above"
    echo "  2. Check recent code changes that may have affected performance"
    echo "  3. Profile the affected indicators with GPU tools"
    echo "  4. Update baselines if regression is intentional (e.g., for correctness)"
    echo ""

    exit 1
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}                         ${RED}✗ CONFIGURATION ERROR${NC}                          ${RED}║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════╝${NC}"

    if [ "$SAVE_REPORT" = true ]; then
        echo ""
        echo -e "${BLUE}Report saved to:${NC} $REPORT_FILE"
    fi

    exit 2
fi
