#!/bin/bash
#
# Comprehensive GPU Batch Backtest Optimization Benchmarks
#
# Hardware: NVIDIA RTX 3500 Ada (12GB VRAM, 14,336 CUDA cores)
# CUDA: 13.0
# Purpose: Validate 2-4x speedup from persistent kernel optimizations
#
# Performance Targets:
# - Persistent kernels: 235ms → 120ms (2x speedup)
# - Phase 3 optimization: 100ms → 70ms (1.4x speedup)
# - Combined: 235ms → 85ms (2.8x speedup)
#
# Usage:
#   ./scripts/run_optimization_benchmarks.sh [--quick|--full|--report-only]
#
# Options:
#   --quick        Run only key configurations (10min)
#   --full         Run all configurations (60min)
#   --report-only  Skip benchmarks, only generate report from existing data

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BENCHMARK_NAME="optimization_comparison"
RESULTS_DIR="target/criterion"
REPORT_FILE="benchmarks/OPTIMIZATION_RESULTS.md"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}GPU Batch Backtest Optimization Benchmarks${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Hardware: NVIDIA RTX 3500 Ada"
echo "CUDA: 13.0"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Parse command line arguments
MODE="${1:-full}"

# ===== Step 1: Environment Validation =====
echo -e "${YELLOW}[1/6] Validating environment...${NC}"

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Is NVIDIA driver installed?${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)

echo "  GPU: $GPU_NAME"
echo "  VRAM: ${GPU_MEMORY} MB"
echo "  CUDA Driver: $CUDA_VERSION"

if [[ ! "$GPU_NAME" =~ "RTX 3500 Ada" ]]; then
    echo -e "${YELLOW}  WARNING: Benchmarks tuned for RTX 3500 Ada${NC}"
fi

# Check VRAM availability (need 1GB+ free)
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
if [ "$GPU_FREE" -lt 1024 ]; then
    echo -e "${RED}ERROR: Insufficient VRAM. Need 1GB+ free, have ${GPU_FREE}MB${NC}"
    exit 1
fi

echo -e "${GREEN}  Environment OK!${NC}"
echo ""

# ===== Step 2: Build Release Binary =====
echo -e "${YELLOW}[2/6] Building release binary with GPU optimizations...${NC}"

cargo build --release --features gpu 2>&1 | grep -E "(Compiling|Finished)" || true

if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo -e "${RED}ERROR: Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}  Build complete!${NC}"
echo ""

# ===== Step 3: Run Benchmarks =====
if [ "$MODE" != "report-only" ]; then
    echo -e "${YELLOW}[3/6] Running benchmarks...${NC}"
    echo ""

    if [ "$MODE" == "quick" ]; then
        echo "  Mode: Quick (10-15 minutes)"
        echo "  Configs: 100x5k, 1000x10k only"
        echo ""

        # Run only key configurations
        cargo bench --bench "$BENCHMARK_NAME" --features gpu -- "100x5k|1000x10k" --save-baseline traditional

    else
        echo "  Mode: Full (60 minutes)"
        echo "  Configs: All dataset sizes"
        echo ""

        # Run all configurations (no filter to avoid criterion matching issues)
        echo -e "${BLUE}  Running all benchmark groups...${NC}"
        cargo bench --bench "$BENCHMARK_NAME" --features gpu
    fi

    echo ""
    echo -e "${GREEN}  Benchmarks complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}[3/6] Skipping benchmarks (report-only mode)${NC}"
    echo ""
fi

# ===== Step 4: Generate Statistical Report =====
echo -e "${YELLOW}[4/6] Generating statistical report...${NC}"

# Check if Python is available for statistical analysis
if command -v python3 &> /dev/null; then
    # TODO: Call Python script for statistical analysis
    # python3 scripts/analyze_optimization_results.py "$RESULTS_DIR" "$REPORT_FILE"
    echo "  (Python statistical analysis not yet implemented)"
else
    echo "  (Skipping statistical analysis - Python not available)"
fi

echo ""

# ===== Step 5: Extract Key Results =====
echo -e "${YELLOW}[5/6] Extracting key results...${NC}"

if [ -d "$RESULTS_DIR" ]; then
    echo ""
    echo "Benchmark results saved to:"
    echo "  - HTML Report: $RESULTS_DIR/report/index.html"
    echo "  - CSV Data: $RESULTS_DIR/$BENCHMARK_NAME/"
    echo ""

    # Display summary of key results if available
    if [ -f "$RESULTS_DIR/$BENCHMARK_NAME/1000x10k/base/estimates.json" ]; then
        echo "Key Result: 1000 strategies × 10K candles"
        # Extract mean from JSON (requires jq)
        if command -v jq &> /dev/null; then
            BASELINE=$(jq -r '.mean.point_estimate' "$RESULTS_DIR/$BENCHMARK_NAME/1_traditional_baseline/strategies_candles/1000x10k/base/estimates.json" 2>/dev/null || echo "N/A")
            PERSISTENT=$(jq -r '.mean.point_estimate' "$RESULTS_DIR/$BENCHMARK_NAME/2_persistent_kernels/strategies_candles/1000x10k/base/estimates.json" 2>/dev/null || echo "N/A")

            if [ "$BASELINE" != "N/A" ] && [ "$PERSISTENT" != "N/A" ]; then
                BASELINE_MS=$(echo "$BASELINE / 1000000" | bc -l)
                PERSISTENT_MS=$(echo "$PERSISTENT / 1000000" | bc -l)
                SPEEDUP=$(echo "$BASELINE / $PERSISTENT" | bc -l)

                printf "  Traditional:  %.2f ms\n" "$BASELINE_MS"
                printf "  Persistent:   %.2f ms\n" "$PERSISTENT_MS"
                printf "  Speedup:      %.2fx\n" "$SPEEDUP"

                # Check if target achieved
                TARGET_SPEEDUP=2.0
                if (( $(echo "$SPEEDUP >= $TARGET_SPEEDUP" | bc -l) )); then
                    echo -e "  ${GREEN}✓ Target achieved! (>= ${TARGET_SPEEDUP}x)${NC}"
                else
                    echo -e "  ${YELLOW}⚠ Target not met (< ${TARGET_SPEEDUP}x)${NC}"
                fi
            fi
        else
            echo "  (Install 'jq' for detailed summary)"
        fi
    fi
else
    echo -e "${YELLOW}  No results found in $RESULTS_DIR${NC}"
fi

echo ""

# ===== Step 6: Next Steps =====
echo -e "${YELLOW}[6/6] Summary and next steps${NC}"
echo ""
echo "✓ Benchmarks complete!"
echo ""
echo "Next steps:"
echo "  1. Review HTML report:"
echo "     firefox $RESULTS_DIR/report/index.html"
echo ""
echo "  2. Check statistical significance:"
echo "     Look for confidence intervals and speedup ratios"
echo ""
echo "  3. Validate targets:"
echo "     - Persistent kernels: >= 2.0x speedup"
echo "     - Phase 3 optimization: >= 1.4x speedup"
echo "     - Combined: >= 2.5x speedup"
echo ""
echo "  4. Update engine thresholds if needed:"
echo "     kimsfinance/core/engine.py"
echo ""

# Display summary table
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Performance Targets Summary${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
printf "%-30s | %-10s | %-10s\n" "Optimization" "Target" "Status"
echo "-----------------------------------------------------------"
printf "%-30s | %-10s | %-10s\n" "Persistent kernels" "2.0x" "TBD"
printf "%-30s | %-10s | %-10s\n" "Phase 3 optimization" "1.4x" "TBD"
printf "%-30s | %-10s | %-10s\n" "Combined" "2.5-3.0x" "TBD"
echo ""
echo "Run with actual benchmark data to populate Status column"
echo ""
