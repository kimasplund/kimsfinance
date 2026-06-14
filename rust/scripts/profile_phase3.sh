#!/bin/bash
#
# Profile Phase 3 Execution Kernel with Nsight Compute
#
# This script profiles both original and optimized kernels to measure:
# - Memory bandwidth (DRAM throughput)
# - Shared memory bank conflicts
# - Register usage per thread
# - SM utilization
#
# Usage:
#   ./scripts/profile_phase3.sh

set -e

echo "🔍 Phase 3 Kernel Profiling with Nsight Compute"
echo "================================================"
echo ""

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "❌ Error: Nsight Compute (ncu) not found"
    echo "   Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Build release binary
echo "🔨 Building release binary..."
cd /home/kim/projects/kimsfinance/rust
cargo build --release --features gpu --example test_persistent_minimal
echo ""

# Create output directory
mkdir -p target/nsight_reports

# Profile original kernel (if exists)
echo "📊 Profiling ORIGINAL kernel (backtest_execution_kernel)..."
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --metrics launch__registers_per_thread \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name backtest_execution_kernel \
    --csv \
    --target-processes all \
    --export target/nsight_reports/phase3_original \
    ./target/release/examples/test_persistent_minimal 2>&1 | tee target/nsight_reports/phase3_original.log || true

echo ""

# Profile optimized kernel
echo "📊 Profiling OPTIMIZED kernel (backtest_execution_kernel_optimized)..."
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --metrics launch__registers_per_thread \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name backtest_execution_kernel_optimized \
    --csv \
    --target-processes all \
    --export target/nsight_reports/phase3_optimized \
    ./target/release/examples/test_persistent_minimal 2>&1 | tee target/nsight_reports/phase3_optimized.log

echo ""
echo "✅ Profiling complete!"
echo ""
echo "📈 Results saved to:"
echo "   - target/nsight_reports/phase3_original.ncu-rep"
echo "   - target/nsight_reports/phase3_optimized.ncu-rep"
echo "   - target/nsight_reports/phase3_original.log"
echo "   - target/nsight_reports/phase3_optimized.log"
echo ""
echo "📊 View with Nsight Compute GUI:"
echo "   ncu-ui target/nsight_reports/phase3_optimized.ncu-rep"
echo ""

# Extract key metrics
echo "🔍 Key Metrics Comparison:"
echo "=========================="
echo ""

echo "ORIGINAL Kernel:"
grep -E "(sm__throughput|dram__throughput|bank_conflicts|registers_per_thread)" target/nsight_reports/phase3_original.log | head -6 || echo "   No data available"
echo ""

echo "OPTIMIZED Kernel:"
grep -E "(sm__throughput|dram__throughput|bank_conflicts|registers_per_thread)" target/nsight_reports/phase3_optimized.log | head -6 || echo "   No data available"
echo ""

echo "✨ To see full analysis, open reports in Nsight Compute GUI"
