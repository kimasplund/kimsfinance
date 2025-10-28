#!/bin/bash
# Profile triple-buffer timeline with Nsight Systems
#
# Shows overlapping H2D, kernel execution, and D2H transfers for validation.

set -e

echo "=== Profiling Triple-Buffer Timeline ==="
echo

# Check if Nsight Systems is installed
if ! command -v nsys &> /dev/null; then
    echo "❌ Error: Nsight Systems (nsys) not found"
    echo "   Install from: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# Check if cargo bench exists
if [ ! -f "benches/async_execution_benchmark.rs" ]; then
    echo "❌ Error: Benchmark not found: benches/async_execution_benchmark.rs"
    exit 1
fi

# Run with Nsight Systems
echo "Running Nsight Systems profiling..."
echo "  Target: async_1000 (1000 strategies × 10K candles)"
echo "  Expected: 296ms (1.3x faster than fused 385ms)"
echo

nsys profile \
    --output=/tmp/triple_buffer_timeline.nsys-rep \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    cargo bench --bench async_execution_benchmark --features gpu -- async_1000 --sample-size 3

echo
echo "✅ Profile saved to /tmp/triple_buffer_timeline.nsys-rep"
echo

# Analyze timeline
echo "=== Timeline Analysis ==="
nsys stats /tmp/triple_buffer_timeline.nsys-rep | grep -E "(CUDA|Memory|Kernel)" | head -30

echo
echo "=== Key Metrics to Look For ==="
echo "1. Multiple kernel launches overlapping"
echo "2. H2D/D2H transfers concurrent with kernels"
echo "3. GPU utilization: 80-90% (vs ~60% synchronous)"
echo "4. Stream usage: 3 streams active"
echo "5. Pipeline depth: 3 batches in flight"
echo

echo "Open in Nsight Systems GUI:"
echo "  nsys-ui /tmp/triple_buffer_timeline.nsys-rep"
echo

echo "Expected Timeline:"
echo "  Time →"
echo "        [H2D-1] [Kernel-1] [D2H-1]"
echo "                [H2D-2]    [Kernel-2] [D2H-2]"
echo "                           [H2D-3]    [Kernel-3] [D2H-3]"
echo

echo "✅ Profiling complete!"
