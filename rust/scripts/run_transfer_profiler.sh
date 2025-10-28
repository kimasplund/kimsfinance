#!/usr/bin/env bash
#
# GPU Data Transfer Overhead Profiler Runner
#
# Profiles GPU batch backtest to identify bottlenecks:
# - H2D transfer time
# - Kernel execution time
# - D2H transfer time
# - Memory allocation time
#
# Usage:
#   ./scripts/run_transfer_profiler.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "  GPU Data Transfer Overhead Profiler"
echo "  Analyzing bottlenecks in persistent kernel execution"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "❌ ERROR: NVIDIA GPU not detected"
    echo "   This profiler requires a CUDA-capable GPU"
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Build in release mode for accurate timing
echo "🔨 Building profiler (release mode)..."
cargo build --example profile_transfer_overhead --features gpu --release --quiet

echo ""
echo "🚀 Running profiler..."
echo ""

# Run the profiler
cargo run --example profile_transfer_overhead --features gpu --release

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Profiling Complete"
echo "════════════════════════════════════════════════════════════════"
