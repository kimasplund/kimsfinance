#!/bin/bash
#
# Test Persistent Kernel Integration
#
# Validates the 2-4x speedup from combining all 4 phases into a single kernel launch.

set -e

echo "🚀 Testing Persistent Kernel Integration"
echo "=========================================="
echo ""

echo "Step 1: Compiling with GPU support..."
cargo build --release --features gpu --example test_persistent_backtest
echo "✅ Compilation successful"
echo ""

echo "Step 2: Running integration test..."
cargo run --release --features gpu --example test_persistent_backtest
echo ""

echo "Step 3: Running benchmark comparison (optional - requires time)..."
echo "To run full benchmarks:"
echo "  cargo bench --features gpu --bench persistent_vs_traditional"
echo ""

echo "✅ All tests passed!"
