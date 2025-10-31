#!/bin/bash
# Test script to capture CUDA debug output from Heston characteristic function

set -e

cd "$(dirname "$0")/.."

echo "=== Building minimal Heston debug test ==="
cargo build --example test_heston_debug --features heston --release

echo ""
echo "=== Running test with CUDA debug output ==="
echo "Looking for CUDA_DEBUG lines and characteristic function diagnostics..."
echo ""
cargo run --example test_heston_debug --features heston --release 2>&1

echo ""
echo "=== Test complete ==="
