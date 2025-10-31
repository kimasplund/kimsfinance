#!/bin/bash
# Automated analysis of Heston debug output to identify where imaginary parts become zero

set -e

cd "$(dirname "$0")/.."

echo "=== Heston Characteristic Function Debug Analyzer ==="
echo ""

# Run test and capture output
echo "[1/5] Building and running debug test..."
cargo run --example test_heston_debug --features heston --release 2>&1 > debug_output.txt 2>&1

echo "[2/5] Extracting CUDA debug lines..."
grep "CUDA_DEBUG" debug_output.txt > cuda_debug_only.txt || {
    echo "ERROR: No CUDA_DEBUG lines found in output!"
    echo "This means the kernel didn't print debug output."
    echo "Check if GPU is available and kernel compiled correctly."
    exit 1
}

echo "[3/5] Separating idx=0 (u=0) and idx=1 (u≠0)..."
grep "idx=0" cuda_debug_only.txt > idx0_debug.txt || echo "No idx=0 lines found"
grep "idx=1" cuda_debug_only.txt > idx1_debug.txt || echo "No idx=1 lines found"

echo ""
echo "=== Analysis Results ==="
echo ""

# Count lines
idx0_count=$(wc -l < idx0_debug.txt || echo 0)
idx1_count=$(wc -l < idx1_debug.txt || echo 0)

echo "Debug lines captured:"
echo "  idx=0 (u=0):  $idx0_count lines"
echo "  idx=1 (u≠0):  $idx1_count lines"
echo ""

# Check for non-zero imaginary parts in idx=1
echo "[4/5] Checking for zero imaginary parts in idx=1 (u≠0)..."
echo ""

if [ ! -f idx1_debug.txt ] || [ ! -s idx1_debug.txt ]; then
    echo "WARNING: No idx=1 debug output found!"
    echo "This could mean:"
    echo "  - FFT size < 2 (only 1 point)"
    echo "  - Kernel not printing idx=1"
    echo "  - GPU kernel failed to execute"
    exit 1
fi

# Find first occurrence of zero imaginary part in idx=1
first_zero=$(grep -n "idx=1" idx1_debug.txt | grep -E "\(.*,\s*[+-]?0\.0+\)" | head -1)

if [ -z "$first_zero" ]; then
    echo "✓ SUCCESS: All imaginary parts are NON-ZERO for idx=1!"
    echo ""
    echo "This means the complex arithmetic is working correctly."
    echo "The problem must be elsewhere (e.g., phi_values array, output transfer)."
    echo ""
    echo "Sample idx=1 values:"
    head -5 idx1_debug.txt
else
    echo "✗ BUG FOUND: First zero imaginary part in idx=1:"
    echo ""
    echo "$first_zero"
    echo ""

    # Extract variable name
    var_name=$(echo "$first_zero" | sed -n 's/.*\]: \([^=]*\)=.*/\1/p' | xargs)

    echo ">>> IDENTIFIED: Imaginary part becomes zero at: $var_name <<<"
    echo ""
    echo "Next steps:"
    echo "1. Check the calculation of '$var_name' in characteristic_function.cu"
    echo "2. Verify complex operator used in that calculation"
    echo "3. Compare formula against Heston reference implementation"
    echo ""
    echo "Debug output files:"
    echo "  - Full output: debug_output.txt"
    echo "  - CUDA debug only: cuda_debug_only.txt"
    echo "  - idx=0 lines: idx0_debug.txt"
    echo "  - idx=1 lines: idx1_debug.txt"
fi

echo ""
echo "[5/5] Showing idx=1 progression..."
echo ""
echo "=== idx=1 Complex Values (first 10) ==="
head -10 idx1_debug.txt

echo ""
echo "=== Analysis Complete ==="
echo ""
echo "Full debug output saved to: debug_output.txt"
echo "CUDA debug lines saved to: cuda_debug_only.txt"
