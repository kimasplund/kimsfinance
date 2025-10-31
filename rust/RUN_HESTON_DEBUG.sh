#!/bin/bash
# Master script to run Heston debug analysis
# This will build, test, and automatically identify the bug

set -e

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Heston Characteristic Function Debug - Root Cause Analyzer       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will:"
echo "  1. Build the debug test with Heston GPU support"
echo "  2. Run the test and capture CUDA printf output"
echo "  3. Analyze output to find where imaginary parts become zero"
echo "  4. Identify the buggy operator in the CUDA kernel"
echo ""
echo "Press Ctrl+C to abort, or wait 3 seconds to continue..."
sleep 3
echo ""

# Make scripts executable
chmod +x scripts/test_heston_debug.sh
chmod +x scripts/analyze_heston_debug.sh

# Run analysis
bash scripts/analyze_heston_debug.sh

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Debug Analysis Complete                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Review the output above to see where imaginary parts first become zero."
echo ""
echo "Generated files:"
echo "  - debug_output.txt       : Full test output"
echo "  - cuda_debug_only.txt    : Only CUDA_DEBUG lines"
echo "  - idx0_debug.txt         : Debug for u=0 (first FFT point)"
echo "  - idx1_debug.txt         : Debug for u≠0 (second FFT point)"
echo ""
echo "Next steps:"
echo "  1. Review idx1_debug.txt to see where imag first becomes zero"
echo "  2. Check corresponding operator in characteristic_function.cu"
echo "  3. Fix the buggy operator"
echo "  4. Re-run this script to verify fix"
echo ""
