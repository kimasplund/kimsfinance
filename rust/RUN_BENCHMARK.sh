#!/bin/bash
#
# Quick launcher for Stochastic Oscillator CPU vs GPU Benchmark
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Stochastic Oscillator: CPU vs GPU Benchmark"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "⚠️  Warning: Virtual environment not found at $PROJECT_DIR/.venv"
    echo "   Running with system Python..."
    echo ""
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_DIR/.venv" ]; then
    echo "✓ Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Check if CuPy is installed
echo "✓ Checking dependencies..."
if ! python -c "import cupy" 2>/dev/null; then
    echo "⚠️  Warning: CuPy not installed"
    echo "   GPU benchmarks will be skipped"
    echo "   Install with: pip install cupy-cuda12x"
    echo ""
fi

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
else
    echo "⚠️  Warning: nvidia-smi not found"
    echo "   GPU may not be available"
    echo ""
fi

# Run benchmark
echo "Starting benchmark..."
echo ""
cd "$PROJECT_DIR"
python "$SCRIPT_DIR/benchmark_gpu_vs_cpu.py" "$@"

# Deactivate virtual environment
if [ -d "$PROJECT_DIR/.venv" ]; then
    deactivate 2>/dev/null || true
fi

echo ""
echo "=================================================="
echo "Benchmark complete!"
echo "=================================================="
