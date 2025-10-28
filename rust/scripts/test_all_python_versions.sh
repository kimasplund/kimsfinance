#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Testing kimsfinance_core Python API on all Python versions ==="
echo

# Function to run tests in a venv
run_tests() {
    local python_ver=$1
    local venv_dir=$2

    if [ ! -d "$venv_dir" ]; then
        echo "⚠️  $venv_dir not found, skipping $python_ver"
        return
    fi

    echo "=== Testing $python_ver ==="
    source "$venv_dir/bin/activate"

    echo "1. Correctness test..."
    python python_tests/test_multiversion_correctness.py

    echo "2. Performance benchmark..."
    python python_tests/benchmark_multiversion_performance.py

    if [[ "$python_ver" == *"3.14t"* ]]; then
        echo "3. Free-threading test..."
        python python_tests/test_free_threading.py
    fi

    deactivate
    echo "✅ $python_ver tests complete!"
    echo
}

# Test each Python version
run_tests "Python 3.13" ".venv313"
run_tests "Python 3.14" ".venv314"
run_tests "Python 3.14t" ".venv314t"

echo "✅ All Python version tests complete!"
