#!/bin/bash
set -e

echo "Building kimsfinance_core for all Python versions..."
cd "$(dirname "$0")/.."

# Update pyo3 dependencies (avoid cargo cache issues)
echo "Updating pyo3 dependencies..."
cargo update -p pyo3 -p pyo3-macros -p pyo3-macros-backend -p pyo3-ffi >/dev/null 2>&1
echo

# Python 3.13
if command -v python3.13 &> /dev/null; then
    echo "========================================"
    echo "Building for Python 3.13..."
    echo "========================================"

    if [ ! -d ".venv313" ]; then
        python3.13 -m venv .venv313
    fi

    source .venv313/bin/activate
    pip install -q maturin numpy
    maturin develop --release --features gpu
    python -c "import kimsfinance_core; print('✅ Python 3.13 build successful')"
    python -c "import sys; print(f'   GIL: {sys._is_gil_enabled() if hasattr(sys, \"_is_gil_enabled\") else \"Enabled (pre-3.13)\"}')"
    deactivate
    echo
else
    echo "⚠️  Python 3.13 not found, skipping"
    echo
fi

# Python 3.14
if command -v python3.14 &> /dev/null; then
    echo "========================================"
    echo "Building for Python 3.14..."
    echo "========================================"

    if [ ! -d ".venv314" ]; then
        python3.14 -m venv .venv314
    fi

    source .venv314/bin/activate
    pip install -q maturin numpy
    maturin develop --release --features gpu
    python -c "import kimsfinance_core; print('✅ Python 3.14 build successful')"
    python -c "import sys; print(f'   GIL: {sys._is_gil_enabled() if hasattr(sys, \"_is_gil_enabled\") else \"Enabled\"}')"
    deactivate
    echo
else
    echo "⚠️  Python 3.14 not found, skipping"
    echo
fi

# Python 3.14t (free-threading)
if command -v python3.14t &> /dev/null; then
    echo "========================================"
    echo "Building for Python 3.14t (free-threading)..."
    echo "========================================"

    # Check if it's really free-threading
    is_freethreaded=$(python3.14t -c "import sys; print('yes' if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled() else 'no')")

    if [ "$is_freethreaded" = "no" ]; then
        echo "⚠️  python3.14t is not a free-threading build, skipping"
        echo
    else
        if [ ! -d ".venv314t" ]; then
            python3.14t -m venv .venv314t
        fi

        source .venv314t/bin/activate
        pip install -q maturin numpy
        maturin develop --release --features gpu
        python -c "import sys; import kimsfinance_core; print(f'✅ Python 3.14t build successful (GIL: {sys._is_gil_enabled()})')"
        deactivate
        echo
    fi
else
    # Fallback: use .venv314t if it exists
    if [ -d ".venv314t" ]; then
        echo "========================================"
        echo "Rebuilding for Python 3.14t (existing venv)..."
        echo "========================================"

        source .venv314t/bin/activate
        pip install -q maturin numpy
        maturin develop --release --features gpu
        python -c "import sys; import kimsfinance_core; print(f'✅ Python 3.14t build successful (GIL: {sys._is_gil_enabled() if hasattr(sys, \"_is_gil_enabled\") else \"Enabled\"})')"
        deactivate
        echo
    else
        echo "⚠️  Python 3.14t not found (neither command nor .venv314t)"
        echo
    fi
fi

echo "========================================"
echo "✅ All Python version builds complete!"
echo "========================================"
