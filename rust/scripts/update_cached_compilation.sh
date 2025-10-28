#!/bin/bash
# Script to update all GPU indicator files to use cached compilation
# This provides 50-200x faster compilation on cache hits (2-4x overall speedup)

set -e

cd "$(dirname "$0")/.."

echo "=== Updating GPU indicator files to use cached compilation ==="
echo ""

# List of files to update (excluding compile.rs, mod.rs, and persistent/* which are already done)
FILES=(
    "src/gpu/aroon.rs"
    "src/gpu/atr.rs"
    "src/gpu/bollinger.rs"
    "src/gpu/cci.rs"
    "src/gpu/cmf.rs"
    "src/gpu/donchian.rs"
    "src/gpu/elder_ray.rs"
    "src/gpu/ema.rs"
    "src/gpu/keltner.rs"
    "src/gpu/kernels_2d.rs"
    "src/gpu/kernels_3d.rs"
    "src/gpu/macd.rs"
    "src/gpu/obv.rs"
    "src/gpu/roc.rs"
    "src/gpu/rsi.rs"
    "src/gpu/rsi_sync.rs"
    "src/gpu/sma.rs"
    "src/gpu/stochastic.rs"
    "src/gpu/vwap.rs"
    "src/gpu/vwma.rs"
    "src/gpu/williams_r.rs"
    "src/gpu/wma.rs"
)

total=${#FILES[@]}
updated=0

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Skipping $file (not found)"
        continue
    fi

    # Check if file uses compile_ptx_optimized
    if ! grep -q "compile_ptx_optimized" "$file"; then
        echo "⏭️  Skipping $file (doesn't use compile_ptx_optimized)"
        continue
    fi

    echo "Updating $file..."

    # Create backup
    cp "$file" "${file}.bak"

    # Update import statement
    sed -i 's/use crate::gpu::compile::compile_ptx_optimized;/use crate::gpu::compile::compile_ptx_optimized_cached;/' "$file"

    # Update function calls with multi-line support
    # Pattern 1: Simple assignment (most common)
    sed -i '/let ptx = compile_ptx_optimized(/,/)/ {
        s/let ptx = compile_ptx_optimized(/let ptx_arc = compile_ptx_optimized_cached(/
        s/)?;/)?;\n    let ptx = Arc::unwrap_or_clone(ptx_arc);/
    }' "$file"

    # Add Arc import if not present
    if ! grep -q "use std::sync::Arc;" "$file"; then
        # Add after other std imports
        sed -i '/^use std::/a use std::sync::Arc;' "$file"
    fi

    # Remove backup if successful
    if [ $? -eq 0 ]; then
        rm "${file}.bak"
        ((updated++))
        echo "✓ Updated $file"
    else
        mv "${file}.bak" "$file"
        echo "✗ Failed to update $file (restored backup)"
    fi
done

echo ""
echo "=== Summary ==="
echo "Updated: $updated / $total files"
echo ""
echo "Next steps:"
echo "1. Run: cargo check --features gpu"
echo "2. Run: cargo test --features gpu"
echo "3. Run: cargo run --example test_kernel_cache --features gpu"
echo ""
