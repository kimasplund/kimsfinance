#!/usr/bin/env python3
"""
Batch update all GPU indicator files to use cached compilation.
This provides 50-200x faster compilation on cache hits (2-4x overall speedup).
"""

import re
import sys
from pathlib import Path

# List of files to update (excluding already updated ones)
FILES_TO_UPDATE = [
    "src/gpu/aroon.rs",
    "src/gpu/atr.rs",
    "src/gpu/bollinger.rs",
    "src/gpu/cci.rs",
    "src/gpu/cmf.rs",
    "src/gpu/donchian.rs",
    "src/gpu/elder_ray.rs",
    "src/gpu/ema.rs",
    "src/gpu/keltner.rs",
    "src/gpu/kernels_2d.rs",
    "src/gpu/kernels_3d.rs",
    "src/gpu/macd.rs",
    "src/gpu/obv.rs",
    "src/gpu/roc.rs",
    # "src/gpu/rsi.rs",  # Already updated
    "src/gpu/rsi_sync.rs",
    "src/gpu/sma.rs",
    "src/gpu/stochastic.rs",
    "src/gpu/vwap.rs",
    "src/gpu/vwma.rs",
    "src/gpu/williams_r.rs",
    "src/gpu/wma.rs",
]

def update_file(filepath: Path) -> bool:
    """Update a single file to use cached compilation. Returns True if updated."""
    try:
        content = filepath.read_text()
        original = content

        # Step 1: Update import statement
        content = re.sub(
            r'use crate::gpu::compile::compile_ptx_optimized;',
            r'use crate::gpu::compile::compile_ptx_optimized_cached;',
            content
        )

        # Step 2: Update compilation calls
        # Pattern: let ptx = compile_ptx_optimized(KERNEL)...
        # Replace with: let ptx_arc = compile_ptx_optimized_cached(KERNEL)...; let ptx = Arc::unwrap_or_clone(ptx_arc);

        # Find all instances of compilation
        pattern = r'([ \t]*)let ptx = compile_ptx_optimized\(([^)]+)\)(.*?)\)\?;'

        def replacer(match):
            indent = match.group(1)
            kernel_name = match.group(2)
            map_err_part = match.group(3)

            return f'''{indent}let ptx_arc = compile_ptx_optimized_cached({kernel_name}){map_err_part})?;
{indent}let ptx = Arc::unwrap_or_clone(ptx_arc);'''

        content = re.sub(pattern, replacer, content, flags=re.DOTALL)

        if content != original:
            # Backup and write
            filepath.with_suffix('.rs.bak').write_text(original)
            filepath.write_text(content)
            print(f"✓ Updated {filepath}")
            return True
        else:
            print(f"⏭️  Skipped {filepath} (no changes needed)")
            return False

    except Exception as e:
        print(f"✗ Failed to update {filepath}: {e}", file=sys.stderr)
        return False

def main():
    base_dir = Path(__file__).parent.parent
    updated_count = 0

    print("=== Updating GPU indicator files to use cached compilation ===\n")

    for file_path in FILES_TO_UPDATE:
        full_path = base_dir / file_path
        if not full_path.exists():
            print(f"⚠️  Skipping {file_path} (not found)")
            continue

        if update_file(full_path):
            updated_count += 1

    print(f"\n=== Summary ===")
    print(f"Updated: {updated_count} / {len(FILES_TO_UPDATE)} files")
    print("\nNext steps:")
    print("1. Run: cargo check --features gpu")
    print("2. Run: cargo test --features gpu")
    print("3. Run: cargo run --example test_kernel_cache --features gpu")

if __name__ == "__main__":
    main()
