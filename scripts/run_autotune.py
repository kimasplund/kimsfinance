#!/usr/bin/env python3
"""
Run Auto-Tune for GPU Crossover Thresholds
===========================================

⚠️  IMPORTANT: This is the BASIC autotune (sequential, 3 indicators only)

For COMPREHENSIVE autotune that includes:
- All 9 indicators
- Batch processing scenarios (6+ indicators)
- Parallel execution patterns (real-world usage)
- 66.7x better GPU efficiency for batch operations

Run: python scripts/run_autotune_comprehensive.py

This basic version is kept for quick testing only.
Results are saved to: ~/.kimsfinance/threshold_cache.json
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimsfinance.core.autotune import run_autotune, load_tuned_thresholds, CACHE_FILE
from kimsfinance.config.gpu_thresholds import GPU_THRESHOLDS


# Color codes for output
class C:
    H = "\033[95m"
    B = "\033[94m"
    C = "\033[96m"
    G = "\033[92m"
    Y = "\033[93m"
    R = "\033[91m"
    E = "\033[0m"
    BOLD = "\033[1m"


def main():
    """Run the basic auto-tuning process."""
    print(f"\n{C.BOLD}{'█'*70}{C.E}")
    print(f"{C.BOLD}GPU Crossover Threshold Auto-Tuning (Basic){C.E}")
    print(f"{C.BOLD}{'█'*70}{C.E}\n")

    print(f"{C.Y}⚠️  WARNING: This is the BASIC autotune (sequential, 3 indicators){C.E}")
    print(f"{C.Y}For comprehensive results, use: run_autotune_comprehensive.py{C.E}\n")

    print(f"{C.C}This will benchmark CPU vs GPU performance at various data sizes")
    print(f"to determine optimal crossover thresholds for your hardware.{C.E}\n")

    # Show current default thresholds
    print(f"{C.BOLD}Current Default Thresholds:{C.E}")
    for key, value in GPU_THRESHOLDS.items():
        print(f"  {C.Y}{key:25s}{C.E}: {value:,} rows")

    # Check if cache exists
    if CACHE_FILE.exists():
        print(f"\n{C.Y}Found existing tuned thresholds: {CACHE_FILE}{C.E}")
        existing = load_tuned_thresholds()
        print(f"{C.BOLD}Existing Tuned Thresholds:{C.E}")
        for key, value in existing.items():
            print(f"  {C.G}{key:25s}{C.E}: {value:,} rows")
        print()

    # Operations to tune
    operations = ["atr", "rsi", "stochastic"]

    print(f"{C.BOLD}Starting Auto-Tune for {len(operations)} operations...{C.E}\n")
    print(f"{C.C}Test sizes: 10K, 50K, 100K, 200K, 500K, 1M rows{C.E}")
    print(f"{C.C}This may take 2-5 minutes...{C.E}\n")

    # Run autotune
    print(f"{C.BOLD}{'='*70}{C.E}")
    tuned_thresholds = run_autotune(operations=operations, save=True)
    print(f"{C.BOLD}{'='*70}{C.E}\n")

    # Display results
    print(f"{C.BOLD}Auto-Tune Results:{C.E}\n")
    for op, threshold in tuned_thresholds.items():
        default = GPU_THRESHOLDS.get("default", 100_000)

        if threshold < default:
            color = C.G
            msg = f"(GPU beneficial at smaller sizes)"
        elif threshold > default:
            color = C.Y
            msg = f"(GPU beneficial at larger sizes)"
        else:
            color = C.C
            msg = f"(matches default)"

        print(f"  {color}{op:15s}: {threshold:>10,} rows {msg}{C.E}")

    print(f"\n{C.G}Results saved to: {CACHE_FILE}{C.E}")
    print(f"\n{C.BOLD}Recommendations:{C.E}")
    print(f"  • Use engine='auto' to automatically apply these thresholds")
    print(f"  • Re-run autotune if you change hardware or drivers")
    print(f"  • Smaller thresholds = GPU is faster on your hardware")
    print(f"  • Larger thresholds = CPU is more efficient for your use case")

    print(f"\n{C.BOLD}{'='*70}{C.E}")
    print(f"{C.G}Auto-tune complete!{C.E}")
    print(f"{C.BOLD}{'='*70}{C.E}\n")


if __name__ == "__main__":
    main()
