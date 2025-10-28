#!/usr/bin/env python3
"""Test free-threading parallel execution on Python 3.14t."""

import numpy as np
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Check if free-threading is enabled
if not hasattr(sys, '_is_gil_enabled') or sys._is_gil_enabled():
    print("⚠️  Skipping free-threading test: GIL is enabled")
    print(f"   Python version: {sys.version}")
    print("   This test requires Python 3.14t (free-threading build)")
    sys.exit(0)

print(f"✅ Running on Python 3.14t (GIL disabled)")

try:
    from kimsfinance_core import batch_backtest
except ImportError:
    print("❌ kimsfinance_core not installed")
    sys.exit(1)

# Generate multiple datasets
np.random.seed(42)
n_datasets = 8
datasets = []

for i in range(n_datasets):
    n_candles = 1000
    ohlcv = np.random.randn(n_candles, 5).cumsum(axis=0) + 100 + i * 10
    ohlcv[:, 0:4] = np.abs(ohlcv[:, 0:4])
    ohlcv[:, 4] = np.abs(ohlcv[:, 4]) * 1000
    datasets.append(ohlcv)

parameters = [[14.0, 20 + i, 70 + i] for i in range(100)]

def process_dataset(idx):
    """Process a single dataset."""
    results = batch_backtest(
        strategy='rsi_crossover',
        ohlcv=datasets[idx],
        parameters=parameters,
        execution_mode='fused'
    )
    return results[0].sharpe_ratio

# Test 1: Sequential execution (baseline)
print(f"\nProcessing {n_datasets} datasets sequentially...")
start = time.time()
sequential_results = [process_dataset(i) for i in range(n_datasets)]
sequential_time = time.time() - start
print(f"Sequential: {sequential_time:.2f}s")

# Test 2: Parallel execution with 2 threads
print(f"\nProcessing {n_datasets} datasets with 2 threads...")
start = time.time()
with ThreadPoolExecutor(max_workers=2) as executor:
    parallel_2_results = list(executor.map(process_dataset, range(n_datasets)))
parallel_2_time = time.time() - start
speedup_2 = sequential_time / parallel_2_time
print(f"Parallel (2 threads): {parallel_2_time:.2f}s")
print(f"Speedup: {speedup_2:.2f}x")

# Test 3: Parallel execution with 4 threads
print(f"\nProcessing {n_datasets} datasets with 4 threads...")
start = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    parallel_4_results = list(executor.map(process_dataset, range(n_datasets)))
parallel_4_time = time.time() - start
speedup_4 = sequential_time / parallel_4_time
print(f"Parallel (4 threads): {parallel_4_time:.2f}s")
print(f"Speedup: {speedup_4:.2f}x")

# Test 4: Parallel execution with 8 threads
print(f"\nProcessing {n_datasets} datasets with 8 threads...")
start = time.time()
with ThreadPoolExecutor(max_workers=8) as executor:
    parallel_8_results = list(executor.map(process_dataset, range(n_datasets)))
parallel_8_time = time.time() - start
speedup_8 = sequential_time / parallel_8_time
print(f"Parallel (8 threads): {parallel_8_time:.2f}s")
print(f"Speedup: {speedup_8:.2f}x")

# Validate correctness
assert sequential_results == parallel_2_results == parallel_4_results == parallel_8_results, \
    "Results differ between sequential and parallel execution!"

print("\n=== Free-Threading Performance Summary ===")
print(f"2 threads:  {speedup_2:.2f}x speedup (expected ~1.8x)")
print(f"4 threads:  {speedup_4:.2f}x speedup (expected ~3.5x)")
print(f"8 threads:  {speedup_8:.2f}x speedup (expected ~7.0x)")

# Validate expectations
if speedup_2 >= 1.5:
    print("✅ 2-thread speedup is good (>= 1.5x)")
else:
    print(f"⚠️  2-thread speedup is low ({speedup_2:.2f}x, expected >= 1.5x)")

if speedup_4 >= 2.5:
    print("✅ 4-thread speedup is good (>= 2.5x)")
else:
    print(f"⚠️  4-thread speedup is low ({speedup_4:.2f}x, expected >= 2.5x)")

if speedup_8 >= 5.0:
    print("✅ 8-thread speedup is good (>= 5.0x)")
else:
    print(f"⚠️  8-thread speedup is low ({speedup_8:.2f}x, expected >= 5.0x)")

print("\n✅ Free-threading test complete!")
