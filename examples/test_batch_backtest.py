#!/usr/bin/env python3
"""
Quick test of GPU batch backtest Python bindings.

This script validates that the PyO3 bindings are working correctly.
"""

import numpy as np
import sys

try:
    from kimsfinance_core import batch_backtest, batch_backtest_info, BacktestResult
    print("✓ Successfully imported batch_backtest functions")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("Make sure to build with GPU feature: cargo build --release --features gpu")
    sys.exit(1)


def main():
    print("\n" + "=" * 60)
    print("GPU Batch Backtest Python Bindings Test")
    print("=" * 60)

    # Test 1: Check GPU info
    print("\n1. GPU Information:")
    info = batch_backtest_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    if not info['gpu_available']:
        print("\n✗ GPU not available. Stopping tests.")
        return

    # Test 2: Generate synthetic OHLCV data
    print("\n2. Generating synthetic OHLCV data (1000 candles)...")
    np.random.seed(42)
    n_candles = 1000

    # Random walk prices
    returns = np.random.randn(n_candles) * 0.01
    close = 100.0 * np.exp(np.cumsum(returns))

    # Generate OHLC
    ohlcv = np.zeros((n_candles, 5))
    ohlcv[:, 3] = close  # close
    ohlcv[:, 1] = close * 1.01  # high
    ohlcv[:, 2] = close * 0.99  # low
    ohlcv[:, 0] = close * (1 + np.random.randn(n_candles) * 0.005)  # open
    ohlcv[:, 4] = np.abs(np.random.randn(n_candles)) * 1000  # volume

    print(f"   OHLCV shape: {ohlcv.shape}")
    print(f"   Price range: ${ohlcv[:, 2].min():.2f} - ${ohlcv[:, 1].max():.2f}")

    # Test 3: Run batch backtest with 10 strategies
    print("\n3. Running batch backtest (10 RSI strategies)...")
    parameters = [
        [14.0, 20.0 + i * 2, 70.0 + i]
        for i in range(10)
    ]

    print(f"   Parameters: {len(parameters)} strategies")
    print(f"   Example: period={parameters[0][0]}, buy={parameters[0][1]}, sell={parameters[0][2]}")

    try:
        import time
        start = time.perf_counter()
        results = batch_backtest(
            strategy='rsi_crossover',
            ohlcv=ohlcv,
            parameters=parameters,
            initial_capital=10000.0,
            trading_fee=0.001,
            slippage=0.0001
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"   Completed in {elapsed_ms:.1f}ms")
    except Exception as e:
        print(f"✗ Batch backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 4: Validate results
    print("\n4. Validating results...")
    print(f"   Number of results: {len(results)}")

    if len(results) != 10:
        print(f"✗ Expected 10 results, got {len(results)}")
        return

    for i, result in enumerate(results):
        if not isinstance(result, BacktestResult):
            print(f"✗ Result {i} is not BacktestResult, got {type(result)}")
            return

    print("   ✓ All results are BacktestResult objects")

    # Test 5: Print top 3 results
    print("\n5. Top 3 strategies:")
    for i, result in enumerate(results[:3]):
        print(f"\n   {i+1}. Strategy #{result.strategy_id}")
        print(f"      Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"      Max Drawdown: {result.max_drawdown:.2%}")
        print(f"      Win Rate: {result.win_rate:.1%}")
        print(f"      Total Return: {result.total_return:.2%}")
        print(f"      Final Equity: ${result.final_equity:.2f}")
        print(f"      Trades: {result.num_trades}")
        print(f"      Fitness: {result.fitness():.2f}")

    # Test 6: Test to_dict() method
    print("\n6. Testing to_dict() method...")
    result_dict = results[0].to_dict()
    print(f"   Keys: {list(result_dict.keys())}")
    print(f"   ✓ to_dict() works")

    # Test 7: Test __repr__
    print("\n7. Testing __repr__...")
    repr_str = repr(results[0])
    print(f"   {repr_str}")
    print(f"   ✓ __repr__ works")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
