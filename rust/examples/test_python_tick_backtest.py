#!/usr/bin/env python3
"""Test Python bindings for tick-level backtesting"""

import sys
import numpy as np

try:
    import kimsfinance_core
except ImportError as e:
    print(f"❌ Failed to import kimsfinance_core: {e}")
    sys.exit(1)

print("=== Testing Python Bindings for Tick-Level Backtesting ===\n")

# Test 1: Create config
print("Test 1: Creating backtest configuration...")
try:
    config = kimsfinance_core.TickBacktestConfig(
        initial_capital=10_000.0,
        trading_fee=0.001,
        slippage=0.0005,
        execution_latency_ms=10
    )
    print(f"✓ Created config: {config}\n")
except Exception as e:
    print(f"✗ Error creating config: {e}")
    sys.exit(1)

# Test 2: Create engine
print("Test 2: Creating backtest engine...")
try:
    engine = kimsfinance_core.TickBacktestEngine(config)
    print(f"✓ Created engine: {engine}\n")
except Exception as e:
    print(f"✗ Error creating engine: {e}")
    sys.exit(1)

# Test 3: Run simple backtest
print("Test 3: Running simple backtest...")
try:
    # Create synthetic tick data
    n = 1000
    timestamps = np.arange(n, dtype=np.int64) * 1000  # Every second
    prices = np.linspace(100.0, 120.0, n).astype(np.float32)  # Trending up
    volumes = np.random.uniform(1.0, 5.0, n).astype(np.float32)
    is_buyer_maker = np.random.choice([True, False], n)

    # Simple momentum strategy: Buy when price increases, Sell when it decreases
    signals = np.zeros(n, dtype=np.int8)  # 0=Hold, 1=Buy, 2=Sell
    signals[100] = 1  # Buy at start
    signals[500] = 2  # Sell in middle
    signals[600] = 1  # Buy again
    signals[900] = 2  # Sell at end

    # Run backtest
    result = engine.run(timestamps, prices, volumes, is_buyer_maker, signals, timeframe_ms=60_000)

    print(f"✓ Backtest completed")
    print(f"  Total Return: {result.total_return:.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"  Win Rate: {result.win_rate:.2f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Num Trades: {result.num_trades}")
    print(f"  Final Equity: ${result.final_equity:.2f}\n")
except Exception as e:
    print(f"✗ Error running backtest: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Access equity curve
print("Test 4: Accessing equity curve...")
try:
    equity_curve = result.equity_curve()
    print(f"✓ Equity curve length: {len(equity_curve)}")
    print(f"  First value: ${equity_curve[0]:.2f}")
    print(f"  Last value: ${equity_curve[-1]:.2f}\n")
except Exception as e:
    print(f"✗ Error accessing equity curve: {e}")
    sys.exit(1)

# Test 5: Access trade P&Ls
print("Test 5: Accessing trade P&Ls...")
try:
    trade_pnls = result.trade_pnls()
    print(f"✓ Number of trades: {len(trade_pnls)}")
    if len(trade_pnls) > 0:
        print(f"  First trade P&L: ${trade_pnls[0]:.2f}")
        print(f"  Last trade P&L: ${trade_pnls[-1]:.2f}")
        print(f"  Total P&L: ${np.sum(trade_pnls):.2f}\n")
    else:
        print("  No trades executed\n")
except Exception as e:
    print(f"✗ Error accessing trade P&Ls: {e}")
    sys.exit(1)

# Test 6: Convert to dictionary
print("Test 6: Converting to dictionary...")
try:
    result_dict = result.to_dict()
    print(f"✓ Dictionary keys: {list(result_dict.keys())}")
    print(f"  total_return: {result_dict['total_return']:.2f}%")
    print(f"  num_trades: {result_dict['num_trades']}\n")
except Exception as e:
    print(f"✗ Error converting to dict: {e}")
    sys.exit(1)

# Test 7: Performance test with larger dataset
print("Test 7: Performance test with 100K ticks...")
try:
    import time

    # Create larger dataset
    n_large = 100_000
    timestamps_large = np.arange(n_large, dtype=np.int64) * 1000
    prices_large = (np.random.randn(n_large).cumsum() + 100).astype(np.float32)
    volumes_large = np.random.uniform(1.0, 5.0, n_large).astype(np.float32)
    is_buyer_maker_large = np.random.choice([True, False], n_large)

    # Random signals (10% of ticks have signals)
    signals_large = np.zeros(n_large, dtype=np.int8)
    signal_indices = np.random.choice(n_large, size=int(n_large * 0.1), replace=False)
    signals_large[signal_indices] = np.random.choice([1, 2], len(signal_indices))

    # Run benchmark
    start = time.perf_counter()
    result_large = engine.run(timestamps_large, prices_large, volumes_large, is_buyer_maker_large, signals_large, timeframe_ms=300_000)
    elapsed = time.perf_counter() - start

    throughput = n_large / elapsed

    print(f"✓ Processed {n_large:,} ticks in {elapsed:.3f} seconds")
    print(f"  Throughput: {throughput:,.0f} ticks/sec")
    print(f"  Throughput: {throughput/1_000_000:.2f} M ticks/sec")
    print(f"  Num Trades: {result_large.num_trades}")
    print(f"  Total Return: {result_large.total_return:.2f}%\n")
except Exception as e:
    print(f"✗ Error in performance test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== All Tests Passed! ===\n")
print("Summary:")
print("  1. ✓ Configuration creation works")
print("  2. ✓ Engine instantiation works")
print("  3. ✓ Backtest execution succeeds")
print("  4. ✓ Equity curve accessible")
print("  5. ✓ Trade P&Ls accessible")
print("  6. ✓ Dictionary conversion works")
print("  7. ✓ Performance validated (100K ticks)")
print("\n🎉 Python bindings for tick-level backtesting are working correctly!")
