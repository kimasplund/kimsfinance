#!/usr/bin/env python3
"""
GPU-Accelerated Orderflow Analysis with kimsfinance_core

This example demonstrates how to use the Python bindings for GPU-accelerated
orderflow feature extraction and multi-strategy signal generation.

Features:
- GPU-accelerated orderflow processing (500M-1B features/sec)
- 6 orderflow features per tick (imbalance, delta, intensity, velocity, etc.)
- Multi-strategy parallel execution (10-20 strategies simultaneously)
- Fused kernel architecture (eliminates 48-60MB intermediate memory transfers)
- INT8 quantization for 8x memory compression

Performance:
- 10 strategies × 1M ticks: ~150-200ms
- Feature throughput: 500M-1B features/sec
- Signal throughput: 3-4B signals/sec
"""

import sys
import numpy as np
import time

# Import kimsfinance_core Rust extension
try:
    import kimsfinance_core
except ImportError as e:
    print(f"❌ Failed to import kimsfinance_core: {e}")
    print("\n💡 Make sure to build the Rust extension first:")
    print("   cargo build --release --features gpu,python")
    print("   export PYTHONPATH=$(pwd)/target/release:$PYTHONPATH")
    sys.exit(1)

print("=" * 70)
print("🚀 GPU-Accelerated Orderflow Analysis Example")
print("=" * 70)


# ============================================================================
# EXAMPLE 1: Basic Usage - Single Strategy
# ============================================================================
def example_1_basic_usage():
    """Demonstrate basic orderflow processing with a single strategy."""
    print("\n📖 EXAMPLE 1: Basic Usage - Single Strategy")
    print("-" * 70)

    print("Step 1: Check GPU availability...")
    if not kimsfinance_core.orderflow_gpu_available():
        print("⚠️  GPU not available! This example requires GPU support.")
        print("   Build with: cargo build --release --features gpu,python")
        return
    print("✅ GPU is available")

    print("\nStep 2: Create orderflow processor...")
    try:
        processor = kimsfinance_core.OrderflowProcessor()
        print(f"✅ Created processor: {processor}")
    except RuntimeError as e:
        print(f"❌ Failed to initialize GPU: {e}")
        return

    print("\nStep 3: Generate synthetic tick data...")
    n = 10_000
    timestamps = np.arange(n, dtype=np.int64) * 100  # 100ms apart

    # Simulate price random walk
    price_changes = np.random.randn(n) * 10.0
    close_prices = (50000.0 + np.cumsum(price_changes)).astype(np.float32)

    # Simulate volume with lognormal distribution
    volumes = np.random.exponential(100.0, n).astype(np.float32)

    # Simulate buy/sell split (with some imbalance)
    imbalance = np.random.uniform(0.3, 0.7, n).astype(np.float32)
    buy_volumes = (volumes * imbalance).astype(np.float32)
    sell_volumes = (volumes - buy_volumes).astype(np.float32)

    print(f"✅ Generated {n:,} ticks")
    print(f"   Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
    print(f"   Avg volume: {volumes.mean():.2f}")

    print("\nStep 4: Configure strategy (momentum)...")
    strategies = [kimsfinance_core.StrategyConfig.momentum()]
    print(f"✅ Strategy: {strategies[0]}")

    print("\nStep 5: Process batch...")
    start = time.perf_counter()
    result = processor.process_batch(
        timestamps, close_prices, volumes, buy_volumes, sell_volumes, strategies
    )
    elapsed = time.perf_counter() - start

    print(f"✅ Processing complete in {elapsed*1000:.2f}ms")
    print(f"   Throughput: {n/elapsed:,.0f} ticks/sec")

    print("\nStep 6: Analyze results...")
    signals = result.signals  # NumPy array [num_strategies, num_ticks]
    features = result.features  # NumPy array [num_strategies, num_ticks * 6]

    print(f"   Signals shape: {signals.shape}")
    print(f"   Features shape: {features.shape}")

    # Count signal types
    momentum_signals = signals[0]
    buy_count = np.sum(momentum_signals == 1)
    sell_count = np.sum(momentum_signals == -1)
    hold_count = np.sum(momentum_signals == 0)

    print(f"\n   Signal distribution:")
    print(f"     Buy:  {buy_count:5d} ({buy_count/n*100:5.2f}%)")
    print(f"     Sell: {sell_count:5d} ({sell_count/n*100:5.2f}%)")
    print(f"     Hold: {hold_count:5d} ({hold_count/n*100:5.2f}%)")

    print("\n✅ Example 1 complete!")


# ============================================================================
# EXAMPLE 2: Multiple Strategies in Parallel
# ============================================================================
def example_2_multiple_strategies():
    """Demonstrate parallel processing of multiple strategies."""
    print("\n📖 EXAMPLE 2: Multiple Strategies in Parallel")
    print("-" * 70)

    if not kimsfinance_core.orderflow_gpu_available():
        print("⚠️  GPU not available! Skipping this example.")
        return

    print("Step 1: Initialize processor...")
    processor = kimsfinance_core.OrderflowProcessor()

    print("\nStep 2: Generate realistic market data (100K ticks)...")
    n = 100_000
    timestamps = np.arange(n, dtype=np.int64) * 100

    # More realistic price simulation with trends
    trend = np.linspace(0, 5000, n)
    noise = np.random.randn(n).cumsum() * 10
    close_prices = (50000.0 + trend + noise).astype(np.float32)

    volumes = np.random.exponential(100.0, n).astype(np.float32)
    imbalance = np.random.uniform(0.3, 0.7, n).astype(np.float32)
    buy_volumes = (volumes * imbalance).astype(np.float32)
    sell_volumes = (volumes - buy_volumes).astype(np.float32)

    print(f"✅ Generated {n:,} ticks")

    print("\nStep 3: Configure 5 different strategies...")
    strategies = [
        kimsfinance_core.StrategyConfig.momentum(),
        kimsfinance_core.StrategyConfig.mean_reversion(),
        kimsfinance_core.StrategyConfig.breakout(),
        kimsfinance_core.StrategyConfig.scalping(),
        kimsfinance_core.StrategyConfig.trend_following(),
    ]

    strategy_names = ["Momentum", "Mean Reversion", "Breakout", "Scalping", "Trend Following"]

    for name, strategy in zip(strategy_names, strategies):
        print(f"   {name}: {strategy.strategy_type}")

    print("\nStep 4: Process all strategies in parallel...")
    start = time.perf_counter()
    result = processor.process_batch(
        timestamps, close_prices, volumes, buy_volumes, sell_volumes, strategies
    )
    elapsed = time.perf_counter() - start

    total_signals = len(strategies) * n
    throughput = total_signals / elapsed

    print(f"✅ Processed {len(strategies)} strategies in {elapsed*1000:.2f}ms")
    print(f"   Total signals: {total_signals:,}")
    print(f"   Throughput: {throughput/1e9:.2f}B signals/sec")

    print("\nStep 5: Compare strategy signals...")
    signals = result.signals

    print("\n   Strategy Performance:")
    print("   " + "=" * 66)
    print("   Strategy           | Buy    | Sell   | Hold   | Buy%  | Sell% |")
    print("   " + "-" * 66)

    for i, name in enumerate(strategy_names):
        strategy_signals = signals[i]
        buy_count = np.sum(strategy_signals == 1)
        sell_count = np.sum(strategy_signals == -1)
        hold_count = np.sum(strategy_signals == 0)

        buy_pct = buy_count / n * 100
        sell_pct = sell_count / n * 100

        print(f"   {name:18} | {buy_count:6d} | {sell_count:6d} | {hold_count:6d} | {buy_pct:5.2f} | {sell_pct:5.2f} |")

    print("   " + "=" * 66)
    print("\n✅ Example 2 complete!")


# ============================================================================
# EXAMPLE 3: Feature Calibration
# ============================================================================
def example_3_calibration():
    """Demonstrate feature range calibration for custom strategies."""
    print("\n📖 EXAMPLE 3: Feature Calibration")
    print("-" * 70)

    if not kimsfinance_core.orderflow_gpu_available():
        print("⚠️  GPU not available! Skipping this example.")
        return

    print("Step 1: Initialize processor...")
    processor = kimsfinance_core.OrderflowProcessor()

    print("\nStep 2: Generate data sample for calibration...")
    n = 50_000
    timestamps = np.arange(n, dtype=np.int64) * 100
    close_prices = (50000.0 + np.random.randn(n).cumsum() * 10).astype(np.float32)
    volumes = np.random.exponential(100.0, n).astype(np.float32)
    imbalance = np.random.uniform(0.3, 0.7, n).astype(np.float32)
    buy_volumes = (volumes * imbalance).astype(np.float32)
    sell_volumes = (volumes - buy_volumes).astype(np.float32)

    print(f"✅ Generated {n:,} calibration ticks")

    print("\nStep 3: Calibrate feature ranges...")
    start = time.perf_counter()
    ranges = processor.calibrate_ranges(
        timestamps, close_prices, volumes, buy_volumes, sell_volumes
    )
    elapsed = time.perf_counter() - start

    print(f"✅ Calibration complete in {elapsed*1000:.2f}ms")

    print("\n   Feature Ranges (for quantization):")
    feature_names = [
        "Buy/Sell Imbalance",
        "Volume Delta",
        "Trade Intensity",
        "Price Velocity",
        "Volume Velocity",
        "Cumulative Volume Delta"
    ]

    for i, name in enumerate(feature_names):
        min_val = ranges[i * 2]
        max_val = ranges[i * 2 + 1]
        print(f"     {i}. {name:25s}: [{min_val:12.4f}, {max_val:12.4f}]")

    print("\nStep 4: Create custom strategy with calibrated ranges...")
    feature_mins = [ranges[i] for i in range(0, 12, 2)]
    feature_maxs = [ranges[i] for i in range(1, 12, 2)]

    custom_strategy = kimsfinance_core.StrategyConfig(
        "momentum", feature_mins, feature_maxs
    )

    print(f"✅ Created custom strategy: {custom_strategy}")

    print("\nStep 5: Process with custom strategy...")
    result = processor.process_batch(
        timestamps, close_prices, volumes, buy_volumes, sell_volumes,
        [custom_strategy]
    )

    signals = result.signals[0]
    buy_count = np.sum(signals == 1)
    sell_count = np.sum(signals == -1)

    print(f"✅ Generated {buy_count} buy signals and {sell_count} sell signals")
    print("\n✅ Example 3 complete!")


# ============================================================================
# EXAMPLE 4: CPU Fallback (Graceful Degradation)
# ============================================================================
def example_4_cpu_fallback():
    """Demonstrate graceful handling when GPU is unavailable."""
    print("\n📖 EXAMPLE 4: CPU Fallback (Graceful Degradation)")
    print("-" * 70)

    print("Step 1: Check GPU availability...")
    gpu_available = kimsfinance_core.orderflow_gpu_available()

    if gpu_available:
        print("✅ GPU is available (this example demonstrates fallback behavior)")
        print("   To test fallback, rebuild without GPU feature:")
        print("   cargo build --release --features python")
    else:
        print("⚠️  GPU not available - demonstrating CPU fallback")

    print("\nStep 2: Attempt processor initialization...")
    try:
        processor = kimsfinance_core.OrderflowProcessor()
        print(f"✅ Processor initialized: {processor}")
        print("   Using GPU acceleration")
    except RuntimeError as e:
        print(f"❌ GPU initialization failed: {e}")
        print("\n💡 Fallback options:")
        print("   1. Use CPU-based orderflow analysis (implement in Python)")
        print("   2. Process in smaller batches")
        print("   3. Use fewer strategies")
        print("\n   Example CPU implementation:")
        print("   def calculate_orderflow_features_cpu(prices, volumes, buy_vols, sell_vols):")
        print("       # Calculate imbalance")
        print("       imbalance = buy_vols / (buy_vols + sell_vols)")
        print("       # Calculate volume delta")
        print("       volume_delta = buy_vols - sell_vols")
        print("       # ... calculate other features")
        print("       return features")

    print("\n✅ Example 4 complete!")


# ============================================================================
# EXAMPLE 5: Integration with Backtesting
# ============================================================================
def example_5_backtesting_integration():
    """Demonstrate using orderflow signals for backtesting."""
    print("\n📖 EXAMPLE 5: Integration with Backtesting")
    print("-" * 70)

    if not kimsfinance_core.orderflow_gpu_available():
        print("⚠️  GPU not available! Skipping this example.")
        return

    print("Step 1: Generate realistic market data...")
    n = 50_000
    timestamps = np.arange(n, dtype=np.int64) * 1000  # 1 second apart

    # Simulate realistic price action
    price = 50000.0
    prices = []
    for i in range(n):
        # Add trend + noise
        price += np.random.randn() * 20 + 0.01  # Slight upward bias
        prices.append(price)

    close_prices = np.array(prices, dtype=np.float32)
    volumes = np.random.exponential(100.0, n).astype(np.float32)
    imbalance = np.random.uniform(0.3, 0.7, n).astype(np.float32)
    buy_volumes = (volumes * imbalance).astype(np.float32)
    sell_volumes = (volumes - buy_volumes).astype(np.float32)

    print(f"✅ Generated {n:,} ticks")
    print(f"   Initial price: ${prices[0]:.2f}")
    print(f"   Final price: ${prices[-1]:.2f}")
    print(f"   Return: {(prices[-1] / prices[0] - 1) * 100:.2f}%")

    print("\nStep 2: Generate orderflow signals...")
    processor = kimsfinance_core.OrderflowProcessor()

    strategies = [
        kimsfinance_core.StrategyConfig.momentum(),
        kimsfinance_core.StrategyConfig.mean_reversion(),
    ]

    result = processor.process_batch(
        timestamps, close_prices, volumes, buy_volumes, sell_volumes, strategies
    )

    print(f"✅ Generated signals for {len(strategies)} strategies")

    print("\nStep 3: Simulate simple backtest...")
    # Simple backtest: trade on momentum signals
    signals = result.signals[0]  # Momentum strategy

    initial_capital = 10000.0
    position = 0.0  # BTC position
    cash = initial_capital

    # Track equity
    equity_curve = []

    for i in range(n):
        # Current equity
        equity = cash + position * close_prices[i]
        equity_curve.append(equity)

        # Execute signals
        if signals[i] == 1 and position == 0:  # Buy signal
            # Buy with 100% of cash (simplified)
            position = cash / close_prices[i]
            cash = 0
        elif signals[i] == -1 and position > 0:  # Sell signal
            # Sell all
            cash = position * close_prices[i]
            position = 0

    # Final equity
    final_equity = cash + position * close_prices[-1]
    equity_curve.append(final_equity)

    total_return = (final_equity / initial_capital - 1) * 100

    print(f"✅ Backtest complete")
    print(f"\n   Results:")
    print(f"     Initial Capital: ${initial_capital:,.2f}")
    print(f"     Final Equity:    ${final_equity:,.2f}")
    print(f"     Total Return:    {total_return:+.2f}%")

    # Calculate max drawdown
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    print(f"     Max Drawdown:    {max_drawdown:.2f}%")

    # Count trades
    position_changes = np.diff(signals != 0).astype(int)
    num_trades = np.sum(position_changes == 1)

    print(f"     Number of Trades: {num_trades}")

    print("\n✅ Example 5 complete!")


# ============================================================================
# EXAMPLE 6: Performance Benchmark
# ============================================================================
def example_6_performance_benchmark():
    """Benchmark orderflow processing performance."""
    print("\n📖 EXAMPLE 6: Performance Benchmark")
    print("-" * 70)

    if not kimsfinance_core.orderflow_gpu_available():
        print("⚠️  GPU not available! Skipping this example.")
        return

    print("Initializing processor...")
    processor = kimsfinance_core.OrderflowProcessor()

    # Test different sizes
    test_sizes = [1_000, 10_000, 100_000, 1_000_000]
    num_strategies = 10

    print(f"\nBenchmarking with {num_strategies} strategies...")
    print("\n   Size      | Time (ms) | Throughput (M ticks/sec) | Signals/sec")
    print("   " + "-" * 68)

    strategies = [
        kimsfinance_core.StrategyConfig.momentum(),
        kimsfinance_core.StrategyConfig.mean_reversion(),
        kimsfinance_core.StrategyConfig.breakout(),
        kimsfinance_core.StrategyConfig.scalping(),
        kimsfinance_core.StrategyConfig.trend_following(),
    ] * 2  # 10 strategies total

    for n in test_sizes:
        # Generate data
        timestamps = np.arange(n, dtype=np.int64) * 100
        close_prices = (50000.0 + np.random.randn(n).cumsum() * 10).astype(np.float32)
        volumes = np.random.exponential(100.0, n).astype(np.float32)
        imbalance = np.random.uniform(0.4, 0.6, n).astype(np.float32)
        buy_volumes = (volumes * imbalance).astype(np.float32)
        sell_volumes = (volumes - buy_volumes).astype(np.float32)

        # Warmup
        _ = processor.process_batch(
            timestamps, close_prices, volumes, buy_volumes, sell_volumes, strategies
        )

        # Benchmark (3 runs)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = processor.process_batch(
                timestamps, close_prices, volumes, buy_volumes, sell_volumes, strategies
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times) * 1000  # ms
        throughput = n / (avg_time / 1000) / 1e6  # M ticks/sec
        signals_per_sec = (n * num_strategies) / (avg_time / 1000)

        print(f"   {n:9,} | {avg_time:9.2f} | {throughput:24.2f} | {signals_per_sec:11.2e}")

    print("\n✅ Benchmark complete!")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Run all examples."""
    print("\nThis script demonstrates 6 different use cases:")
    print("  1. Basic Usage - Single Strategy")
    print("  2. Multiple Strategies in Parallel")
    print("  3. Feature Calibration")
    print("  4. CPU Fallback (Graceful Degradation)")
    print("  5. Integration with Backtesting")
    print("  6. Performance Benchmark")
    print()

    try:
        example_1_basic_usage()
        example_2_multiple_strategies()
        example_3_calibration()
        example_4_cpu_fallback()
        example_5_backtesting_integration()
        example_6_performance_benchmark()

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)

        print("\n💡 Next Steps:")
        print("   1. Integrate with real tick data from exchanges")
        print("   2. Implement custom strategy logic")
        print("   3. Combine with tick-level backtesting (TickBacktestEngine)")
        print("   4. Use features for ML model training")
        print("   5. Profile with Nsight Systems: nsys profile python3 python_orderflow_example.py")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
