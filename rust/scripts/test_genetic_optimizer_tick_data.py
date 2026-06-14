#!/usr/bin/env python3
"""
Test and benchmark genetic optimizer with tick-level Parquet data

Compares:
1. Traditional OHLCV aggregated data (existing approach)
2. Tick-level data from Parquet files (new capability)

Measures:
- Backtest accuracy (tick vs OHLCV differences)
- Performance (trades/sec processing speed)
- Optimization quality (better parameter discovery)
"""

import polars as pl
import time
from pathlib import Path
from datetime import datetime, timedelta
import json

# Paths
TICK_DATA_BASE = Path("/home/kim/projects/binance-data/futures")
OUTPUT_DIR = Path("/tmp/genetic_optimizer_tick_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)

class SimpleMovingAverageCrossStrategy:
    """Simple MA crossover strategy for testing"""

    def __init__(self, fast_period: int, slow_period: int):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position = 0
        self.equity = 10000.0
        self.trades = []
        self.price_history = []

    def on_tick(self, price: float, qty: float, side: str, timestamp):
        """Process each tick"""
        self.price_history.append(price)

        # Wait for enough data
        if len(self.price_history) < self.slow_period:
            return

        # Calculate MAs from recent prices
        fast_ma = sum(self.price_history[-self.fast_period:]) / self.fast_period
        slow_ma = sum(self.price_history[-self.slow_period:]) / self.slow_period

        # Generate signals
        if fast_ma > slow_ma and self.position == 0:
            # Buy signal
            self.position = self.equity / price
            self.trades.append({
                'type': 'buy',
                'price': price,
                'time': timestamp,
                'equity': self.equity
            })
        elif fast_ma < slow_ma and self.position > 0:
            # Sell signal
            self.equity = self.position * price
            self.trades.append({
                'type': 'sell',
                'price': price,
                'time': timestamp,
                'equity': self.equity
            })
            self.position = 0

    def finalize(self, final_price: float):
        """Close any open position"""
        if self.position > 0:
            self.equity = self.position * final_price
            self.position = 0
        return self.equity

    def get_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'final_equity': self.equity,
                'return_pct': 0.0,
                'win_rate': 0.0
            }

        # Calculate win rate
        wins = 0
        for i in range(1, len(self.trades)):
            if self.trades[i]['type'] == 'sell':
                prev_buy = None
                for j in range(i-1, -1, -1):
                    if self.trades[j]['type'] == 'buy':
                        prev_buy = self.trades[j]
                        break
                if prev_buy and self.trades[i]['price'] > prev_buy['price']:
                    wins += 1

        total_complete_trades = len([t for t in self.trades if t['type'] == 'sell'])
        win_rate = (wins / total_complete_trades * 100) if total_complete_trades > 0 else 0

        return {
            'total_trades': len(self.trades),
            'final_equity': self.equity,
            'return_pct': ((self.equity - 10000) / 10000) * 100,
            'win_rate': win_rate
        }


def load_tick_data_month(pair: str, month: str, max_ticks: int = None):
    """Load tick data from Parquet for a specific month"""
    print(f"Loading {pair} tick data for {month}...")

    data_dir = TICK_DATA_BASE / pair / "trades_parquet" / month
    if not data_dir.exists():
        raise FileNotFoundError(f"No data for {pair} {month}")

    # Load all parquet files for the month
    df = pl.read_parquet(f"{data_dir}/*.parquet")

    if max_ticks:
        df = df.head(max_ticks)

    print(f"  Loaded {len(df):,} ticks")
    return df


def backtest_tick_data(df: pl.DataFrame, fast_period: int, slow_period: int):
    """Run backtest on tick-level data"""
    strategy = SimpleMovingAverageCrossStrategy(fast_period, slow_period)

    start_time = time.time()

    # Process each tick
    for row in df.iter_rows(named=True):
        strategy.on_tick(
            price=row['price'],
            qty=row['qty'],
            side=row['side'],
            timestamp=row['timestamp']
        )

    # Finalize with last price
    final_price = df['price'][-1]
    final_equity = strategy.finalize(final_price)

    elapsed = time.time() - start_time
    ticks_per_sec = len(df) / elapsed if elapsed > 0 else 0

    metrics = strategy.get_metrics()
    metrics['processing_time'] = elapsed
    metrics['ticks_per_sec'] = ticks_per_sec
    metrics['total_ticks'] = len(df)

    return metrics


def aggregate_to_ohlcv(df: pl.DataFrame, timeframe: str = "1m"):
    """Aggregate tick data to OHLCV candles"""
    print(f"Aggregating to {timeframe} OHLCV...")

    # Group by minute
    ohlcv = df.group_by_dynamic(
        "timestamp",
        every=timeframe,
    ).agg([
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("qty").sum().alias("volume"),
    ]).sort("timestamp")

    print(f"  Aggregated to {len(ohlcv):,} candles")
    return ohlcv


def backtest_ohlcv_data(df: pl.DataFrame, fast_period: int, slow_period: int):
    """Run backtest on OHLCV aggregated data (simpler version)"""
    # Convert to list for easier processing
    data = df.select(['close']).to_series().to_list()

    position = 0
    equity = 10000.0
    trades = []

    start_time = time.time()

    for i in range(max(fast_period, slow_period), len(data)):
        fast_ma = sum(data[i-fast_period:i]) / fast_period
        slow_ma = sum(data[i-slow_period:i]) / slow_period
        price = data[i]

        if fast_ma > slow_ma and position == 0:
            position = equity / price
            trades.append({'type': 'buy', 'price': price})
        elif fast_ma < slow_ma and position > 0:
            equity = position * price
            trades.append({'type': 'sell', 'price': price})
            position = 0

    # Close position
    if position > 0:
        equity = position * data[-1]
        position = 0

    elapsed = time.time() - start_time
    candles_per_sec = len(data) / elapsed if elapsed > 0 else 0

    return {
        'total_trades': len(trades),
        'final_equity': equity,
        'return_pct': ((equity - 10000) / 10000) * 100,
        'processing_time': elapsed,
        'candles_per_sec': candles_per_sec,
        'total_candles': len(data)
    }


def run_genetic_optimization_tick(df: pl.DataFrame, generations: int = 20, population: int = 50):
    """Run simple genetic algorithm to find best MA parameters using tick data"""
    import random

    print(f"\nRunning genetic optimization (tick-level)...")
    print(f"  Generations: {generations}")
    print(f"  Population: {population}")

    # Initialize population (fast_period, slow_period)
    population_list = []
    for _ in range(population):
        fast = random.randint(5, 50)
        slow = random.randint(fast + 5, 200)
        population_list.append((fast, slow))

    best_overall = None
    best_fitness = -float('inf')

    start_time = time.time()

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for fast, slow in population_list:
            result = backtest_tick_data(df, fast, slow)
            fitness = result['return_pct']  # Use return as fitness
            fitness_scores.append((fitness, (fast, slow), result))

        # Sort by fitness
        fitness_scores.sort(reverse=True, key=lambda x: x[0])

        # Track best
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_overall = fitness_scores[0][1:]

        print(f"  Gen {gen+1}/{generations}: Best return = {fitness_scores[0][0]:.2f}% "
              f"(MA {fitness_scores[0][1][0]}/{fitness_scores[0][1][1]})")

        # Selection (keep top 50%)
        elite = [params for _, params, _ in fitness_scores[:population//2]]

        # Crossover and mutation
        new_population = elite.copy()
        while len(new_population) < population:
            parent1, parent2 = random.sample(elite, 2)
            # Crossover
            child_fast = random.choice([parent1[0], parent2[0]])
            child_slow = random.choice([parent1[1], parent2[1]])
            # Mutation (20% chance)
            if random.random() < 0.2:
                child_fast += random.randint(-5, 5)
                child_fast = max(5, min(50, child_fast))
            if random.random() < 0.2:
                child_slow += random.randint(-10, 10)
                child_slow = max(child_fast + 5, min(200, child_slow))
            new_population.append((child_fast, child_slow))

        population_list = new_population

    elapsed = time.time() - start_time

    return {
        'best_params': best_overall[0],
        'best_result': best_overall[1],
        'optimization_time': elapsed,
        'generations': generations,
        'population': population
    }


def main():
    """Main benchmark"""
    print("=" * 80)
    print("GENETIC OPTIMIZER TICK-LEVEL DATA BENCHMARK")
    print("=" * 80)
    print()

    # Configuration
    PAIR = "BTCUSDT"
    MONTH = "2024-01"  # January 2024
    MAX_TICKS = 1_000_000  # 1M ticks for testing (about 1 day of data)

    # Load tick data
    print(f"\n1. Loading tick data ({PAIR} {MONTH})...")
    tick_df = load_tick_data_month(PAIR, MONTH, max_ticks=MAX_TICKS)

    # Test 1: Simple backtest with tick data
    print(f"\n2. Testing tick-level backtest (MA 10/30)...")
    tick_result = backtest_tick_data(tick_df, fast_period=10, slow_period=30)
    print(f"  Return: {tick_result['return_pct']:.2f}%")
    print(f"  Trades: {tick_result['total_trades']}")
    print(f"  Speed: {tick_result['ticks_per_sec']:,.0f} ticks/sec")

    # Test 2: Aggregate to OHLCV and backtest
    print(f"\n3. Testing OHLCV aggregated backtest (MA 10/30)...")
    ohlcv_df = aggregate_to_ohlcv(tick_df, timeframe="1m")
    ohlcv_result = backtest_ohlcv_data(ohlcv_df, fast_period=10, slow_period=30)
    print(f"  Return: {ohlcv_result['return_pct']:.2f}%")
    print(f"  Trades: {ohlcv_result['total_trades']}")
    print(f"  Speed: {ohlcv_result['candles_per_sec']:,.0f} candles/sec")

    # Test 3: Genetic optimization with tick data
    print(f"\n4. Running genetic optimization with tick data...")
    optimization_result = run_genetic_optimization_tick(
        tick_df.head(100_000),  # Use subset for faster optimization
        generations=10,
        population=20
    )
    print(f"\n  Best parameters found: MA {optimization_result['best_params'][0]}/{optimization_result['best_params'][1]}")
    print(f"  Best return: {optimization_result['best_result']['return_pct']:.2f}%")
    print(f"  Optimization time: {optimization_result['optimization_time']:.1f}s")

    # Summary comparison
    print(f"\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\nTick-Level (MA 10/30):")
    print(f"  Return: {tick_result['return_pct']:.2f}%")
    print(f"  Processing: {tick_result['ticks_per_sec']:,.0f} ticks/sec")
    print(f"  Data points: {tick_result['total_ticks']:,}")

    print(f"\nOHLCV Aggregated (MA 10/30):")
    print(f"  Return: {ohlcv_result['return_pct']:.2f}%")
    print(f"  Processing: {ohlcv_result['candles_per_sec']:,.0f} candles/sec")
    print(f"  Data points: {ohlcv_result['total_candles']:,}")

    print(f"\nDifference:")
    print(f"  Return delta: {abs(tick_result['return_pct'] - ohlcv_result['return_pct']):.2f}%")
    print(f"  Data granularity: {tick_result['total_ticks'] / ohlcv_result['total_candles']:.0f}x more tick data")

    print(f"\nGenetic Optimization:")
    print(f"  Best strategy: MA {optimization_result['best_params'][0]}/{optimization_result['best_params'][1]}")
    print(f"  Optimized return: {optimization_result['best_result']['return_pct']:.2f}%")
    print(f"  Improvement: {optimization_result['best_result']['return_pct'] - tick_result['return_pct']:.2f}%")

    # Save results
    results_file = OUTPUT_DIR / f"tick_benchmark_{PAIR}_{MONTH}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'pair': PAIR,
            'month': MONTH,
            'max_ticks': MAX_TICKS,
            'tick_result': tick_result,
            'ohlcv_result': ohlcv_result,
            'optimization': {
                'best_params': optimization_result['best_params'],
                'best_return': optimization_result['best_result']['return_pct'],
                'optimization_time': optimization_result['optimization_time']
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {results_file}")
    print(f"\n🚀 Tick-level genetic optimization is working!")


if __name__ == "__main__":
    main()
