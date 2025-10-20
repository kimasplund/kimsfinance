import timeit
import numpy as np
from kimsfinance.plotting.renderer import render_ohlcv_chart


def generate_data(num_candles):
    """Generate sample OHLCV data."""
    ohlc = {
        "open": np.random.uniform(90, 100, num_candles),
        "high": np.random.uniform(100, 110, num_candles),
        "low": np.random.uniform(80, 90, num_candles),
        "close": np.random.uniform(90, 100, num_candles),
    }
    volume = np.random.uniform(1000, 5000, num_candles)
    return {"ohlc": ohlc, "volume": volume}


def run_benchmark(use_batch_drawing, data, num_runs=10):
    """Run benchmark for a given rendering mode."""
    stmt = lambda: render_ohlcv_chart(
        ohlc=data["ohlc"], volume=data["volume"], use_batch_drawing=use_batch_drawing
    )
    times = timeit.repeat(stmt, repeat=3, number=num_runs)
    return min(times) / num_runs


def main():
    """Main benchmark function."""
    for num_candles in [1000, 10000, 50000]:
        data = generate_data(num_candles)

        print(f"--- Benchmarking Chart Rendering ({num_candles} candles) ---")

        # Benchmark sequential drawing
        sequential_time = run_benchmark(False, data)
        print(f"Sequential Drawing: {sequential_time:.6f} seconds per chart")

        # Benchmark batch drawing
        batch_time = run_benchmark(True, data)
        print(f"Batch Drawing:      {batch_time:.6f} seconds per chart")

        # Compare results
        if batch_time > 0:
            speedup = sequential_time / batch_time
            print(
                f"\nBatch drawing is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than sequential drawing."
            )
        else:
            print("\nBatch drawing is significantly faster.")
        print("-" * 40)


if __name__ == "__main__":
    main()
