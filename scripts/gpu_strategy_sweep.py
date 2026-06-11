import yfinance as yf
import pandas as pd
import json
from kimsfinance.batch import find_best_parameters, BacktestConfig, get_gpu_info

def main():
    info = get_gpu_info()
    if not info.get('gpu_available'):
        print("GPU not available. Falling back to CPU, but kimsfinance Rust core will still execute extremely fast.")
    else:
        print(f"GPU Engine Initialized! Expected Speedup: {info.get('expected_speedup')}x")

    print("Downloading 1-Hour BTC data (2 Years)...")
    # Download data for BTC to optimize the benchmark
    df = yf.download("BTC-USD", period="730d", interval="1h", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Kimsfinance core expects lowercase columns
    df.columns = [col.lower() for col in df.columns]

    # Clean data
    df = df.dropna()

    # Configure the GPU Backtester with exact Binance VIP 0 + BNB Taker Fees
    config = BacktestConfig(
        initial_capital=10000.0,
        trading_fee=0.00045, # Taker fee
        slippage=0.0001
    )

    # Define the parameter combinations to sweep on the GPU
    # This will generate 4 * 1 * 3 * 3 = 36 combinations
    parameter_ranges = {
        'period': [10.0, 20.0, 30.0, 40.0, 50.0],
        'std_dev': [2.0], # Used for internal calculations
        'entry_std': [2.0, 2.5, 3.0, 3.5], # How overextended the price must be to enter
        'exit_std': [0.0, 0.5, -0.5] # Where to take profit (0.0 is exactly the mean)
    }

    print("Executing thousands of combinations on the GPU Core...")
    
    # We optimize for 'sharpe_ratio' as it balances win rate, risk, and total return
    results = find_best_parameters(
        strategy='bollinger',
        data=df,
        parameter_ranges=parameter_ranges,
        config=config,
        objective='sharpe_ratio'
    )

    best_params = results['best_params']
    best_score = results['best_score']
    best_run = next(r for r in results['all_results'] if r['params'] == best_params)

    print("\n" + "="*50)
    print("GPU SWEEP COMPLETE - BEST CONFIGURATION FOUND")
    print("="*50)
    print(f"Best Parameters: {json.dumps(best_params, indent=2)}")
    print(f"Sharpe Ratio: {best_score:.4f}")
    print(f"Win Rate: {best_run['win_rate']*100:.2f}%")
    print(f"Total Trades: {best_run['num_trades']}")
    print(f"Final Equity: ${best_run['final_equity']:,.2f}")
    print(f"Net Profit: ${best_run['final_equity'] - 10000.0:,.2f}")
    print("="*50)
    
    # Save results to research folder
    with open("/home/kim/projects/kimsfinance/research/gpu_sweep_results.json", "w") as f:
        json.dump(results['all_results'], f, indent=4)

if __name__ == "__main__":
    main()
