import requests
import pandas as pd
import time
import os

RESULTS_DIR = "/home/kim/projects/kimsfinance/research"

def get_binance_funding_rates(symbol="BTCUSDT", limit=1000):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_rates = []
    
    # We want to go back in time, Binance API returns max 1000 items from startTime
    # We will just fetch the most recent 1000 to get a solid 333 days (almost 1 year)
    # Since funding is 3 times a day (every 8 hours), 1000 data points = 333 days.
    
    print(f"Fetching last 1000 Funding Rates for {symbol}...")
    params = {
        'symbol': symbol,
        'limit': limit
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        for item in data:
            all_rates.append({
                'timestamp': pd.to_datetime(item['fundingTime'], unit='ms'),
                'funding_rate': float(item['fundingRate'])
            })
    else:
        print(f"Error fetching data: {response.text}")
        
    df = pd.DataFrame(all_rates)
    return df

def simulate_delta_neutral(df):
    print("\nSimulating Delta Neutral Cash and Carry...")
    
    # Starting Capital: $10,000
    # Allocation: $10,000 Spot BTC (Long), $10,000 Futures BTC (Short) using 1x Leverage on Futures
    # Total Capital Required: $10k + $10k (if purely un-leveraged cash and carry)
    # Actually, you can use Spot BTC as collateral for Coin-M futures, so you only need $10,000 total capital!
    
    capital = 10000.0
    
    # For every funding interval, if rate is positive (Longs pay Shorts), we RECEIVE the rate on our Short position.
    # If rate is negative (Shorts pay Longs), we PAY the rate on our Short position.
    
    df['Yield_Amount'] = capital * df['funding_rate']
    
    total_yield = df['Yield_Amount'].sum()
    days = len(df) / 3
    
    annualized_yield = (total_yield / capital) * (365 / days) * 100 if days > 0 else 0
    
    print("\n" + "="*50)
    print("FUNDING RATE ARBITRAGE (CASH & CARRY) RESULTS")
    print("="*50)
    print(f"Days Simulated: {days:.1f}")
    print(f"Total Funding Intervals: {len(df)}")
    print(f"Gross Yield Collected: ${total_yield:,.2f}")
    print(f"Annualized APY: {annualized_yield:.2f}% (Completely Risk-Free to Price)")
    print("="*50)

def main():
    df = get_binance_funding_rates()
    if not df.empty:
        simulate_delta_neutral(df)

if __name__ == "__main__":
    main()
