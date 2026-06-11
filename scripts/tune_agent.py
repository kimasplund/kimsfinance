import os
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
import time

INSTRUMENTS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT']
DATA_DIR = "/home/kim/projects/kimsfinance/data/binance"

# Load data into memory once to share across processes
dfs = {}
for ticker in INSTRUMENTS:
    path = f"{DATA_DIR}/{ticker}_5m_1y.parquet"
    if os.path.exists(path):
        dfs[ticker] = pd.read_parquet(path)

# Gather timestamps
all_timestamps = sorted(list(set.union(*[set(df.index) for df in dfs.values()])))

# Pre-calculate indices for fast lookups
df_indices = {ticker: {time: i for i, time in enumerate(dfs[ticker].index)} for ticker in dfs}

def run_simulation(params):
    target1_r, target2_r, daily_limit_r, ema_trend_period = params
    
    # Pre-calculate EMA trend filter
    local_dfs = {}
    for ticker, df in dfs.items():
        temp_df = df.copy()
        temp_df['EMA_9'] = temp_df['Close'].ewm(span=9, adjust=False).mean()
        temp_df['EMA_20'] = temp_df['Close'].ewm(span=20, adjust=False).mean()
        temp_df['EMA_50'] = temp_df['Close'].ewm(span=50, adjust=False).mean()
        temp_df['EMA_Trend'] = temp_df['Close'].ewm(span=ema_trend_period, adjust=False).mean()
        temp_df['EMA_Trend_Slope'] = temp_df['EMA_Trend'].diff(3)
        local_dfs[ticker] = temp_df
        
    open_positions = {ticker: None for ticker in INSTRUMENTS}
    trades = []
    
    current_day = None
    daily_pnl_R = 0.0
    daily_limit_hit = False
    
    INITIAL_CAPITAL = 10000.0
    RISK_PER_R = 100.0
    current_capital = INITIAL_CAPITAL

    for current_time in all_timestamps:
        day = current_time.date()
        if current_day != day:
            current_day = day
            daily_pnl_R = 0.0
            daily_limit_hit = False
            
        for ticker, df in local_dfs.items():
            if current_time not in df_indices[ticker]:
                continue
                
            i = df_indices[ticker][current_time]
            if i < 3:
                continue
                
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            prev_prev_row = df.iloc[i-2]
            
            pos = open_positions[ticker]
            
            # --- MANAGE OPEN POSITION ---
            if pos is not None:
                high = row['High']
                low = row['Low']
                
                # Check Stop Loss
                if low <= pos['stop_loss']:
                    if pos['size'] == 1.0:
                        r_realized = -1.0
                    else:
                        r_realized = 0.0 
                        
                    daily_pnl_R += r_realized
                    current_capital += r_realized * RISK_PER_R
                    trades.append(r_realized)
                    open_positions[ticker] = None
                    
                    if daily_pnl_R <= -daily_limit_r:
                        daily_limit_hit = True
                        
                # Check Target 2
                elif high >= pos['target2'] and pos['size'] == 0.5:
                    r_realized = target2_r * 0.5
                    daily_pnl_R += r_realized
                    current_capital += r_realized * RISK_PER_R
                    trades.append(r_realized)
                    open_positions[ticker] = None
                    
                # Check Target 1
                elif high >= pos['target1'] and pos['size'] == 1.0:
                    r_realized = target1_r * 0.5
                    daily_pnl_R += r_realized
                    current_capital += r_realized * RISK_PER_R
                    pos['size'] = 0.5
                    pos['stop_loss'] = pos['entry_price']
                continue

            # --- FIND NEW SETUPS ---
            if daily_limit_hit:
                continue

            # Trend Filter
            is_trending_up = prev_row['EMA_Trend_Slope'] > 0 and prev_row['EMA_Trend'] > prev_row['EMA_50']
            if not is_trending_up:
                continue

            # Crossover Setup
            if prev_prev_row['EMA_9'] <= prev_prev_row['EMA_20'] and prev_row['EMA_9'] > prev_row['EMA_20']:
                entry_price = row['Open']
                stop_loss = min(prev_prev_row['Low'], prev_row['Low'], row['Low'])
                risk = entry_price - stop_loss
                if risk > 0:
                    open_positions[ticker] = {
                        'entry_price': entry_price, 'stop_loss': stop_loss,
                        'target1': entry_price + target1_r * risk,
                        'target2': entry_price + target2_r * risk,
                        'size': 1.0
                    }
                continue

            # Bounce Setup
            if prev_row['High'] > prev_row['EMA_20'] and prev_row['Low'] <= prev_row['EMA_20']:
                if prev_row['Close'] > prev_row['Open']:
                    entry_price = row['Open']
                    stop_loss = prev_row['Low'] - 0.01
                    risk = entry_price - stop_loss
                    if risk > 0:
                        open_positions[ticker] = {
                            'entry_price': entry_price, 'stop_loss': stop_loss,
                            'target1': entry_price + target1_r * risk,
                            'target2': entry_price + target2_r * risk,
                            'size': 1.0
                        }

    total_r = sum(trades)
    return {
        'params': params,
        'trades': len(trades),
        'total_r': total_r,
        'final_capital': current_capital
    }

if __name__ == "__main__":
    print("Starting parameter tuning grid search...")
    
    # Grid definition
    target1_range = [1.0, 1.5, 2.0]
    target2_range = [2.0, 2.5, 3.0]
    daily_limit_range = [2.0, 3.0, 4.0]
    ema_trend_range = [20, 30] # 20 is default, 30 is smoother
    
    grid = list(itertools.product(target1_range, target2_range, daily_limit_range, ema_trend_range))
    print(f"Total combinations to test: {len(grid)}")
    
    start_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_simulation, grid)
        
    results.sort(key=lambda x: x['total_r'], reverse=True)
    
    best = results[0]
    print(f"Optimization complete in {time.time() - start_time:.2f} seconds")
    print(f"Best Parameters:")
    print(f"  Target 1 (Partial): {best['params'][0]}R")
    print(f"  Target 2 (Runner): {best['params'][1]}R")
    print(f"  Daily Limit: -{best['params'][2]}R")
    print(f"  Trend EMA: {best['params'][3]}")
    print(f"Best Performance: {best['total_r']:.2f}R (+${best['final_capital'] - 10000:.2f}) over {best['trades']} trades.")
