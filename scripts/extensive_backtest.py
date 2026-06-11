import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import multiprocessing
import itertools

INSTRUMENTS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD']
RESULTS_DIR = "/home/kim/projects/kimsfinance/research"

# Realistic Parameters
INITIAL_CAPITAL = 10000.0
RISK_PER_R = 100.0  # 1% of $10k
TAKER_FEE = 0.00045 # Binance Futures BNB Discount Taker
MAKER_FEE = 0.00018 # Binance Futures BNB Discount Maker
MAX_LEVERAGE = 10.0 # Standard leverage

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def download_data(timeframe):
    dfs = {}
    period = "730d"
    for ticker in INSTRUMENTS:
        try:
            df = yf.download(ticker, period=period, interval=timeframe, progress=False)
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [col.capitalize() for col in df.columns]
            
            df['EMA_9'] = calculate_ema(df['Close'], 9)
            df['EMA_20'] = calculate_ema(df['Close'], 20)
            df['EMA_50'] = calculate_ema(df['Close'], 50)
            df['EMA_20_Slope'] = df['EMA_20'].diff(3)
            
            dfs[ticker] = df
        except Exception as e:
            pass
    return dfs

def run_simulation(args):
    dfs, tf, t1_r, t2_r = args
    if not dfs: return {}

    all_timestamps = sorted(list(set.union(*[set(df.index) for df in dfs.values()])))
    df_indices = {ticker: {time: i for i, time in enumerate(dfs[ticker].index)} for ticker in dfs}

    open_positions = {ticker: None for ticker in INSTRUMENTS}
    trades = []
    
    current_day = None
    daily_pnl_R = 0.0
    daily_limit_hit = False
    current_capital = INITIAL_CAPITAL
    
    gross_pnl_total = 0.0
    fees_total = 0.0

    for current_time in all_timestamps:
        day = current_time.date()
        if current_day != day:
            current_day = day
            daily_pnl_R = 0.0
            daily_limit_hit = False
            
        for ticker, df in dfs.items():
            if current_time not in df_indices[ticker]: continue
            i = df_indices[ticker][current_time]
            if i < 3: continue
            
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            prev_prev_row = df.iloc[i-2]
            
            pos = open_positions[ticker]
            
            # --- MANAGE OPEN POSITION ---
            if pos is not None:
                high = row['High']
                low = row['Low']
                
                # Stop Loss
                if low <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    qty_exiting = pos['qty'] * pos['size']
                    
                    exit_fee = (qty_exiting * exit_price) * TAKER_FEE
                    pnl = (exit_price - pos['entry_price']) * qty_exiting
                    
                    r_realized = -1.0 if pos['size'] == 1.0 else 0.0
                    result = 'Loss' if pos['size'] == 1.0 else 'Win'
                    
                    daily_pnl_R += r_realized
                    net_pnl = pnl - exit_fee
                    current_capital += net_pnl
                    
                    gross_pnl_total += pnl
                    fees_total += exit_fee + pos['entry_fee_prorated'] * pos['size']
                    
                    trades.append({'Result': result, 'Net_PnL': net_pnl})
                    open_positions[ticker] = None
                    if daily_pnl_R <= -3.0: daily_limit_hit = True
                        
                # Target 2
                elif high >= pos['target2'] and pos['size'] == 0.5:
                    exit_price = pos['target2']
                    qty_exiting = pos['qty'] * pos['size']
                    
                    exit_fee = (qty_exiting * exit_price) * MAKER_FEE
                    pnl = (exit_price - pos['entry_price']) * qty_exiting
                    
                    r_realized = t2_r * 0.5
                    daily_pnl_R += r_realized
                    net_pnl = pnl - exit_fee
                    current_capital += net_pnl
                    
                    gross_pnl_total += pnl
                    fees_total += exit_fee + pos['entry_fee_prorated'] * 0.5
                    
                    trades.append({'Result': 'Win', 'Net_PnL': net_pnl})
                    open_positions[ticker] = None
                    
                # Target 1
                elif high >= pos['target1'] and pos['size'] == 1.0:
                    exit_price = pos['target1']
                    qty_exiting = pos['qty'] * 0.5
                    
                    exit_fee = (qty_exiting * exit_price) * MAKER_FEE
                    pnl = (exit_price - pos['entry_price']) * qty_exiting
                    
                    r_realized = t1_r * 0.5
                    daily_pnl_R += r_realized
                    
                    gross_pnl_total += pnl
                    fees_total += exit_fee + pos['entry_fee_prorated'] * 0.5
                    
                    pos['size'] = 0.5
                    pos['stop_loss'] = pos['entry_price']
                continue

            # --- FIND NEW SETUPS ---
            if daily_limit_hit: continue

            # Trend Filter
            is_trending_up = prev_row['EMA_20_Slope'] > 0 and prev_row['EMA_20'] > prev_row['EMA_50']
            if not is_trending_up: continue

            if prev_prev_row['EMA_9'] <= prev_prev_row['EMA_20'] and prev_row['EMA_9'] > prev_row['EMA_20']:
                entry_price = row['Open']
                stop_loss = min(prev_prev_row['Low'], prev_row['Low'], row['Low'])
                
                risk = entry_price - stop_loss
                if risk > 0:
                    ideal_qty = RISK_PER_R / risk
                    max_qty = (current_capital * MAX_LEVERAGE) / entry_price
                    qty = min(ideal_qty, max_qty)
                    
                    entry_fee = (qty * entry_price) * TAKER_FEE
                    
                    open_positions[ticker] = {
                        'qty': qty,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'risk': risk,
                        'target1': entry_price + (t1_r * risk),
                        'target2': entry_price + (t2_r * risk),
                        'size': 1.0,
                        'entry_fee_prorated': entry_fee
                    }

    total_trades = len(trades)
    if total_trades == 0:
        return {'Net_Profit': 0, 'Win_Rate': 0, 'Total_Trades': 0, 'EV': 0, 'Avg_Win': 0, 'Avg_Loss': 0}
        
    wins = [t for t in trades if t['Result'] == 'Win']
    losses = [t for t in trades if t['Result'] == 'Loss']
    
    num_wins = len(wins)
    num_losses = len(losses)
    
    avg_win = sum(t['Net_PnL'] for t in wins) / num_wins if num_wins > 0 else 0
    avg_loss = sum(t['Net_PnL'] for t in losses) / num_losses if num_losses > 0 else 0
    
    win_rate = num_wins / total_trades
    ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    net_profit = current_capital - INITIAL_CAPITAL
    
    return {
        'Net_Profit': net_profit,
        'Win_Rate': win_rate * 100,
        'Total_Trades': total_trades,
        'EV': ev,
        'Avg_Win': avg_win,
        'Avg_Loss': avg_loss
    }

def main():
    TIMEFRAMES = ["1h", "4h"]
    T1_VALUES = [1.0, 1.5, 2.0, 3.0]
    T2_VALUES = [2.0, 3.0, 4.0, 5.0]
    
    # Filter combinations where T2 > T1
    targets = [(t1, t2) for t1 in T1_VALUES for t2 in T2_VALUES if t2 > t1]
    
    all_results = []
    
    # Run by timeframe to manage memory
    for tf in TIMEFRAMES:
        print(f"Downloading {tf} data...")
        dfs = download_data(tf)
        
        tasks = []
        for t1, t2 in targets:
            tasks.append((dfs, tf, t1, t2))
            
        print(f"Running {len(tasks)} parallel combinations for {tf}...")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(run_simulation, tasks)
            
        for i, res in enumerate(results):
            res['Timeframe'] = tasks[i][1]
            res['T1_R'] = tasks[i][2]
            res['T2_R'] = tasks[i][3]
            all_results.append(res)
            
    df_res = pd.DataFrame(all_results)
    df_res = df_res.sort_values(by='Net_Profit', ascending=False)
    
    report_path = os.path.join(RESULTS_DIR, "extensive_sweep_results.md")
    with open(report_path, "w") as f:
        f.write("# Extensive Parameter Sweep Results\n\n")
        f.write("Modeled using Binance Futures (VIP 0 + BNB), 10x Max Leverage.\n\n")
        
        f.write("| Timeframe | Targets (T1 / T2) | Total Trades | Win Rate | EV per Trade | Avg Win | Avg Loss | Net Profit |\n")
        f.write("|-----------|-------------------|--------------|----------|--------------|---------|----------|------------|\n")
        for _, row in df_res.iterrows():
            targets_str = f"{row['T1_R']}R / {row['T2_R']}R"
            f.write(f"| {row['Timeframe']} | {targets_str} | {row['Total_Trades']} | {row['Win_Rate']:.1f}% | ${row['EV']:,.2f} | ${row['Avg_Win']:,.2f} | ${row['Avg_Loss']:,.2f} | ${row['Net_Profit']:,.2f} |\n")
            
    print(f"Sweep complete. Saved to {report_path}")

if __name__ == "__main__":
    main()
