import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os

try:
    from kimsfinance.plotting import render_and_save
    HAS_KIMSFINANCE = True
except ImportError:
    HAS_KIMSFINANCE = False

USE_BINANCE = True

if USE_BINANCE:
    INSTRUMENTS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT']
else:
    INSTRUMENTS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD']

RESULTS_DIR = "/home/kim/projects/kimsfinance/research"
CHARTS_DIR = os.path.join(RESULTS_DIR, "advanced_paper_trades")

os.makedirs(CHARTS_DIR, exist_ok=True)

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def render_trade_chart(df, entry_time, exit_time, ticker, skill, result, trade_id):
    if not HAS_KIMSFINANCE:
        return
        
    # Get integer indices for the slice
    try:
        entry_idx = df.index.get_loc(entry_time)
        exit_idx = df.index.get_loc(exit_time)
    except KeyError:
        return
        
    start_idx = max(0, entry_idx - 30)
    end_idx = min(len(df) - 1, exit_idx + 10)
    
    df_slice = df.iloc[start_idx:end_idx+1]
    
    ohlc = {
        'open': df_slice['Open'].values,
        'high': df_slice['High'].values,
        'low': df_slice['Low'].values,
        'close': df_slice['Close'].values,
    }
    volume = df_slice['Volume'].values
    
    filename = f"{skill.replace(' ', '_')}_{ticker}_{result}_{trade_id}.webp"
    output_path = os.path.join(CHARTS_DIR, filename)
    
    try:
        render_and_save(
            ohlc=ohlc,
            volume=volume,
            output_path=output_path,
            width=800,
            height=400,
            theme='modern',
            format='webp',
            speed='fast'
        )
    except Exception as e:
        print(f"Failed to render chart for {ticker}: {e}")

def run_simulation():
    print("Fetching data from yfinance...")
    dfs = {}
    
    for ticker in INSTRUMENTS:
        try:
            if USE_BINANCE:
                df = pd.read_parquet(f"/home/kim/projects/kimsfinance/data/binance/{ticker}_5m_1y.parquet")
            else:
                df = yf.download(ticker, period="1y", interval="1h", progress=False)
                
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [col.capitalize() for col in df.columns]
            
            df['EMA_9'] = calculate_ema(df['Close'], 9)
            df['EMA_20'] = calculate_ema(df['Close'], 20)
            df['EMA_50'] = calculate_ema(df['Close'], 50)
            
            # Precompute slope for trend filter (simple momentum of EMA_20)
            df['EMA_20_Slope'] = df['EMA_20'].diff(3)
            
            dfs[ticker] = df
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            
    if not dfs:
        print("No data fetched.")
        return

    # Gather all unique timestamps to step chronologically
    all_timestamps = sorted(list(set.union(*[set(df.index) for df in dfs.values()])))
    
    # State tracking
    open_positions = {ticker: None for ticker in INSTRUMENTS}
    trades = []
    trade_counter = 0
    charts_generated = {'Win': 0, 'Loss': 0, 'Partial': 0}
    
    current_day = None
    daily_pnl_R = 0.0
    daily_limit_hit = False
    limit_hit_count = 0
    
    INITIAL_CAPITAL = 10000.0
    RISK_PER_R = 100.0  # 1% of $10k
    current_capital = INITIAL_CAPITAL

    print("Running chronological simulation...")
    
    # Pre-calculate indices for fast lookups
    df_indices = {ticker: {time: i for i, time in enumerate(dfs[ticker].index)} for ticker in dfs}

    for current_time in all_timestamps:
        day = current_time.date()
        if current_day != day:
            current_day = day
            daily_pnl_R = 0.0
            daily_limit_hit = False
            
        for ticker, df in dfs.items():
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
                    # Calculate realized R based on remaining size
                    if pos['size'] == 1.0:
                        # Full stop out (-1R)
                        r_realized = -1.0
                        result = 'Loss'
                    else:
                        # Stopped out on runner at breakeven (0R for this half, total trade was +0.75R from first half)
                        r_realized = 0.0 
                        result = 'Win' # Still a winning trade overall since we secured 1.5R on half
                    
                    daily_pnl_R += r_realized
                    current_capital += r_realized * RISK_PER_R
                    
                    trades.append({
                        'Ticker': ticker, 'Skill': pos['skill'], 
                        'Result': result, 'Total_R': pos['secured_r'] + r_realized, 
                        'Entry': pos['entry_price']
                    })
                    
                    if charts_generated[result] < 5:
                        render_trade_chart(df, pos['entry_time'], current_time, ticker, pos['skill'], result, trade_counter)
                        charts_generated[result] += 1
                        
                    open_positions[ticker] = None
                    trade_counter += 1
                    
                    if daily_pnl_R <= -3.0:
                        if not daily_limit_hit:
                            limit_hit_count += 1
                        daily_limit_hit = True
                        
                # Check Target 2 (Runner Target)
                elif high >= pos['target2'] and pos['size'] == 0.5:
                    r_realized = 2.5 * 0.5 # 1.25R on the runner
                    daily_pnl_R += r_realized
                    current_capital += r_realized * RISK_PER_R
                    
                    trades.append({
                        'Ticker': ticker, 'Skill': pos['skill'], 
                        'Result': 'Win', 'Total_R': pos['secured_r'] + r_realized, 
                        'Entry': pos['entry_price']
                    })
                    
                    if charts_generated['Win'] < 5:
                        render_trade_chart(df, pos['entry_time'], current_time, ticker, pos['skill'], 'Win_Full', trade_counter)
                        charts_generated['Win'] += 1
                        
                    open_positions[ticker] = None
                    trade_counter += 1
                    
                # Check Target 1 (Partial Profit)
                elif high >= pos['target1'] and pos['size'] == 1.0:
                    r_realized = 1.5 * 0.5 # 0.75R secured
                    daily_pnl_R += r_realized
                    current_capital += r_realized * RISK_PER_R
                    
                    # Update position to runner status
                    pos['size'] = 0.5
                    pos['secured_r'] += r_realized
                    pos['stop_loss'] = pos['entry_price'] # Move stop to breakeven
                    
                    if charts_generated['Partial'] < 5:
                        render_trade_chart(df, pos['entry_time'], current_time, ticker, pos['skill'], 'Partial', trade_counter)
                        charts_generated['Partial'] += 1
                        
                continue

            # --- FIND NEW SETUPS ---
            # Do not take new trades if daily limit hit
            if daily_limit_hit:
                continue

            # Trend Filter (EMA 20 must be sloped upwards and above EMA 50)
            is_trending_up = prev_row['EMA_20_Slope'] > 0 and prev_row['EMA_20'] > prev_row['EMA_50']
            
            if not is_trending_up:
                continue

            # EMA Crossover Setup
            if prev_prev_row['EMA_9'] <= prev_prev_row['EMA_20'] and prev_row['EMA_9'] > prev_row['EMA_20']:
                entry_price = row['Open']
                stop_loss = min(prev_prev_row['Low'], prev_row['Low'], row['Low'])
                
                risk = entry_price - stop_loss
                if risk > 0:
                    open_positions[ticker] = {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'risk': risk,
                        'target1': entry_price + 1.5 * risk,
                        'target2': entry_price + 2.5 * risk,
                        'size': 1.0,
                        'secured_r': 0.0,
                        'skill': "EMA Crossover",
                        'entry_time': current_time
                    }
                continue

            # EMA Bounce Setup
            if prev_row['High'] > prev_row['EMA_20'] and prev_row['Low'] <= prev_row['EMA_20']:
                if prev_row['Close'] > prev_row['Open']:
                    entry_price = row['Open']
                    stop_loss = prev_row['Low'] - 0.01
                    
                    risk = entry_price - stop_loss
                    if risk > 0:
                        open_positions[ticker] = {
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'risk': risk,
                            'target1': entry_price + 1.5 * risk,
                            'target2': entry_price + 2.5 * risk,
                            'size': 1.0,
                            'secured_r': 0.0,
                            'skill': "EMA Bounce",
                            'entry_time': current_time
                        }

    print("Generating simulation report...")
    report_path = os.path.join(RESULTS_DIR, "simulation_results.md")
    
    with open(report_path, "w") as f:
        f.write("# Advanced Agent Paper Trading Simulation Results\n\n")
        f.write("This simulation implemented the contextual Trend Filter and the advanced risk management rules from **Book 11: The Risk Architect** (Scaling Out at 1.5R, Breakeven Trailing, and -3R Daily Limits).\n\n")
        if USE_BINANCE:
            f.write("**Parameters**: 1 Year of 5-Minute Data (Binance Vision), $10,000 Starting Account, 1% Risk ($100) per trade.\n\n")
        else:
            f.write("**Parameters**: 1 Year of 1-Hour Data (yfinance), $10,000 Starting Account, 1% Risk ($100) per trade.\n\n")
        
        df_results = pd.DataFrame(trades)
        if df_results.empty:
            f.write("No trades were executed.\n")
            return
            
        total_trades = len(df_results)
        wins = len(df_results[df_results['Result'] == 'Win'])
        losses = len(df_results[df_results['Result'] == 'Loss'])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_r = df_results['Total_R'].sum()
        
        f.write(f"- **Total Trades Taken**: {total_trades}\n")
        f.write(f"- **Wins**: {wins}\n")
        f.write(f"- **Losses**: {losses}\n")
        f.write(f"- **Win Rate**: {win_rate:.2f}%\n")
        f.write(f"- **Total Profit (in R Units)**: {total_r:.2f}R\n")
        f.write(f"- **Final Account Balance**: ${current_capital:,.2f} (Net Profit: ${current_capital - INITIAL_CAPITAL:,.2f})\n")
        f.write(f"- **Daily Circuit Breaker Triggered (-3R)**: {limit_hit_count} times\n\n")

    print(f"Report written to {report_path}")
    print(f"Charts saved to {CHARTS_DIR}")

if __name__ == "__main__":
    run_simulation()
