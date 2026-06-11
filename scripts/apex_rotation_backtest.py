import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# Strategy Parameters
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.00045 # Binance Taker Fee
CONFIDENCE_THRESHOLD = 0.10 # Must beat current by 10% to rotate
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"]

def get_data(tickers, period="1460d"):
    print("Downloading historical data...")
    dfs = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty: continue
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df[['Close']].copy()
        
        # Calculate Indicators
        df['ROC_10'] = df['Close'].pct_change(10)
        df['ROC_30'] = df['Close'].pct_change(30)
        df['ROC_60'] = df['Close'].pct_change(60)
        
        daily_ret = df['Close'].pct_change()
        df['VOL_30'] = daily_ret.rolling(30).std() * np.sqrt(365) # Annualized vol
        
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        dfs[ticker] = df.dropna()
        
    # Align dates
    common_dates = None
    for ticker, df in dfs.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))
            
    common_dates = sorted(list(common_dates))
    
    for ticker in dfs:
        dfs[ticker] = dfs[ticker].loc[common_dates]
        
    return dfs, common_dates

def run_apex_rotation():
    dfs, dates = get_data(TICKERS)
    
    capital = INITIAL_CAPITAL
    current_holding = "CASH"
    qty = 0.0
    
    portfolio_history = []
    trade_log = []
    
    print("Running Apex Rotation Logic...")
    
    for i, current_date in enumerate(dates):
        if i == 0: continue
        
        # 1. Update Portfolio Value before rebalancing
        current_value = capital
        if current_holding != "CASH":
            current_price = dfs[current_holding].loc[current_date, 'Close']
            current_value = qty * current_price
            
        portfolio_history.append({'Date': current_date, 'Value': current_value, 'Holding': current_holding})
        
        # 2. Calculate Scores for today
        scores = {}
        for ticker, df in dfs.items():
            row = df.loc[current_date]
            
            roc_10 = row['ROC_10']
            roc_30 = row['ROC_30']
            roc_60 = row['ROC_60']
            vol = row['VOL_30']
            sma_50 = row['SMA_50']
            price = row['Close']
            
            if vol == 0 or pd.isna(vol): vol = 0.01
            
            weighted_mom = (roc_10 * 0.5) + (roc_30 * 0.3) + (roc_60 * 0.2)
            risk_adj_mom = weighted_mom / vol
            
            # Absolute Trend Filter: If price is below 50-day SMA, momentum score becomes 0
            trend_filter = 1.0 if price > sma_50 else 0.0
            
            final_score = risk_adj_mom * trend_filter
            scores[ticker] = final_score
            
        # 3. Find Best Asset
        best_asset = max(scores, key=scores.get)
        best_score = scores[best_asset]
        
        target_asset = current_holding
        
        # 4. Rotation Logic
        if best_score <= 0:
            # If the best asset in the market has negative momentum or is below 50-day SMA, go to cash
            target_asset = "CASH"
        elif current_holding == "CASH":
            if best_score > 0.05: # Require momentum to re-enter
                target_asset = best_asset
        else:
            current_score = scores[current_holding]
            if best_score > current_score * (1 + CONFIDENCE_THRESHOLD):
                target_asset = best_asset
                
        # 5. Execute Rotation
        if target_asset != current_holding:
            # Liquidate current
            if current_holding != "CASH":
                exit_price = dfs[current_holding].loc[current_date, 'Close']
                gross_proceeds = qty * exit_price
                fee = gross_proceeds * TAKER_FEE
                capital = gross_proceeds - fee
                
                trade_log.append(f"[{current_date.date()}] SOLD {current_holding} at ${exit_price:.2f} | Value: ${capital:.2f}")
                
            # Buy new
            if target_asset != "CASH":
                entry_price = dfs[target_asset].loc[current_date, 'Close']
                fee = capital * TAKER_FEE
                capital_to_invest = capital - fee
                qty = capital_to_invest / entry_price
                
                trade_log.append(f"[{current_date.date()}] BOUGHT {target_asset} at ${entry_price:.2f}")
                
            current_holding = target_asset

    # Final Liquidation to calculate exactly
    if current_holding != "CASH":
        final_date = dates[-1]
        exit_price = dfs[current_holding].loc[final_date, 'Close']
        gross_proceeds = qty * exit_price
        fee = gross_proceeds * TAKER_FEE
        capital = gross_proceeds - fee
        portfolio_history.append({'Date': final_date, 'Value': capital, 'Holding': "CASH"})

    # --- Metrics Calculation ---
    df_port = pd.DataFrame(portfolio_history)
    df_port.set_index('Date', inplace=True)
    
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Calculate Max Drawdown
    df_port['Peak'] = df_port['Value'].cummax()
    df_port['Drawdown'] = (df_port['Value'] - df_port['Peak']) / df_port['Peak']
    max_dd = df_port['Drawdown'].min() * 100
    
    # Calculate Buy & Hold BTC Return for the same period
    btc_df = dfs["BTC-USD"]
    btc_start = btc_df.iloc[0]['Close']
    btc_end = btc_df.iloc[-1]['Close']
    btc_return = ((btc_end - btc_start) / btc_start) * 100
    
    print("\n" + "="*50)
    print("THE APEX ROTATION RESULTS")
    print("="*50)
    print(f"Total Rotations Executed: {len(trade_log) // 2}") # Divide by 2 because buy/sell are 2 logs
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:   ${capital:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Max Drawdown:    {max_dd:.2f}%")
    print("-" * 50)
    print(f"Benchmark (Buy & Hold BTC): {btc_return:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_apex_rotation()
