import yfinance as yf
import pandas as pd
import numpy as np
import os

RESULTS_DIR = "/home/kim/projects/kimsfinance/research"

# Realistic Fees
TAKER_FEE = 0.00045 # 0.045%
MAKER_FEE = 0.00018 # 0.018%

INITIAL_CAPITAL = 10000.0
# Allocate capital: $10,000 long, $10,000 short (Total Gross Exposure $20,000, which is 2x leverage)
EXPOSURE_PER_LEG = 10000.0 

def main():
    print("Downloading 1-Hour data for BTC and ETH (2 Years)...")
    
    btc = yf.download("BTC-USD", period="730d", interval="1h", progress=False)
    eth = yf.download("ETH-USD", period="730d", interval="1h", progress=False)
    
    # Flatten MultiIndex if exists
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.droplevel(1)
        eth.columns = eth.columns.droplevel(1)
        
    btc = btc[['Close']].rename(columns={'Close': 'BTC_Close'})
    eth = eth[['Close']].rename(columns={'Close': 'ETH_Close'})
    
    # Inner join to ensure perfect timestamp alignment
    df = btc.join(eth, how='inner').dropna()
    
    print(f"Data aligned. Total hours: {len(df)}")
    
    # Calculate Ratio and Z-Score
    WINDOW = 100
    df['Ratio'] = df['BTC_Close'] / df['ETH_Close']
    df['SMA'] = df['Ratio'].rolling(window=WINDOW).mean()
    df['STD'] = df['Ratio'].rolling(window=WINDOW).std()
    df['Z_Score'] = (df['Ratio'] - df['SMA']) / df['STD']
    
    df = df.dropna()
    
    capital = INITIAL_CAPITAL
    position = 0 # 1 means Long BTC / Short ETH. -1 means Short BTC / Long ETH.
    
    # Trackers
    entry_btc_price = 0.0
    entry_eth_price = 0.0
    btc_qty = 0.0
    eth_qty = 0.0
    
    trades = []
    
    print("Running Z-Score Mean Reversion Logic...")
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # ENTRY LOGIC
        if position == 0:
            if prev_row['Z_Score'] < -2.0:
                # BTC is underpriced relative to ETH -> Long BTC, Short ETH
                entry_btc_price = row['BTC_Close']
                entry_eth_price = row['ETH_Close']
                
                btc_qty = EXPOSURE_PER_LEG / entry_btc_price
                eth_qty = EXPOSURE_PER_LEG / entry_eth_price
                
                # Pay Taker fees on both entry legs
                fees = (EXPOSURE_PER_LEG * TAKER_FEE) * 2
                capital -= fees
                
                position = 1
                
            elif prev_row['Z_Score'] > 2.0:
                # BTC is overpriced relative to ETH -> Short BTC, Long ETH
                entry_btc_price = row['BTC_Close']
                entry_eth_price = row['ETH_Close']
                
                btc_qty = EXPOSURE_PER_LEG / entry_btc_price
                eth_qty = EXPOSURE_PER_LEG / entry_eth_price
                
                # Pay Taker fees on both entry legs
                fees = (EXPOSURE_PER_LEG * TAKER_FEE) * 2
                capital -= fees
                
                position = -1
                
        # EXIT LOGIC (Revert to Mean Z=0 OR Stop Loss Z=4.0)
        elif position == 1: # We are Long BTC, Short ETH
            if row['Z_Score'] >= 0.0 or row['Z_Score'] <= -4.0:
                exit_btc_price = row['BTC_Close']
                exit_eth_price = row['ETH_Close']
                
                # Calculate PnL
                btc_pnl = (exit_btc_price - entry_btc_price) * btc_qty
                eth_pnl = (entry_eth_price - exit_eth_price) * eth_qty # Short PnL
                
                gross_pnl = btc_pnl + eth_pnl
                
                # Pay Maker fees on limit (Z=0) or Taker fees on stop loss (Z=-4.0)
                exit_exposure = (btc_qty * exit_btc_price) + (eth_qty * exit_eth_price)
                fee_rate = MAKER_FEE if row['Z_Score'] >= 0.0 else TAKER_FEE
                fees = exit_exposure * fee_rate
                
                net_pnl = gross_pnl - fees
                capital += net_pnl
                
                exit_type = 'Revert (Win)' if row['Z_Score'] >= 0.0 else 'Stop Loss'
                trades.append({
                    'Type': f'Long BTC / Short ETH [{exit_type}]',
                    'Gross_PnL': gross_pnl,
                    'Fees': fees + ((EXPOSURE_PER_LEG * TAKER_FEE) * 2),
                    'Net_PnL': net_pnl
                })
                
                position = 0
                
        elif position == -1: # We are Short BTC, Long ETH
            if row['Z_Score'] <= 0.0 or row['Z_Score'] >= 4.0:
                exit_btc_price = row['BTC_Close']
                exit_eth_price = row['ETH_Close']
                
                # Calculate PnL
                btc_pnl = (entry_btc_price - exit_btc_price) * btc_qty # Short PnL
                eth_pnl = (exit_eth_price - entry_eth_price) * eth_qty
                
                gross_pnl = btc_pnl + eth_pnl
                
                # Pay Maker fees on limit (Z=0) or Taker fees on stop loss (Z=4.0)
                exit_exposure = (btc_qty * exit_btc_price) + (eth_qty * exit_eth_price)
                fee_rate = MAKER_FEE if row['Z_Score'] <= 0.0 else TAKER_FEE
                fees = exit_exposure * fee_rate
                
                net_pnl = gross_pnl - fees
                capital += net_pnl
                
                exit_type = 'Revert (Win)' if row['Z_Score'] <= 0.0 else 'Stop Loss'
                trades.append({
                    'Type': f'Short BTC / Long ETH [{exit_type}]',
                    'Gross_PnL': gross_pnl,
                    'Fees': fees + ((EXPOSURE_PER_LEG * TAKER_FEE) * 2),
                    'Net_PnL': net_pnl
                })
                
                position = 0
                
    # Close open position at end of simulation
    if position != 0:
        row = df.iloc[-1]
        exit_btc_price = row['BTC_Close']
        exit_eth_price = row['ETH_Close']
        
        if position == 1:
            btc_pnl = (exit_btc_price - entry_btc_price) * btc_qty
            eth_pnl = (entry_eth_price - exit_eth_price) * eth_qty
        else:
            btc_pnl = (entry_btc_price - exit_btc_price) * btc_qty
            eth_pnl = (exit_eth_price - entry_eth_price) * eth_qty
            
        gross_pnl = btc_pnl + eth_pnl
        exit_exposure = (btc_qty * exit_btc_price) + (eth_qty * exit_eth_price)
        fees = exit_exposure * TAKER_FEE # Market order to close
        
        net_pnl = gross_pnl - fees
        capital += net_pnl
        
        trades.append({
            'Type': 'Force Close',
            'Gross_PnL': gross_pnl,
            'Fees': fees + ((EXPOSURE_PER_LEG * TAKER_FEE) * 2),
            'Net_PnL': net_pnl
        })
                
    # Reporting
    print("\n" + "="*50)
    print("STATISTICAL ARBITRAGE (PAIRS TRADING) RESULTS")
    print("="*50)
    
    total_trades = len(trades)
    if total_trades == 0:
        print("No trades executed.")
        return
        
    wins = len([t for t in trades if t['Net_PnL'] > 0])
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    
    gross_profit = sum(t['Gross_PnL'] for t in trades)
    total_fees = sum(t['Fees'] for t in trades)
    net_profit = capital - INITIAL_CAPITAL
    
    avg_win = sum(t['Net_PnL'] for t in trades if t['Net_PnL'] > 0) / wins if wins > 0 else 0
    avg_loss = sum(t['Net_PnL'] for t in trades if t['Net_PnL'] <= 0) / losses if losses > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Gross PnL: ${gross_profit:,.2f}")
    print(f"Total Fees: ${total_fees:,.2f}")
    print(f"Net Profit: ${net_profit:,.2f}")
    print(f"Final Equity: ${capital:,.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
