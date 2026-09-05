import yfinance as yf
import pandas as pd
import json

RESULTS_DIR = "/home/kim/projects/kimsfinance/research"

INITIAL_CAPITAL = 10000.0
RISK_PER_R = 100.0
TAKER_FEE = 0.00045  # Binance Futures BNB Discount Taker
MAKER_FEE = 0.00018  # Binance Futures BNB Discount Maker
MAX_LEVERAGE = 10.0


def run_bollinger_simulation(df, period, std_dev, entry_std, exit_std):
    # Calculate Bollinger Bands
    df["SMA"] = df["Close"].rolling(window=int(period)).mean()
    df["STD"] = df["Close"].rolling(window=int(period)).std()

    df["Upper_Entry"] = df["SMA"] + (df["STD"] * entry_std)
    df["Lower_Entry"] = df["SMA"] - (df["STD"] * entry_std)

    df["Upper_Exit"] = df["SMA"] + (df["STD"] * exit_std)
    df["Lower_Exit"] = df["SMA"] - (df["STD"] * exit_std)

    # Pre-calculate signals (Long Only for simplicity, or Long/Short)
    # Let's do Long and Short since Mean Reversion works both ways

    capital = INITIAL_CAPITAL
    position = 0  # 1 for Long, -1 for Short, 0 for Flat
    entry_price = 0.0
    qty = 0.0
    trades = []

    for i in range(int(period), len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # If flat, look for entry
        if position == 0:
            # Long Signal: Price dips below Lower Entry Band
            if prev_row["Low"] <= prev_row["Lower_Entry"]:
                entry_price = row["Open"]

                # Assume stop loss is 2 standard deviations below entry
                stop_loss = entry_price - (row["STD"] * 2)
                risk = entry_price - stop_loss
                if risk > 0:
                    ideal_qty = RISK_PER_R / risk
                    max_qty = (capital * MAX_LEVERAGE) / entry_price
                    qty = min(ideal_qty, max_qty)

                    entry_fee = (qty * entry_price) * TAKER_FEE
                    capital -= entry_fee
                    position = 1

            # Short Signal: Price spikes above Upper Entry Band
            elif prev_row["High"] >= prev_row["Upper_Entry"]:
                entry_price = row["Open"]

                stop_loss = entry_price + (row["STD"] * 2)
                risk = stop_loss - entry_price
                if risk > 0:
                    ideal_qty = RISK_PER_R / risk
                    max_qty = (capital * MAX_LEVERAGE) / entry_price
                    qty = min(ideal_qty, max_qty)

                    entry_fee = (qty * entry_price) * TAKER_FEE
                    capital -= entry_fee
                    position = -1

        # Manage Long
        elif position == 1:
            # Exit at Mean/Exit Band
            if row["High"] >= row["Lower_Exit"]:
                exit_price = row["Lower_Exit"]
                exit_fee = (qty * exit_price) * MAKER_FEE

                pnl = (exit_price - entry_price) * qty
                net_pnl = pnl - exit_fee
                capital += net_pnl
                trades.append(net_pnl)
                position = 0

            # Stop Loss
            elif row["Low"] <= entry_price - (row["STD"] * 2):
                exit_price = entry_price - (row["STD"] * 2)
                exit_fee = (qty * exit_price) * TAKER_FEE

                pnl = (exit_price - entry_price) * qty
                net_pnl = pnl - exit_fee
                capital += net_pnl
                trades.append(net_pnl)
                position = 0

        # Manage Short
        elif position == -1:
            # Exit at Mean/Exit Band
            if row["Low"] <= row["Upper_Exit"]:
                exit_price = row["Upper_Exit"]
                exit_fee = (qty * exit_price) * MAKER_FEE

                pnl = (entry_price - exit_price) * qty
                net_pnl = pnl - exit_fee
                capital += net_pnl
                trades.append(net_pnl)
                position = 0

            # Stop Loss
            elif row["High"] >= entry_price + (row["STD"] * 2):
                exit_price = entry_price + (row["STD"] * 2)
                exit_fee = (qty * exit_price) * TAKER_FEE

                pnl = (entry_price - exit_price) * qty
                net_pnl = pnl - exit_fee
                capital += net_pnl
                trades.append(net_pnl)
                position = 0

    num_trades = len(trades)
    if num_trades == 0:
        return {"net_profit": 0, "win_rate": 0, "total_trades": 0}

    wins = [t for t in trades if t > 0]
    win_rate = len(wins) / num_trades

    return {
        "net_profit": capital - INITIAL_CAPITAL,
        "win_rate": win_rate * 100,
        "total_trades": num_trades,
        "final_equity": capital,
    }


def main():
    print("Downloading 1-Hour BTC data (2 Years)...")
    df = yf.download("BTC-USD", period="730d", interval="1h", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.dropna()

    PERIODS = [10.0, 20.0, 30.0, 40.0, 50.0]
    STD_DEVS = [2.0]
    ENTRY_STDS = [2.0, 2.5, 3.0]
    EXIT_STDS = [0.0, -0.5, 0.5]

    all_results = []

    print("Running Python Bollinger Parameter Sweep...")
    for period in PERIODS:
        for std_dev in STD_DEVS:
            for entry_std in ENTRY_STDS:
                for exit_std in EXIT_STDS:
                    res = run_bollinger_simulation(df.copy(), period, std_dev, entry_std, exit_std)

                    all_results.append(
                        {
                            "params": {
                                "period": period,
                                "std_dev": std_dev,
                                "entry_std": entry_std,
                                "exit_std": exit_std,
                            },
                            "net_profit": res["net_profit"],
                            "win_rate": res["win_rate"],
                            "total_trades": res["total_trades"],
                            "final_equity": res["final_equity"],
                        }
                    )

    # Sort by net profit
    all_results.sort(key=lambda x: x["net_profit"], reverse=True)

    best_run = all_results[0]

    print("\n" + "=" * 50)
    print("SWEEP COMPLETE - BEST CONFIGURATION FOUND")
    print("=" * 50)
    print(f"Best Parameters: {json.dumps(best_run['params'], indent=2)}")
    print(f"Win Rate: {best_run['win_rate']:.2f}%")
    print(f"Total Trades: {best_run['total_trades']}")
    print(f"Final Equity: ${best_run['final_equity']:,.2f}")
    print(f"Net Profit: ${best_run['net_profit']:,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
