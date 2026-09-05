import yfinance as yf
import pandas as pd
import numpy as np

# Realistic Parameters
INITIAL_CAPITAL = 10000.0
RISK_PERCENT = 0.01  # Risk 1% of account per trade
TAKER_FEE = 0.00045  # Binance Futures BNB Discount Taker
MAX_LEVERAGE = 10.0


def calculate_atr(df, period=20):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


def run_daily_trend_backtest():
    print("Downloading 5 Years of Daily BTC-USD Data...")
    df = yf.download("BTC-USD", period="1825d", interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.dropna()

    # Calculate Donchian Channels
    # Shift by 1 so today's breakout uses yesterday's 20-day high
    df["20_High"] = df["High"].rolling(20).max().shift(1)
    df["20_Low"] = df["Low"].rolling(20).min().shift(1)
    df["10_High"] = df["High"].rolling(10).max().shift(1)
    df["10_Low"] = df["Low"].rolling(10).min().shift(1)

    # Calculate ATR for position sizing
    df["ATR"] = calculate_atr(df, 20).shift(1)

    df = df.dropna()

    capital = INITIAL_CAPITAL
    position = 0  # 1 = Long, -1 = Short, 0 = Flat

    entry_price = 0.0
    qty = 0.0
    trades = []

    print("Running Turtle Trend Following Logic...")

    for i in range(len(df)):
        row = df.iloc[i]

        # We process exits first
        if position == 1:
            # Trailing Stop Hit (10-Day Low)
            if row["Low"] <= row["10_Low"]:
                exit_price = row["10_Low"]
                # Gaps can happen on daily, but crypto is 24/7 so daily gaps are minimal. We assume execution at the level.
                if row["Open"] < exit_price:
                    exit_price = row["Open"]  # Gap down

                exit_fee = (qty * exit_price) * TAKER_FEE
                pnl = (exit_price - entry_price) * qty
                net_pnl = pnl - exit_fee
                capital += net_pnl

                trades.append(
                    {
                        "Type": "Long",
                        "Entry": entry_price,
                        "Exit": exit_price,
                        "Return_%": ((exit_price - entry_price) / entry_price) * 100,
                        "Net_PnL": net_pnl,
                        "Capital": capital,
                    }
                )
                position = 0

        elif position == -1:
            # Trailing Stop Hit (10-Day High)
            if row["High"] >= row["10_High"]:
                exit_price = row["10_High"]
                if row["Open"] > exit_price:
                    exit_price = row["Open"]  # Gap up

                exit_fee = (qty * exit_price) * TAKER_FEE
                pnl = (entry_price - exit_price) * qty
                net_pnl = pnl - exit_fee
                capital += net_pnl

                trades.append(
                    {
                        "Type": "Short",
                        "Entry": entry_price,
                        "Exit": exit_price,
                        "Return_%": ((entry_price - exit_price) / entry_price) * 100,
                        "Net_PnL": net_pnl,
                        "Capital": capital,
                    }
                )
                position = 0

        # We process entries if flat
        if position == 0:
            # Check Long Breakout
            if row["High"] >= row["20_High"]:
                entry_price = row["20_High"]
                if row["Open"] > entry_price:
                    entry_price = row["Open"]

                # Turtle Rules use 2 * ATR for Stop Loss, but we'll stick to 10-Day Low trailing stop distance to calculate risk
                # Calculate initial risk per coin
                stop_loss = row["10_Low"]
                risk_per_coin = entry_price - stop_loss

                if risk_per_coin > 0:
                    account_risk = capital * RISK_PERCENT
                    ideal_qty = account_risk / risk_per_coin
                    max_qty = (capital * MAX_LEVERAGE) / entry_price
                    qty = min(ideal_qty, max_qty)

                    entry_fee = (qty * entry_price) * TAKER_FEE
                    capital -= entry_fee
                    position = 1

            # Check Short Breakout
            elif row["Low"] <= row["20_Low"]:
                entry_price = row["20_Low"]
                if row["Open"] < entry_price:
                    entry_price = row["Open"]

                stop_loss = row["10_High"]
                risk_per_coin = stop_loss - entry_price

                if risk_per_coin > 0:
                    account_risk = capital * RISK_PERCENT
                    ideal_qty = account_risk / risk_per_coin
                    max_qty = (capital * MAX_LEVERAGE) / entry_price
                    qty = min(ideal_qty, max_qty)

                    entry_fee = (qty * entry_price) * TAKER_FEE
                    capital -= entry_fee
                    position = -1

    # Force close at end
    if position != 0:
        row = df.iloc[-1]
        exit_price = row["Close"]
        exit_fee = (qty * exit_price) * TAKER_FEE

        if position == 1:
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        net_pnl = pnl - exit_fee
        capital += net_pnl
        trades.append(
            {
                "Type": "Force Close",
                "Entry": entry_price,
                "Exit": exit_price,
                "Return_%": (
                    ((exit_price - entry_price) / entry_price) * 100
                    if position == 1
                    else ((entry_price - exit_price) / entry_price) * 100
                ),
                "Net_PnL": net_pnl,
                "Capital": capital,
            }
        )

    print("\n" + "=" * 50)
    print("DAILY TREND FOLLOWING (TURTLE) RESULTS")
    print("=" * 50)

    total_trades = len(trades)
    if total_trades == 0:
        print("No trades executed.")
        return

    wins = [t for t in trades if t["Net_PnL"] > 0]
    losses = [t for t in trades if t["Net_PnL"] <= 0]

    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / total_trades) * 100

    avg_win = sum(t["Net_PnL"] for t in wins) / num_wins if num_wins > 0 else 0
    avg_loss = sum(t["Net_PnL"] for t in losses) / num_losses if num_losses > 0 else 0
    avg_win_pct = sum(t["Return_%"] for t in wins) / num_wins if num_wins > 0 else 0

    net_profit = capital - INITIAL_CAPITAL

    print(f"Total Trades: {total_trades}")
    print(f"Wins: {num_wins}")
    print(f"Losses: {num_losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Winner (Net): ${avg_win:.2f}")
    print(f"Average Loser (Net): ${avg_loss:.2f}")
    print(f"Average Winning Move: {avg_win_pct:.2f}%")
    print(f"Net Profit: ${net_profit:,.2f}")
    print(f"Final Equity: ${capital:,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    run_daily_trend_backtest()
