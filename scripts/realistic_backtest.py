import yfinance as yf
import pandas as pd
import os

USE_BINANCE = False

if USE_BINANCE:
    INSTRUMENTS = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "DOTUSDT",
        "LINKUSDT",
    ]
else:
    INSTRUMENTS = [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "SOL-USD",
        "XRP-USD",
        "ADA-USD",
        "DOGE-USD",
        "AVAX-USD",
        "DOT-USD",
        "LINK-USD",
    ]

RESULTS_DIR = "/home/kim/projects/kimsfinance/research"

# Realistic Parameters
INITIAL_CAPITAL = 10000.0
RISK_PER_R = 100.0  # 1% of $10k
TAKER_FEE = 0.00045  # Binance Futures BNB Discount Taker
MAKER_FEE = 0.00018  # Binance Futures BNB Discount Maker
MAX_LEVERAGE = 10.0  # Standard leverage
MIN_TARGET_DISTANCE = 0.003  # 0.3% minimum price movement required for Target 1


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def run_simulation():
    print("Fetching data from yfinance/parquet...")
    dfs = {}

    for ticker in INSTRUMENTS:
        try:
            if USE_BINANCE:
                df = pd.read_parquet(
                    f"/home/kim/projects/kimsfinance/data/binance/{ticker}_5m_1y.parquet"
                )
            else:
                df = yf.download(ticker, period="730d", interval="1h", progress=False)

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [col.capitalize() for col in df.columns]

            df["EMA_9"] = calculate_ema(df["Close"], 9)
            df["EMA_20"] = calculate_ema(df["Close"], 20)
            df["EMA_50"] = calculate_ema(df["Close"], 50)
            df["EMA_20_Slope"] = df["EMA_20"].diff(3)

            dfs[ticker] = df
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if not dfs:
        print("No data fetched.")
        return

    all_timestamps = sorted(list(set.union(*[set(df.index) for df in dfs.values()])))
    open_positions = {ticker: None for ticker in INSTRUMENTS}
    trades = []

    current_day = None
    daily_pnl_R = 0.0
    daily_limit_hit = False
    limit_hit_count = 0
    current_capital = INITIAL_CAPITAL

    print("Running chronological realistic simulation...")

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
            prev_row = df.iloc[i - 1]
            prev_prev_row = df.iloc[i - 2]

            pos = open_positions[ticker]

            # --- MANAGE OPEN POSITION ---
            if pos is not None:
                high = row["High"]
                low = row["Low"]

                # Check Stop Loss
                if low <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    qty_exiting = pos["qty"] * pos["size"]

                    exit_notional = qty_exiting * exit_price
                    exit_fee = exit_notional * TAKER_FEE
                    pos["fees"] += exit_fee

                    pnl = (exit_price - pos["entry_price"]) * qty_exiting
                    pos["gross_pnl"] += pnl

                    # Estimate realized R for circuit breaker logic
                    r_realized = -1.0 if pos["size"] == 1.0 else 0.0
                    result = "Loss" if pos["size"] == 1.0 else "Win"

                    daily_pnl_R += r_realized
                    net_pnl = pos["gross_pnl"] - pos["fees"]
                    current_capital += net_pnl

                    trades.append(
                        {
                            "Ticker": ticker,
                            "Skill": pos["skill"],
                            "Result": result,
                            "Gross_PnL": pos["gross_pnl"],
                            "Fees": pos["fees"],
                            "Net_PnL": net_pnl,
                            "Entry": pos["entry_price"],
                        }
                    )

                    open_positions[ticker] = None
                    if daily_pnl_R <= -3.0:
                        if not daily_limit_hit:
                            limit_hit_count += 1
                        daily_limit_hit = True

                # Check Target 2 (Runner Target)
                elif high >= pos["target2"] and pos["size"] == 0.5:
                    exit_price = pos["target2"]
                    qty_exiting = pos["qty"] * pos["size"]

                    exit_notional = qty_exiting * exit_price
                    exit_fee = exit_notional * MAKER_FEE
                    pos["fees"] += exit_fee

                    pnl = (exit_price - pos["entry_price"]) * qty_exiting
                    pos["gross_pnl"] += pnl

                    r_realized = 1.25  # approx added R for this half
                    daily_pnl_R += r_realized
                    net_pnl = pos["gross_pnl"] - pos["fees"]
                    current_capital += net_pnl

                    trades.append(
                        {
                            "Ticker": ticker,
                            "Skill": pos["skill"],
                            "Result": "Win",
                            "Gross_PnL": pos["gross_pnl"],
                            "Fees": pos["fees"],
                            "Net_PnL": net_pnl,
                            "Entry": pos["entry_price"],
                        }
                    )
                    open_positions[ticker] = None

                # Check Target 1 (Partial Profit)
                elif high >= pos["target1"] and pos["size"] == 1.0:
                    exit_price = pos["target1"]
                    qty_exiting = pos["qty"] * 0.5

                    exit_notional = qty_exiting * exit_price
                    exit_fee = exit_notional * MAKER_FEE
                    pos["fees"] += exit_fee

                    pnl = (exit_price - pos["entry_price"]) * qty_exiting
                    pos["gross_pnl"] += pnl

                    r_realized = 0.75  # approx R
                    daily_pnl_R += r_realized

                    # Update position to runner status
                    pos["size"] = 0.5
                    pos["stop_loss"] = pos["entry_price"]  # Move stop to breakeven

                continue

            # --- FIND NEW SETUPS ---
            if daily_limit_hit:
                continue

            # Trend Filter
            is_trending_up = (
                prev_row["EMA_20_Slope"] > 0 and prev_row["EMA_20"] > prev_row["EMA_50"]
            )
            if not is_trending_up:
                continue

            skill = None
            entry_price = row["Open"]
            stop_loss = 0.0

            # EMA Crossover Setup
            if (
                prev_prev_row["EMA_9"] <= prev_prev_row["EMA_20"]
                and prev_row["EMA_9"] > prev_row["EMA_20"]
            ):
                stop_loss = min(prev_prev_row["Low"], prev_row["Low"], row["Low"])
                skill = "EMA Crossover"

            # EMA Bounce Setup
            elif (
                prev_row["High"] > prev_row["EMA_20"]
                and prev_row["Low"] <= prev_row["EMA_20"]
                and prev_row["Close"] > prev_row["Open"]
            ):
                stop_loss = prev_row["Low"] - 0.01
                skill = "EMA Bounce"

            if skill:
                risk = entry_price - stop_loss
                if risk > 0:
                    target1 = entry_price + 1.5 * risk
                    target2 = entry_price + 2.5 * risk
                    pct_movement_to_t1 = (target1 - entry_price) / entry_price

                    if pct_movement_to_t1 >= MIN_TARGET_DISTANCE:
                        ideal_qty = RISK_PER_R / risk
                        max_notional = current_capital * MAX_LEVERAGE
                        max_qty = max_notional / entry_price
                        qty = min(ideal_qty, max_qty)

                        entry_notional = qty * entry_price
                        entry_fee = entry_notional * TAKER_FEE

                        open_positions[ticker] = {
                            "qty": qty,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "risk": risk,
                            "target1": target1,
                            "target2": target2,
                            "size": 1.0,
                            "secured_r": 0.0,
                            "skill": skill,
                            "entry_time": current_time,
                            "fees": entry_fee,
                            "gross_pnl": 0.0,
                        }

    print("Generating realistic simulation report...")
    report_path = os.path.join(RESULTS_DIR, "realistic_simulation_results.md")

    with open(report_path, "w") as f:
        f.write("# Realistic Paper Trading Simulation Results\n\n")
        f.write(
            "This simulation explicitly models Binance Futures Maker/Taker fees (0.018%/0.045%) and 10x Buying Power limitations.\n\n"
        )

        df_results = pd.DataFrame(trades)
        if df_results.empty:
            f.write("No trades were executed.\n")
            return

        total_trades = len(df_results)
        wins = len(df_results[df_results["Result"] == "Win"])
        losses = len(df_results[df_results["Result"] == "Loss"])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = df_results["Gross_PnL"].sum()
        total_fees = df_results["Fees"].sum()
        net_profit = gross_profit - total_fees

        f.write("### Strategy Performance Breakdown\n")
        f.write(f"- **Gross Profit (Pure Price Movement)**: ${gross_profit:,.2f}\n")
        f.write(f"- **Total Fees Paid to Binance**: ${total_fees:,.2f}\n")
        f.write(f"- **Net Profit**: ${net_profit:,.2f}\n")
        f.write(f"- **Final Account Balance**: ${current_capital:,.2f}\n\n")

        f.write("### Trading Metrics\n")
        f.write(f"- **Total Trades Taken**: {total_trades}\n")
        f.write(f"- **Wins**: {wins}\n")
        f.write(f"- **Losses**: {losses}\n")
        f.write(f"- **Win Rate**: {win_rate:.2f}%\n")

        avg_winner = df_results[df_results["Result"] == "Win"]["Net_PnL"].mean()
        avg_loser = df_results[df_results["Result"] == "Loss"]["Net_PnL"].mean()
        f.write(f"- **Average Winner (Net)**: ${avg_winner:,.2f}\n")
        f.write(f"- **Average Loser (Net)**: ${avg_loser:,.2f}\n")

    print(f"Report written to {report_path}")


if __name__ == "__main__":
    run_simulation()
