import yfinance as yf
import pandas as pd
import os

INSTRUMENTS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
RESULTS_DIR = "/home/kim/projects/kimsfinance/research"

# Constraints
INITIAL_CAPITAL = 10000.0
RISK_PER_R = 100.0
MAX_LEVERAGE = 50.0


def get_pip_size(ticker):
    return 0.01 if "JPY" in ticker else 0.0001


SPREAD_PIPS = 1.5


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def download_data(timeframe):
    dfs = {}
    period = "60d" if timeframe == "5m" else "1y"
    for ticker in INSTRUMENTS:
        try:
            df = yf.download(ticker, period=period, interval=timeframe, progress=False)
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [col.capitalize() for col in df.columns]

            df["EMA_9"] = calculate_ema(df["Close"], 9)
            df["EMA_20"] = calculate_ema(df["Close"], 20)
            df["EMA_50"] = calculate_ema(df["Close"], 50)
            df["EMA_20_Slope"] = df["EMA_20"].diff(3)
            df["RSI_14"] = calculate_rsi(df["Close"], 14)

            dfs[ticker] = df
        except Exception:
            pass
    return dfs


def run_simulation(dfs, strategy, t1_r, t2_r):
    if not dfs:
        return {}

    all_timestamps = sorted(list(set.union(*[set(df.index) for df in dfs.values()])))
    df_indices = {ticker: {time: i for i, time in enumerate(dfs[ticker].index)} for ticker in dfs}

    open_positions = {ticker: None for ticker in INSTRUMENTS}
    trades = []

    current_day = None
    daily_pnl_R = 0.0
    daily_limit_hit = False
    current_capital = INITIAL_CAPITAL

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
            pip_size = get_pip_size(ticker)
            spread_cost_per_share = SPREAD_PIPS * pip_size

            # --- MANAGE OPEN POSITION ---
            if pos is not None:
                high = row["High"]
                low = row["Low"]

                # Check Stop Loss
                if low <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    qty_exiting = pos["qty"] * pos["size"]

                    pnl = (exit_price - pos["entry_price"]) * qty_exiting
                    spread_fee = spread_cost_per_share * qty_exiting

                    r_realized = -1.0 if pos["size"] == 1.0 else 0.0
                    result = "Loss" if pos["size"] == 1.0 else "Win"

                    daily_pnl_R += r_realized
                    net_pnl = pnl - spread_fee
                    current_capital += net_pnl

                    trades.append({"Result": result, "Net_PnL": net_pnl})
                    open_positions[ticker] = None
                    if daily_pnl_R <= -3.0:
                        daily_limit_hit = True

                # Check Target 2
                elif high >= pos["target2"] and pos["size"] == 0.5:
                    exit_price = pos["target2"]
                    qty_exiting = pos["qty"] * pos["size"]

                    pnl = (exit_price - pos["entry_price"]) * qty_exiting
                    spread_fee = spread_cost_per_share * qty_exiting

                    r_realized = t2_r * 0.5
                    daily_pnl_R += r_realized
                    net_pnl = pnl - spread_fee
                    current_capital += net_pnl

                    trades.append({"Result": "Win", "Net_PnL": net_pnl})
                    open_positions[ticker] = None

                # Check Target 1
                elif high >= pos["target1"] and pos["size"] == 1.0:
                    exit_price = pos["target1"]
                    qty_exiting = pos["qty"] * 0.5

                    pnl = (exit_price - pos["entry_price"]) * qty_exiting
                    spread_fee = spread_cost_per_share * qty_exiting

                    r_realized = t1_r * 0.5
                    daily_pnl_R += r_realized

                    pos["size"] = 0.5
                    pos["stop_loss"] = pos["entry_price"]

                continue

            # --- FIND NEW SETUPS ---
            if daily_limit_hit:
                continue

            entry_price = row["Open"]
            stop_loss = 0.0
            valid_setup = False

            if strategy == "EMA_Crossover":
                is_trending_up = (
                    prev_row["EMA_20_Slope"] > 0 and prev_row["EMA_20"] > prev_row["EMA_50"]
                )
                if (
                    is_trending_up
                    and prev_prev_row["EMA_9"] <= prev_prev_row["EMA_20"]
                    and prev_row["EMA_9"] > prev_row["EMA_20"]
                ):
                    stop_loss = min(prev_prev_row["Low"], prev_row["Low"], row["Low"])
                    valid_setup = True
            elif strategy == "RSI_MeanReversion":
                # Buy when RSI dips below 30 and starts curling up
                if prev_prev_row["RSI_14"] < 30 and prev_row["RSI_14"] > prev_prev_row["RSI_14"]:
                    stop_loss = min(prev_prev_row["Low"], prev_row["Low"]) - (2 * pip_size)
                    valid_setup = True

            if valid_setup:
                risk = entry_price - stop_loss
                if risk > 0:
                    ideal_qty = RISK_PER_R / risk
                    max_notional = current_capital * MAX_LEVERAGE
                    max_qty = max_notional / entry_price
                    qty = min(ideal_qty, max_qty)

                    # Spread entry fee equivalent (since we assume round trip cost on exit, we just log it)
                    # We deduct total spread fee on exit

                    open_positions[ticker] = {
                        "qty": qty,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "risk": risk,
                        "target1": entry_price + (t1_r * risk),
                        "target2": entry_price + (t2_r * risk),
                        "size": 1.0,
                    }

    total_trades = len(trades)
    if total_trades == 0:
        return {"Net_Profit": 0, "Win_Rate": 0, "Total_Trades": 0}

    wins = len([t for t in trades if t["Result"] == "Win"])
    return {
        "Net_Profit": current_capital - INITIAL_CAPITAL,
        "Win_Rate": (wins / total_trades) * 100,
        "Total_Trades": total_trades,
    }


def main():
    TIMEFRAMES = ["5m", "1h"]
    STRATEGIES = ["EMA_Crossover", "RSI_MeanReversion"]
    TARGETS = [(1.5, 2.5), (2.0, 3.0), (3.0, 4.0)]

    results = []

    for tf in TIMEFRAMES:
        print(f"Downloading {tf} data...")
        dfs = download_data(tf)

        for strategy in STRATEGIES:
            for t1, t2 in TARGETS:
                print(f"Running: {tf} | {strategy} | T1:{t1}R T2:{t2}R")
                res = run_simulation(dfs, strategy, t1, t2)
                res["Timeframe"] = tf
                res["Strategy"] = strategy
                res["T1_R"] = t1
                res["T2_R"] = t2
                results.append(res)

    # Save Results
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="Net_Profit", ascending=False)

    report_path = os.path.join(RESULTS_DIR, "forex_grid_search.md")
    with open(report_path, "w") as f:
        f.write("# Forex Grid Search Results\n\n")
        f.write("Modeled using Oanda (Zero Commission, 1.5 Pip Spread, 50x Max Leverage).\n\n")

        # Format as Markdown Table
        f.write("| Timeframe | Strategy | Targets | Total Trades | Win Rate | Net Profit |\n")
        f.write("|-----------|----------|---------|--------------|----------|------------|\n")
        for _, row in df_res.iterrows():
            targets = f"{row['T1_R']}R / {row['T2_R']}R"
            f.write(
                f"| {row['Timeframe']} | {row['Strategy']} | {targets} | {row['Total_Trades']} | {row['Win_Rate']:.1f}% | ${row['Net_Profit']:,.2f} |\n"
            )


if __name__ == "__main__":
    main()
