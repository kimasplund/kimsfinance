# Binance BTCUSDT Futures Backtest Results

## Test Configuration

- **Data Source**: /home/kim-asplund/projects/binance-data/futures/BTCUSDT/trades/BTCUSDT-trades-2024-05-31.zip
- **Date**: 2024-05-31 (1 day)
- **Initial Capital**: $10,000
- **Trading Fee**: 0.1% (Binance futures taker fee)
- **Slippage**: 0.05%
- **GPU Acceleration**: DISABLED (CPU)
- **Total Test Duration**: 2.15s

## Results by Timeframe

### 1min Timeframe

| Strategy | Return % | Sharpe | Max DD % | Win Rate % | Trades | Profit Factor | Final Equity |
|----------|----------|--------|----------|------------|--------|---------------|--------------|
| RSI(14, 30, 70) | -20.78 | -0.70 | 23.76 | 25.00 | 4 | 0.10 | $7922.48 |
| RSI(14, 25, 75) | -8.45 | -0.23 | 19.41 | 50.00 | 4 | 0.57 | $9154.67 |
| RSI(21, 30, 70) | -9.71 | -0.28 | 18.80 | 33.33 | 3 | 0.14 | $9028.82 |
| ATR(14) | -12.66 | -0.29 | 22.04 | 0.00 | 1 | -0.00 | $8734.17 |
| ATR(7) | -123.81 | 0.30 | 122.55 | 16.36 | 55 | 0.17 | $-2381.42 |
| MACD(12, 26, 9) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | $10000.00 |
| MACD(5, 13, 5) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | $10000.00 |

### 5min Timeframe

| Strategy | Return % | Sharpe | Max DD % | Win Rate % | Trades | Profit Factor | Final Equity |
|----------|----------|--------|----------|------------|--------|---------------|--------------|
| RSI(14, 30, 70) | 8.16 | 1.11 | 7.52 | 100.00 | 3 | inf | $10815.84 |
| RSI(14, 25, 75) | 3.02 | 0.45 | 7.73 | 100.00 | 1 | inf | $10302.39 |
| RSI(21, 30, 70) | -0.08 | 0.22 | 7.73 | 100.00 | 1 | inf | $9992.16 |
| ATR(14) | -11.52 | -0.61 | 20.85 | 0.00 | 1 | -0.00 | $8848.34 |
| ATR(7) | -16.57 | -0.93 | 21.90 | 25.00 | 4 | 0.01 | $8342.66 |
| MACD(12, 26, 9) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | $10000.00 |
| MACD(5, 13, 5) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | $10000.00 |

### 15min Timeframe

| Strategy | Return % | Sharpe | Max DD % | Win Rate % | Trades | Profit Factor | Final Equity |
|----------|----------|--------|----------|------------|--------|---------------|--------------|
| RSI(14, 30, 70) | -0.08 | 0.38 | 6.66 | 100.00 | 1 | inf | $9992.16 |
| RSI(14, 25, 75) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | $10000.00 |
| RSI(21, 30, 70) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | $10000.00 |
| ATR(14) | -8.71 | -0.85 | 19.16 | 0.00 | 1 | -0.00 | $9129.47 |
| ATR(7) | -9.78 | -0.91 | 19.36 | 0.00 | 1 | -0.00 | $9021.60 |
| MACD(12, 26, 9) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | $10000.00 |
| MACD(5, 13, 5) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | $10000.00 |

## Best Performing Strategies

### Top 5 by Total Return

| Rank | Strategy | Timeframe | Return % | Sharpe | Trades |
|------|----------|-----------|----------|--------|--------|
| 1 | RSI(14, 30, 70) | 5min | 8.16 | 1.11 | 3 |
| 2 | RSI(14, 25, 75) | 5min | 3.02 | 0.45 | 1 |
| 3 | MACD(12, 26, 9) | 1min | 0.00 | 0.00 | 0 |
| 4 | MACD(5, 13, 5) | 1min | 0.00 | 0.00 | 0 |
| 5 | MACD(12, 26, 9) | 5min | 0.00 | 0.00 | 0 |

### Top 5 by Sharpe Ratio

| Rank | Strategy | Timeframe | Sharpe | Return % | Max DD % |
|------|----------|-----------|--------|----------|----------|
| 1 | RSI(14, 30, 70) | 5min | 1.11 | 8.16 | 7.52 |
| 2 | RSI(14, 25, 75) | 5min | 0.45 | 3.02 | 7.73 |
| 3 | RSI(14, 30, 70) | 15min | 0.38 | -0.08 | 6.66 |
| 4 | ATR(7) | 1min | 0.30 | -123.81 | 122.55 |
| 5 | RSI(21, 30, 70) | 5min | 0.22 | -0.08 | 7.73 |

## Strategy Comparison Across Timeframes

### RSI(14, 25, 75)

| Timeframe | Return % | Sharpe | Max DD % | Trades |
|-----------|----------|--------|----------|--------|
| 1min | -8.45 | -0.23 | 19.41 | 4 |
| 5min | 3.02 | 0.45 | 7.73 | 1 |
| 15min | 0.00 | 0.00 | 0.00 | 0 |

### RSI(14, 30, 70)

| Timeframe | Return % | Sharpe | Max DD % | Trades |
|-----------|----------|--------|----------|--------|
| 1min | -20.78 | -0.70 | 23.76 | 4 |
| 5min | 8.16 | 1.11 | 7.52 | 3 |
| 15min | -0.08 | 0.38 | 6.66 | 1 |

### RSI(21, 30, 70)

| Timeframe | Return % | Sharpe | Max DD % | Trades |
|-----------|----------|--------|----------|--------|
| 1min | -9.71 | -0.28 | 18.80 | 3 |
| 5min | -0.08 | 0.22 | 7.73 | 1 |
| 15min | 0.00 | 0.00 | 0.00 | 0 |

### ATR(14)

| Timeframe | Return % | Sharpe | Max DD % | Trades |
|-----------|----------|--------|----------|--------|
| 1min | -12.66 | -0.29 | 22.04 | 1 |
| 5min | -11.52 | -0.61 | 20.85 | 1 |
| 15min | -8.71 | -0.85 | 19.16 | 1 |

### ATR(7)

| Timeframe | Return % | Sharpe | Max DD % | Trades |
|-----------|----------|--------|----------|--------|
| 1min | -123.81 | 0.30 | 122.55 | 55 |
| 5min | -16.57 | -0.93 | 21.90 | 4 |
| 15min | -9.78 | -0.91 | 19.36 | 1 |

### MACD(12, 26, 9)

| Timeframe | Return % | Sharpe | Max DD % | Trades |
|-----------|----------|--------|----------|--------|
| 1min | 0.00 | 0.00 | 0.00 | 0 |
| 5min | 0.00 | 0.00 | 0.00 | 0 |
| 15min | 0.00 | 0.00 | 0.00 | 0 |

### MACD(5, 13, 5)

| Timeframe | Return % | Sharpe | Max DD % | Trades |
|-----------|----------|--------|----------|--------|
| 1min | 0.00 | 0.00 | 0.00 | 0 |
| 5min | 0.00 | 0.00 | 0.00 | 0 |
| 15min | 0.00 | 0.00 | 0.00 | 0 |

## Performance Summary

- **Total Strategies Tested**: 21
- **Average Return**: -10.05%
- **Winning Strategies**: 2 (9.5%)
- **Average Backtest Time**: 0.87ms

## Key Findings

1. **Best Overall Strategy**: RSI(14, 30, 70) on 5min timeframe with 8.16% return and 1.11 Sharpe ratio
2. **Best Risk-Adjusted Returns**: RSI(14, 30, 70) on 5min timeframe with 1.11 Sharpe ratio
3. **RSI Strategies**: Average return -3.10% across 9 tests
4. **ATR Strategies**: Average return -30.51% across 6 tests
5. **MACD Strategies**: Average return 0.00% across 6 tests

---

*Report generated on 2025-10-26 using kimsfinance_core backtesting engine*
