# Data Downloader System

Comprehensive market data downloader with organized Parquet storage.

## Features

- **Multiple Sources**: Binance Vision, Yahoo Finance, IBKR (planned)
- **Parquet Storage**: Fast, compressed columnar format perfect for backtesting
- **Organized Structure**: `exchange/type/instrument/timeframe/`
- **Parallel Downloads**: Configurable concurrent downloads
- **Resume Support**: Skip already-downloaded files
- **Checksum Verification**: Ensure data integrity

## Directory Structure

```
data/
├── binance/
│   ├── spot/
│   │   └── BTCUSDT/
│   │       ├── trades/              # Raw trade ZIP files
│   │       │   ├── 2024-01.zip
│   │       │   └── 2024-02.zip
│   │       └── ohlcv/              # Aggregated OHLCV Parquet
│   │           ├── 1m/
│   │           │   ├── 2024-01.parquet
│   │           │   └── 2024-02.parquet
│   │           ├── 5m/
│   │           ├── 15m/
│   │           └── 1h/
│   └── futures/
│       └── BTCUSDT/
├── yahoo/
│   ├── stocks/
│   │   └── AAPL/
│   │       └── daily/
│   │           └── 2024.parquet
│   └── options/
│       └── AAPL/
│           └── chain/
│               ├── 2024-01-19.parquet  # Expiration date
│               └── 2024-02-16.parquet
└── metadata/
    ├── binance_checksums.json
    └── download_history.db
```

## Usage

### CLI Tool

```bash
# Download Binance BTC spot trades for 2024
cargo run --release --example data_downloader_cli -- binance spot BTCUSDT 2024

# Download specific month
cargo run --release --example data_downloader_cli -- binance futures ETHUSDT 2024 1

# Download Yahoo Finance stock data
cargo run --release --example data_downloader_cli -- yahoo stock AAPL 2024-01-01 2024-12-31

# Download options chain (nearest expiration)
cargo run --release --example data_downloader_cli -- yahoo options SPY
```

### Programmatic API

```rust
use kimsfinance_core::data::downloaders::{BinanceDownloader, DownloadConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DownloadConfig {
        base_path: "data".into(),
        parallel_downloads: 4,
        verify_checksums: true,
        resume: true,
    };

    // Download Binance data
    let downloader = BinanceDownloader::new(config);

    // Download all 2024 BTC spot trades
    downloader.download_spot_trades("BTCUSDT", 2024, None).await?;

    // Aggregate to multiple timeframes
    downloader.aggregate_to_ohlcv("BTCUSDT", "1m", 2024).await?;
    downloader.aggregate_to_ohlcv("BTCUSDT", "5m", 2024).await?;
    downloader.aggregate_to_ohlcv("BTCUSDT", "1h", 2024).await?;

    Ok(())
}
```

## Data Sources

### 1. Binance Vision (Free)

**Source**: https://data.binance.vision/

**Available Data**:
- Spot market trade data (tick-level)
- Futures market trade data
- Monthly archives (YYYY-MM format)
- Free, no API key required

**Symbols**: All Binance trading pairs (BTCUSDT, ETHUSDT, etc.)

**Example Download**:
```bash
# Direct download with wget
wget https://data.binance.vision/data/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2024-01.zip

# Or use our CLI
cargo run --release --example data_downloader_cli -- binance spot BTCUSDT 2024
```

### 2. Yahoo Finance (Free)

**API Endpoint**: https://query1.finance.yahoo.com/v7/finance/

**Available Data**:
- Historical stock prices (daily)
- Options chains with Greeks
- Real-time quotes
- Free, no API key required

**Symbols**: All US stocks + major international (AAPL, SPY, TSLA, etc.)

**Example**:
```rust
use kimsfinance_core::data::downloaders::YahooDownloader;

let downloader = YahooDownloader::new(config);

// Download stock data
downloader.download_stock("AAPL", "2024-01-01", "2024-12-31").await?;

// Download options chain
downloader.download_options_chain("SPY", None).await?;
```

**Note**: Yahoo Finance options API implementation is pending. See:
- Reference: https://github.com/pydata/pandas-datareader/blob/main/pandas_datareader/yahoo/options.py
- API endpoint structure is in place

### 3. IBKR (Interactive Brokers)

**Status**: Connector stub exists, implementation pending

**Requires**: IBKR account + API access

## Parquet Schema

### OHLCV Candles (Binance/Yahoo Stocks)

```
Schema:
  - timestamp: int64 (Unix milliseconds)
  - open: float64
  - high: float64
  - low: float64
  - close: float64
  - volume: float64
  - num_trades: uint32
```

### Options Chain (Yahoo/IBKR)

```
Schema (planned):
  - timestamp: int64
  - strike: float64
  - expiration: int64
  - bid: float64
  - ask: float64
  - last: float64
  - volume: uint64
  - open_interest: uint64
  - implied_volatility: float64
  - delta: float64 (optional)
  - gamma: float64 (optional)
  - theta: float64 (optional)
  - vega: float64 (optional)
  - rho: float64 (optional)
```

## Reading Parquet Files

### With Polars (Recommended)

```rust
use polars::prelude::*;

fn load_ohlcv(path: &str) -> Result<DataFrame, PolarsError> {
    LazyFrame::scan_parquet(path, Default::default())?
        .select([
            col("timestamp"),
            col("open"),
            col("high"),
            col("low"),
            col("close"),
            col("volume"),
        ])
        .collect()
}

// Example: Load and filter
let df = load_ohlcv("data/binance/spot/BTCUSDT/ohlcv/5m/2024-01.parquet")?;
let filtered = df
    .lazy()
    .filter(col("close").gt(50000.0))
    .collect()?;
```

### With Arrow

```rust
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

fn load_parquet(path: &str) -> Result<Vec<RecordBatch>, ArrowError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    reader.collect()
}
```

## Performance

### Binance Downloads

- **Network Speed**: Typically 10-50 MB/s (depends on connection)
- **Aggregation**: ~500K trades/sec to OHLCV
- **Parquet Compression**: ~60-80% smaller than CSV
- **2024 BTC Full Year**: ~5GB trades → ~500MB Parquet (1m candles)

### Yahoo Downloads

- **Stock Data**: Instant (CSV download < 1MB for 1 year daily)
- **Options Chain**: ~2-5 seconds per expiration
- **Rate Limits**: ~2000 requests/hour (Yahoo unofficial limit)

## Backtesting Integration

### Using Downloaded Data

```rust
use polars::prelude::*;
use kimsfinance_core::quantitative::heston::execution::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load Binance OHLCV data
    let df = LazyFrame::scan_parquet(
        "data/binance/spot/BTCUSDT/ohlcv/5m/2024-*.parquet",
        Default::default()
    )?
    .collect()?;

    // Extract spot prices
    let spot_prices: Vec<f64> = df
        .column("close")?
        .f64()?
        .into_iter()
        .filter_map(|x| x)
        .collect();

    // Run backtest with execution engine
    let config = ExecutionConfig {
        initial_capital: 100_000.0,
        max_position_pct: 0.1,
        margin_requirement: 0.2,
    };

    let mut engine = ExecutionEngine::new(config)?;

    // Backtest loop
    for (i, &spot) in spot_prices.iter().enumerate() {
        // Generate option quotes from Heston model
        let options = generate_synthetic_options(spot, heston_params);

        // Generate signals
        let signals = your_strategy.generate_signals(&options);

        // Execute
        let result = engine.process_timestep(&market_data, &signals)?;
    }

    Ok(())
}
```

## TODO

### Yahoo Finance Options API

The Yahoo Finance options downloader has the structure in place but needs implementation:

**Reference**: https://github.com/pydata/pandas-datareader/blob/main/pandas_datareader/yahoo/options.py

**API Endpoints**:
```
# Get expiration dates
GET https://query1.finance.yahoo.com/v7/finance/options/{symbol}

# Get options chain for specific expiration
GET https://query1.finance.yahoo.com/v7/finance/options/{symbol}?date={unix_timestamp}
```

**Implementation needed**:
1. Parse JSON response from Yahoo API
2. Extract calls/puts with Greeks
3. Convert to `OptionQuote` struct
4. Write to Parquet with proper schema

### IBKR Connector

Implement Interactive Brokers data connector using TWS API.

### Checksum Verification

Add MD5 checksum verification for Binance downloads (checksums available at data.binance.vision).

### Parallel Download Optimization

Implement tokio-based parallel downloads for faster bulk downloads.

## Contributing

To add a new data source:

1. Create `src/data/downloaders/{source}.rs`
2. Implement `Downloader` trait
3. Add Parquet schema for the data format
4. Update CLI in `examples/data_downloader_cli.rs`
5. Document in this file

## License

Same as kimsfinance project.
