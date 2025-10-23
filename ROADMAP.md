# kimsfinance Roadmap

This document outlines the future development plans for kimsfinance.

---

## v0.1.0 (Current - Beta Release)

**Status**: In Beta Testing
**Release Date**: October 2025

### Highlights
- 32 GPU-accelerated technical indicators
- 6 chart types (Candlestick, OHLC, Line, Hollow, Renko, PnF)
- 28.8x average speedup over mplfinance (validated: 7.3x - 70.1x)
- WebP optimization (61x faster encoding, 79% smaller files)
- SVG/SVGZ vector export
- Native PIL rendering

### Available Indicators (32 Total)
- **Moving Averages**: SMA, EMA, WMA, DEMA, TEMA, HMA, VWMA
- **Momentum**: RSI, MACD, Stochastic, Williams %R, CCI, ROC, TSI
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian Channels
- **Volume**: OBV, VWAP, CMF, Volume Profile, MFI
- **Trend**: ADX, Parabolic SAR, Aroon, Supertrend, Ichimoku Cloud
- **Support/Resistance**: Fibonacci Retracement, Pivot Points, Elder Ray

---

## v0.2.0 (Q1 2026) - Enhanced Indicator Suite

**Focus**: Additional popular indicators and performance optimizations

### Planned Features

#### New Indicators (8-10)
- [ ] **Squeeze Indicator** (Bollinger Bands + Keltner Channels)
- [ ] **VWAP Bands** (Standard deviation bands around VWAP)
- [ ] **Accumulation/Distribution Line**
- [ ] **Chaikin Oscillator** (MACD of A/D Line)
- [ ] **Force Index** (Price change × Volume)
- [ ] **Elder Impulse System** (EMA + MACD-Histogram)
- [ ] **Know Sure Thing (KST)** (Smoothed momentum oscillator)
- [ ] **Ease of Movement (EMV)**

#### Performance Enhancements
- [ ] Custom CUDA kernels for iterative indicators (Parabolic SAR, Ichimoku)
- [ ] Batch indicator calculation optimization
- [ ] Memory pooling for GPU operations
- [ ] Multi-GPU support for massive datasets

#### Usability Improvements
- [ ] Indicator presets (e.g., "momentum_suite", "volume_analysis")
- [ ] Indicator combination strategies
- [ ] Built-in backtesting support

---

## v0.3.0 (Q2 2026) - Advanced Chart Types

**Focus**: Additional chart types and visualization options

### Planned Features

#### New Chart Types (4-5)
- [ ] **Heikin Ashi** candlesticks
- [ ] **Kagi Charts** (price-only, direction-based)
- [ ] **Three Line Break** charts
- [ ] **Volume Candles** (candle width = volume)
- [ ] **Range Bars** (constant range per bar)

#### Visualization Enhancements
- [ ] Multi-panel layouts (2-4 panels)
- [ ] Indicator overlays on main chart
- [ ] Customizable panel heights
- [ ] Drawing tools (trendlines, rectangles, annotations)
- [ ] Chart templates (save/load configurations)

#### Export Formats
- [ ] PDF export (vector)
- [ ] High-DPI PNG (4K, 8K)
- [ ] Animated GIFs (for time series)

---

## v0.4.0 (Q3 2026) - Real-Time & Streaming

**Focus**: Real-time data processing and streaming support

### Planned Features

#### Streaming Support
- [ ] WebSocket integration (Binance, Coinbase, etc.)
- [ ] Real-time indicator updates (rolling window)
- [ ] Live chart updates (sub-second latency)
- [ ] Sliding window optimizations

#### Data Integration
- [ ] Built-in data downloaders (Yahoo Finance, Alpha Vantage)
- [ ] Database connectors (PostgreSQL, TimescaleDB, InfluxDB)
- [ ] Cloud storage integration (S3, GCS, Azure Blob)
- [ ] Parquet streaming (read/write)

#### Performance
- [ ] Zero-copy data transfer (GPU ↔ CPU)
- [ ] Async rendering pipeline
- [ ] Parallel indicator calculation (multi-GPU)

---

## v0.5.0 (Q4 2026) - Machine Learning Integration

**Focus**: ML/AI features for chart pattern recognition and prediction

### Planned Features

#### Pattern Recognition
- [ ] Candlestick pattern detection (50+ patterns)
- [ ] Chart pattern detection (head & shoulders, triangles, etc.)
- [ ] Support/resistance level detection (ML-based)
- [ ] Trend line detection (automatic)

#### ML Pipeline Integration
- [ ] PyTorch DataLoader for chart images
- [ ] TensorFlow Dataset integration
- [ ] Labeled dataset generation (supervised learning)
- [ ] Feature extraction API

#### Analysis Tools
- [ ] Similarity search (find similar chart patterns)
- [ ] Correlation analysis (inter-symbol)
- [ ] Volatility forecasting

---

## v1.0.0 (2027) - Production-Grade Platform

**Focus**: Enterprise features and stability

### Planned Features

#### Enterprise Features
- [ ] High-availability architecture
- [ ] Distributed computing support (Dask, Ray)
- [ ] Kubernetes deployment
- [ ] REST API server
- [ ] Authentication & authorization

#### Advanced Analytics
- [ ] Portfolio analysis
- [ ] Risk metrics (VaR, CVaR, Sharpe, Sortino)
- [ ] Multi-asset correlation matrices
- [ ] Event study analysis

#### Platform Integration
- [ ] Jupyter Notebook widgets (interactive charts)
- [ ] Streamlit components
- [ ] Dash/Plotly integration
- [ ] VS Code extension

---

## Long-Term Vision (2028+)

### Potential Features

#### WebAssembly Support
- [ ] Browser-based rendering (no Python required)
- [ ] Client-side indicator calculation
- [ ] Embedded charts in web apps

#### 3D Visualization
- [ ] 3D surface plots (price × time × volume)
- [ ] VR/AR chart exploration
- [ ] Multi-dimensional visualization

#### AI-Powered Features
- [ ] Natural language queries ("Show me stocks breaking out")
- [ ] Automated trading signal generation
- [ ] Market regime classification

---

## Community Input Welcome!

We value community feedback. To suggest features or vote on priorities:

1. **GitHub Issues**: [github.com/kimasplund/kimsfinance/issues](https://github.com/kimasplund/kimsfinance/issues)
2. **Discussions**: [github.com/kimasplund/kimsfinance/discussions](https://github.com/kimasplund/kimsfinance/discussions)
3. **Email**: hello@asplund.kim

### How to Request Features

When requesting a feature, please include:
- **Use case**: Why do you need this feature?
- **Priority**: How important is this to your workflow?
- **Alternatives**: What workarounds are you currently using?
- **References**: Links to documentation or examples

---

## Contributing

Want to help build these features? We welcome contributions!

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Disclaimer

This roadmap is subject to change based on:
- Community feedback and demand
- Technical feasibility
- Resource availability
- Market conditions

Features may be added, removed, or rescheduled as priorities evolve.

---

**Last Updated**: 2025-10-23
**Version**: v0.1.0 Beta
