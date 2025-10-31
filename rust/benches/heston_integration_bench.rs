//! Performance Benchmarks for Heston-Backtest Integration
//!
//! Validates performance targets:
//! - Phase 0 (Heston pricing): <20ms for 1000 options
//! - Phase 1 (Indicators): <50ms for 1000 strategies × 10K candles
//! - Phase 2 (Signals): <30ms for 1000 strategies
//! - Phase 3 (Execution): <100ms for 1000 strategies
//! - Phase 4 (Metrics): <10ms for 1000 strategies
//! - Total pipeline: <250ms for 1000 strategies × 10K candles
//! - GPU memory: <1GB VRAM

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kimsfinance_core::backtest::batch::{BatchBacktestSweep, ExecutionMode, StrategyType};
use kimsfinance_core::backtest::engine::BacktestConfig;
use kimsfinance_core::gpu::{GpuDevice, HestonGpuPricer};
use kimsfinance_core::quantitative::heston::{GreeksGpuCalculator, HestonParams, OptionType, OptionQuote};
use std::sync::Arc;

// Test data module
mod test_data {
    use chrono::Utc;
    use kimsfinance_core::quantitative::heston::{HestonParams, OptionQuote, OptionType};
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    pub fn generate_btc_ohlcv(
        num_candles: usize,
        volatility: f64,
        seed: u64,
    ) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut timestamps = Vec::with_capacity(num_candles);
        let mut open = Vec::with_capacity(num_candles);
        let mut high = Vec::with_capacity(num_candles);
        let mut low = Vec::with_capacity(num_candles);
        let mut close = Vec::with_capacity(num_candles);
        let mut volume = Vec::with_capacity(num_candles);

        let start_time = Utc::now().timestamp() - (num_candles as i64 * 60);
        let mut current_price = 50000.0;

        for i in 0..num_candles {
            timestamps.push(start_time + i as i64 * 60);
            let dt = 1.0 / 525600.0;
            let vol_term = volatility * current_price * dt.sqrt() * rng.sample(rand_distr::StandardNormal);

            let open_price = current_price;
            current_price += vol_term;
            let close_price = current_price;

            let range = volatility * current_price * dt.sqrt();
            let high_price = current_price.max(open_price) + range * rng.gen::<f64>();
            let low_price = current_price.min(open_price) - range * rng.gen::<f64>();

            open.push(open_price);
            high.push(high_price);
            low.push(low_price);
            close.push(close_price);
            volume.push(100.0 * (1.0 + (high_price - low_price) / close_price));
        }

        (timestamps, open, high, low, close, volume)
    }

    pub fn generate_options_chain(
        spot_price: f64,
        num_strikes: usize,
        expiry_days: i64,
    ) -> Vec<OptionQuote> {
        let now = Utc::now().timestamp();
        let expiration = now + (expiry_days * 24 * 3600);
        let mut options = Vec::with_capacity(num_strikes * 2);

        let strike_min = spot_price * 0.80;
        let strike_max = spot_price * 1.20;
        let strike_step = (strike_max - strike_min) / (num_strikes - 1) as f64;

        for i in 0..num_strikes {
            let strike = strike_min + i as f64 * strike_step;
            options.push(OptionQuote {
                underlying: "BTC".to_string(),
                strike,
                expiration,
                option_type: OptionType::Call,
                spot_price,
                risk_free_rate: 0.05,
                bid: None,
                ask: None,
                last: None,
                implied_vol: Some(0.5),
                volume: 100.0,
                open_interest: 500.0,
                greeks: None,
            });
            options.push(OptionQuote {
                underlying: "BTC".to_string(),
                strike,
                expiration,
                option_type: OptionType::Put,
                spot_price,
                risk_free_rate: 0.05,
                bid: None,
                ask: None,
                last: None,
                implied_vol: Some(0.5),
                volume: 100.0,
                open_interest: 500.0,
                greeks: None,
            });
        }

        options
    }

    pub fn test_heston_params() -> HestonParams {
        HestonParams::new(2.0, 0.09, 0.3, -0.7, 0.09).expect("Valid Heston params")
    }

    pub fn generate_strategy_params(num_combinations: usize) -> Vec<Vec<f64>> {
        (0..num_combinations)
            .map(|i| vec![0.05 + (i as f64 * 0.001), 0.10 + (i as f64 * 0.001)])
            .collect()
    }
}

use test_data::*;

// ========== Phase 0: Heston Pricing Benchmarks ==========

fn bench_phase0_heston_pricing(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase0_heston_pricing");

    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let mut pricer = HestonGpuPricer::new(device).expect("Pricer creation failed");
    let params = test_heston_params();

    for num_options in [10, 100, 500, 1000].iter() {
        let options = generate_options_chain(50000.0, num_options / 2, 30);

        group.throughput(Throughput::Elements(*num_options as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_options),
            num_options,
            |b, _| {
                b.iter(|| {
                    let prices = pricer.price_options(&params, &options).expect("Pricing failed");
                    black_box(prices)
                });
            },
        );
    }

    group.finish();
}

// ========== Phase 0b: Greeks Calculation Benchmarks ==========

fn bench_phase0_greeks_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase0_greeks_calculation");

    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let calculator = GreeksGpuCalculator::new(device).expect("Calculator creation failed");
    let params = test_heston_params();

    for num_options in [10, 100, 500, 1000].iter() {
        let options = generate_options_chain(50000.0, num_options / 2, 30);

        group.throughput(Throughput::Elements(*num_options as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_options),
            num_options,
            |b, _| {
                b.iter(|| {
                    let greeks = calculator.calculate_batch(&params, &options).expect("Greeks failed");
                    black_box(greeks)
                });
            },
        );
    }

    group.finish();
}

// ========== Full Pipeline Benchmarks ==========

fn bench_full_pipeline_options_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline_options");
    group.sample_size(10); // Reduce samples for slow benchmarks

    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let params_heston = test_heston_params();

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
    };

    // Test scaling: 10, 100, 1000 strategies
    for num_strategies in [10, 100, 1000].iter() {
        let (timestamps, open, high, low, close, volume) = generate_btc_ohlcv(10_000, 0.3, 42);
        let options = generate_options_chain(close[close.len() - 1], 50, 30);
        let params_strategy = generate_strategy_params(*num_strategies);

        group.throughput(Throughput::Elements(*num_strategies as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_strategies),
            num_strategies,
            |b, _| {
                b.iter(|| {
                    let results = BatchBacktestSweep::new(device.clone())
                        .strategy_type(StrategyType::LongStraddle)
                        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                        .parameters_batch(&params_strategy)
                        .config(config.clone())
                        .heston_params(params_heston.clone())
                        .options_data(options.clone())
                        .execute()
                        .expect("Backtest failed");
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

fn bench_full_pipeline_equity_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline_equity");
    group.sample_size(10);

    let device = Arc::new(GpuDevice::new().expect("GPU required"));

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
    };

    for num_strategies in [10, 100, 1000].iter() {
        let (timestamps, open, high, low, close, volume) = generate_btc_ohlcv(10_000, 0.3, 42);
        let params_strategy: Vec<Vec<f64>> = (0..*num_strategies)
            .map(|i| vec![14.0, 20.0 + i as f64 * 0.1, 70.0 + i as f64 * 0.1])
            .collect();

        group.throughput(Throughput::Elements(*num_strategies as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_strategies),
            num_strategies,
            |b, _| {
                b.iter(|| {
                    let results = BatchBacktestSweep::new(device.clone())
                        .strategy_type(StrategyType::RsiCrossover)
                        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                        .parameters_batch(&params_strategy)
                        .config(config.clone())
                        .execute()
                        .expect("Backtest failed");
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// ========== Execution Mode Comparison ==========

fn bench_execution_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_modes");
    group.sample_size(10);

    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let params_heston = test_heston_params();

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
    };

    let (timestamps, open, high, low, close, volume) = generate_btc_ohlcv(5_000, 0.3, 42);
    let options = generate_options_chain(close[close.len() - 1], 20, 30);
    let params_strategy = generate_strategy_params(100);

    for mode in [
        ExecutionMode::Traditional,
        ExecutionMode::Fused,
        ExecutionMode::Async,
        ExecutionMode::Auto,
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", mode)),
            mode,
            |b, &mode| {
                b.iter(|| {
                    let results = BatchBacktestSweep::new(device.clone())
                        .strategy_type(StrategyType::LongStraddle)
                        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                        .parameters_batch(&params_strategy)
                        .config(config.clone())
                        .heston_params(params_heston.clone())
                        .options_data(options.clone())
                        .execution_mode(*mode)
                        .execute()
                        .expect("Backtest failed");
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// ========== Strategy Comparison ==========

fn bench_all_options_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("options_strategies");
    group.sample_size(10);

    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let params_heston = test_heston_params();

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
    };

    let (timestamps, open, high, low, close, volume) = generate_btc_ohlcv(1_000, 0.3, 42);
    let options = generate_options_chain(close[close.len() - 1], 20, 30);
    let params_strategy = generate_strategy_params(10);

    let strategies = vec![
        StrategyType::LongStraddle,
        StrategyType::ShortStraddle,
        StrategyType::CoveredCall,
        StrategyType::IronCondor,
        StrategyType::DeltaNeutral,
        StrategyType::VolatilityArbitrage,
    ];

    for strategy in strategies {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", strategy)),
            &strategy,
            |b, &strategy| {
                b.iter(|| {
                    let results = BatchBacktestSweep::new(device.clone())
                        .strategy_type(strategy)
                        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                        .parameters_batch(&params_strategy)
                        .config(config.clone())
                        .heston_params(params_heston.clone())
                        .options_data(options.clone())
                        .execute()
                        .expect("Backtest failed");
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// ========== Scaling Benchmarks ==========

fn bench_data_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_size_scaling");
    group.sample_size(10);

    let device = Arc::new(GpuDevice::new().expect("GPU required"));
    let params_heston = test_heston_params();

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        trading_fee: 0.001,
        slippage: 0.0005,
    };

    let params_strategy = generate_strategy_params(100);

    for num_candles in [1_000, 5_000, 10_000].iter() {
        let (timestamps, open, high, low, close, volume) = generate_btc_ohlcv(*num_candles, 0.3, 42);
        let options = generate_options_chain(close[close.len() - 1], 20, 30);

        group.throughput(Throughput::Elements(*num_candles as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_candles),
            num_candles,
            |b, _| {
                b.iter(|| {
                    let results = BatchBacktestSweep::new(device.clone())
                        .strategy_type(StrategyType::LongStraddle)
                        .data_ohlcv(&timestamps, &open, &high, &low, &close, &volume)
                        .parameters_batch(&params_strategy)
                        .config(config.clone())
                        .heston_params(params_heston.clone())
                        .options_data(options.clone())
                        .execute()
                        .expect("Backtest failed");
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_phase0_heston_pricing,
    bench_phase0_greeks_calculation,
    bench_full_pipeline_options_strategy,
    bench_full_pipeline_equity_strategy,
    bench_execution_modes,
    bench_all_options_strategies,
    bench_data_size_scaling,
);

criterion_main!(benches);
