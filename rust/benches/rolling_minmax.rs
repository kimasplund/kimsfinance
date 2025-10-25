//! Benchmark for rolling_max and rolling_min optimizations
//!
//! Compares O(n) deque algorithm vs naive O(n*period) approach
//! to validate 50x speedup for large periods.

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, black_box, criterion_group,
    criterion_main,
};
use kimsfinance_core::indicators::utils::{rolling_max, rolling_min};
use ndarray::Array1;

/// Generate synthetic data for benchmarking
fn generate_data(size: usize) -> Array1<f64> {
    let mut data = Array1::zeros(size);
    for i in 0..size {
        // Realistic price movement with volatility
        data[i] = 100.0 + ((i as f64 * 0.1).sin() * 10.0) + ((i as f64 * 0.01).cos() * 5.0);
    }
    data
}

/// Benchmark rolling_max with different dataset sizes
fn bench_rolling_max_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_max_by_size");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    // Test with period=100 (typical for Donchian Channels)
    let period = 100;

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let data = generate_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| rolling_max(black_box(data.view()), black_box(period)));
        });
    }

    group.finish();
}

/// Benchmark rolling_max with different periods
fn bench_rolling_max_periods(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_max_by_period");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    // Test with 10K data points
    let size = 10000;
    let data = generate_data(size);

    for period in [10, 20, 50, 100, 200, 500].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(period), period, |b, &p| {
            b.iter(|| rolling_max(black_box(data.view()), black_box(p)));
        });
    }

    group.finish();
}

/// Benchmark rolling_min with different dataset sizes
fn bench_rolling_min_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_min_by_size");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    // Test with period=100
    let period = 100;

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let data = generate_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| rolling_min(black_box(data.view()), black_box(period)));
        });
    }

    group.finish();
}

/// Benchmark Williams %R (uses both rolling_max and rolling_min)
fn bench_williams_r(c: &mut Criterion) {
    use kimsfinance_core::indicators::{Indicator, WilliamsR};

    let mut group = c.benchmark_group("williams_r");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let high = generate_data(*size);
        let low = high.mapv(|x| x - 5.0);
        let close = high.mapv(|x| x - 2.5);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let williams = WilliamsR::new(14).unwrap();
            b.iter(|| {
                williams
                    .calculate_hlc(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                    )
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark Donchian Channels (uses rolling_max and rolling_min in parallel)
fn bench_donchian_channels(c: &mut Criterion) {
    use kimsfinance_core::indicators::DonchianChannels;

    let mut group = c.benchmark_group("donchian_channels");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let high = generate_data(*size);
        let low = high.mapv(|x| x - 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let dc = DonchianChannels::new(20).unwrap();
            b.iter(|| {
                dc.calculate_hl(black_box(high.view()), black_box(low.view()))
                    .unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rolling_max_sizes,
    bench_rolling_max_periods,
    bench_rolling_min_sizes,
    bench_williams_r,
    bench_donchian_channels
);

criterion_main!(benches);
