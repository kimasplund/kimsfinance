use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kimsfinance_core::indicators::*;
use ndarray::Array1;

fn generate_hlc_data(size: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut high = Array1::zeros(size);
    let mut low = Array1::zeros(size);
    let mut close = Array1::zeros(size);

    let mut base = 100.0;
    for i in 0..size {
        let volatility = 2.0 + (i as f64 * 0.01).sin() * 1.5;
        high[i] = base + volatility;
        low[i] = base - volatility;
        close[i] = base + volatility * (0.5 - (i as f64 * 0.02).cos() * 0.5);
        base += (i as f64 * 0.1).sin() * 0.5;
    }

    (high, low, close)
}

fn generate_price_data(size: usize) -> Array1<f64> {
    let mut prices = Array1::zeros(size);
    let mut base = 100.0;

    for i in 0..size {
        prices[i] = base;
        base += (i as f64 * 0.1).sin() * 0.5;
    }

    prices
}

fn bench_atr(c: &mut Criterion) {
    let mut group = c.benchmark_group("ATR");

    for size in [100, 500, 1000].iter() {
        let (high, low, close) = generate_hlc_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let atr = ATR::new(14).unwrap();
            b.iter(|| {
                atr.calculate_hlc(
                    black_box(high.view()),
                    black_box(low.view()),
                    black_box(close.view()),
                )
            });
        });
    }

    group.finish();
}

fn bench_bollinger_bands(c: &mut Criterion) {
    let mut group = c.benchmark_group("BollingerBands");

    for size in [100, 500, 1000].iter() {
        let prices = generate_price_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let bb = BollingerBands::new(20, 2.0).unwrap();
            b.iter(|| {
                bb.calculate_multi(black_box(prices.view()))
            });
        });
    }

    group.finish();
}

fn bench_keltner_channels(c: &mut Criterion) {
    let mut group = c.benchmark_group("KeltnerChannels");

    for size in [100, 500, 1000].iter() {
        let (high, low, close) = generate_hlc_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let kc = KeltnerChannels::new(20, 10, 2.0).unwrap();
            b.iter(|| {
                kc.calculate_hlc(
                    black_box(high.view()),
                    black_box(low.view()),
                    black_box(close.view()),
                )
            });
        });
    }

    group.finish();
}

fn bench_donchian_channels(c: &mut Criterion) {
    let mut group = c.benchmark_group("DonchianChannels");

    for size in [100, 500, 1000].iter() {
        let (high, low, _close) = generate_hlc_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let dc = DonchianChannels::new(20).unwrap();
            b.iter(|| {
                dc.calculate_hl(
                    black_box(high.view()),
                    black_box(low.view()),
                )
            });
        });
    }

    group.finish();
}

fn bench_elder_ray(c: &mut Criterion) {
    let mut group = c.benchmark_group("ElderRay");

    for size in [100, 500, 1000].iter() {
        let (high, low, close) = generate_hlc_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            let er = ElderRay::new(13).unwrap();
            b.iter(|| {
                er.calculate_hlc(
                    black_box(high.view()),
                    black_box(low.view()),
                    black_box(close.view()),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_atr,
    bench_bollinger_bands,
    bench_keltner_channels,
    bench_donchian_channels,
    bench_elder_ray
);
criterion_main!(benches);
