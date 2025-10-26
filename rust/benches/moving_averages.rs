use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::indicators::{DEMA, EMA, HMA, Indicator, SMA, TEMA, VWMA, WMA};
use ndarray::Array1;

fn benchmark_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group("SMA");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let prices = Array1::linspace(100.0, 150.0, *size);
        let sma = SMA::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| sma.calculate(black_box(prices.view())).unwrap());
        });
    }

    group.finish();
}

fn benchmark_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group("EMA");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let prices = Array1::linspace(100.0, 150.0, *size);
        let ema = EMA::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| ema.calculate(black_box(prices.view())).unwrap());
        });
    }

    group.finish();
}

fn benchmark_wma(c: &mut Criterion) {
    let mut group = c.benchmark_group("WMA");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let prices = Array1::linspace(100.0, 150.0, *size);
        let wma = WMA::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| wma.calculate(black_box(prices.view())).unwrap());
        });
    }

    group.finish();
}

fn benchmark_vwma(c: &mut Criterion) {
    let mut group = c.benchmark_group("VWMA");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let prices = Array1::linspace(100.0, 150.0, *size);
        let volumes = Array1::linspace(1000.0, 2000.0, *size);
        let vwma = VWMA::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                vwma.calculate_with_volume(black_box(prices.view()), black_box(volumes.view()))
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn benchmark_dema(c: &mut Criterion) {
    let mut group = c.benchmark_group("DEMA");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let prices = Array1::linspace(100.0, 150.0, *size);
        let dema = DEMA::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| dema.calculate(black_box(prices.view())).unwrap());
        });
    }

    group.finish();
}

fn benchmark_tema(c: &mut Criterion) {
    let mut group = c.benchmark_group("TEMA");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let prices = Array1::linspace(100.0, 150.0, *size);
        let tema = TEMA::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| tema.calculate(black_box(prices.view())).unwrap());
        });
    }

    group.finish();
}

fn benchmark_hma(c: &mut Criterion) {
    let mut group = c.benchmark_group("HMA");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let prices = Array1::linspace(100.0, 150.0, *size);
        let hma = HMA::new(14).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| hma.calculate(black_box(prices.view())).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_sma,
    benchmark_ema,
    benchmark_wma,
    benchmark_vwma,
    benchmark_dema,
    benchmark_tema,
    benchmark_hma
);
criterion_main!(benches);
