use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::indicators::*;
use ndarray::{Array1, arr1};

fn generate_ohlcv(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let high: Array1<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.1).collect();
    let low: Array1<f64> = (0..n).map(|i| 95.0 + (i as f64) * 0.1).collect();
    let close: Array1<f64> = (0..n).map(|i| 98.0 + (i as f64) * 0.1).collect();
    let volume: Array1<f64> = (0..n).map(|i| 1000.0 + (i as f64) * 10.0).collect();
    (high, low, close, volume)
}

fn bench_obv(c: &mut Criterion) {
    let mut group = c.benchmark_group("OBV");

    for size in [100, 500, 1000, 5000].iter() {
        let (_h, _l, close, volume) = generate_ohlcv(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let obv = OBV::new();
            b.iter(|| {
                black_box(
                    obv.calculate_with_volume(black_box(close.view()), black_box(volume.view()))
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

fn bench_vwap(c: &mut Criterion) {
    let mut group = c.benchmark_group("VWAP");

    for size in [100, 500, 1000, 5000].iter() {
        let (high, low, close, volume) = generate_ohlcv(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let vwap = VWAP::new();
            b.iter(|| {
                black_box(
                    vwap.calculate_hlcv(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                        black_box(volume.view()),
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

fn bench_cmf(c: &mut Criterion) {
    let mut group = c.benchmark_group("CMF");

    for size in [100, 500, 1000, 5000].iter() {
        let (high, low, close, volume) = generate_ohlcv(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let cmf = CMF::new(20).unwrap();
            b.iter(|| {
                black_box(
                    cmf.calculate_hlcv(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                        black_box(volume.view()),
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

fn bench_volume_profile(c: &mut Criterion) {
    let mut group = c.benchmark_group("VolumeProfile");

    // Test both sequential and parallel paths
    for size in [100, 500, 1000, 2000, 5000].iter() {
        let (high, low, close, volume) = generate_ohlcv(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let vp = VolumeProfile::new(50).unwrap();
            b.iter(|| {
                black_box(
                    vp.calculate_hlcv(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                        black_box(volume.view()),
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

fn bench_volume_profile_poc(c: &mut Criterion) {
    let mut group = c.benchmark_group("VolumeProfile_POC");

    for size in [100, 1000, 5000].iter() {
        let (high, low, close, volume) = generate_ohlcv(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let vp = VolumeProfile::new(50).unwrap();
            b.iter(|| {
                black_box(
                    vp.point_of_control(
                        black_box(high.view()),
                        black_box(low.view()),
                        black_box(close.view()),
                        black_box(volume.view()),
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_obv,
    bench_vwap,
    bench_cmf,
    bench_volume_profile,
    bench_volume_profile_poc
);
criterion_main!(benches);
