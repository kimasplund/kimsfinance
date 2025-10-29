//! Benchmark for IncompleteCandle performance
//!
//! Validates that update() operations are <10ns in the hot path

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kimsfinance_core::binance::{IncompleteCandle, Trade};

fn make_trade(price: f64, quantity: f64, timestamp: i64) -> Trade {
    Trade {
        trade_id: 0,
        price,
        quantity,
        quote_quantity: price * quantity,
        timestamp_ms: timestamp,
        is_buyer_maker: false,
    }
}

fn bench_incomplete_candle_new(c: &mut Criterion) {
    let trade = make_trade(100.0, 1.0, 1000);

    c.bench_function("incomplete_candle_new", |b| {
        b.iter(|| {
            black_box(IncompleteCandle::new(&trade, 0))
        })
    });
}

fn bench_incomplete_candle_update(c: &mut Criterion) {
    let trade1 = make_trade(100.0, 1.0, 1000);
    let trade2 = make_trade(105.0, 2.0, 2000);

    c.bench_function("incomplete_candle_update", |b| {
        b.iter(|| {
            let mut candle = IncompleteCandle::new(&trade1, 0);
            black_box(candle.update(&trade2));
            black_box(candle)
        })
    });
}

fn bench_incomplete_candle_update_only(c: &mut Criterion) {
    // Benchmark JUST the update() call (hot path)
    let trade1 = make_trade(100.0, 1.0, 1000);
    let trade2 = make_trade(105.0, 2.0, 2000);
    let mut candle = IncompleteCandle::new(&trade1, 0);

    c.bench_function("incomplete_candle_update_only", |b| {
        b.iter(|| {
            black_box(candle.update(&trade2));
        })
    });
}

fn bench_incomplete_candle_complete(c: &mut Criterion) {
    let trade = make_trade(100.0, 1.0, 1000);

    c.bench_function("incomplete_candle_complete", |b| {
        b.iter(|| {
            let candle = IncompleteCandle::new(&trade, 0);
            black_box(candle.complete())
        })
    });
}

fn bench_incomplete_candle_realistic(c: &mut Criterion) {
    // Realistic scenario: Build a candle with 100 trades
    let mut trades = Vec::new();
    for i in 0..100 {
        trades.push(make_trade(100.0 + i as f64, 1.0, i * 100));
    }

    c.bench_function("incomplete_candle_100_trades", |b| {
        b.iter(|| {
            let mut candle = IncompleteCandle::new(&trades[0], 0);
            for trade in &trades[1..] {
                candle.update(trade);
            }
            black_box(candle.complete())
        })
    });
}

fn bench_comparison_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("incomplete_candle_batch");

    for batch_size in [10, 50, 100, 500, 1000].iter() {
        let mut trades = Vec::new();
        for i in 0..*batch_size {
            trades.push(make_trade(100.0 + i as f64, 1.0, i * 100));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let mut candle = IncompleteCandle::new(&trades[0], 0);
                    for trade in &trades[1..] {
                        candle.update(trade);
                    }
                    black_box(candle.complete())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_incomplete_candle_new,
    bench_incomplete_candle_update,
    bench_incomplete_candle_update_only,
    bench_incomplete_candle_complete,
    bench_incomplete_candle_realistic,
    bench_comparison_batch_sizes,
);
criterion_main!(benches);
