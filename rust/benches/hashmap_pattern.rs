//! Micro-benchmark to profile HashMap allocation patterns
//!
//! This benchmark investigates the 21.5% slowdown from "zero-allocation" optimization.
//!
//! Hypothesis:
//! - HashMap::clear() + reuse may have overhead from:
//!   1. clear() operation itself (O(capacity) traversal)
//!   2. String key cloning (not eliminated)
//!   3. Cache effects (old allocations vs fresh allocations)
//!   4. Hash table resizing/rehashing
//!
//! Run with: `cargo bench --bench hashmap_pattern`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;

/// Simulate the backtest engine pattern: HashMap<String, f64> with indicator data
fn generate_indicator_data(num_indicators: usize) -> HashMap<String, Vec<f64>> {
    let mut data = HashMap::new();
    for i in 0..num_indicators {
        let key = format!("RSI_{}", i);
        let values: Vec<f64> = (0..1000).map(|j| (i + j) as f64).collect();
        data.insert(key, values);
    }
    data
}

/// Pattern 1: Allocate new HashMap each iteration (baseline - what we had before)
fn allocate_each_iteration(indicator_data: &HashMap<String, Vec<f64>>, n_candles: usize) {
    for i in 0..n_candles {
        let mut bar_indicators = HashMap::new();
        for (key, values) in indicator_data {
            bar_indicators.insert(key.clone(), values[i]);
        }
        black_box(&bar_indicators);
    }
}

/// Pattern 2: Reuse HashMap with clear() (current "optimization")
fn reuse_with_clear(indicator_data: &HashMap<String, Vec<f64>>, n_candles: usize) {
    let mut bar_indicators = HashMap::with_capacity(indicator_data.len());
    for i in 0..n_candles {
        bar_indicators.clear();
        for (key, values) in indicator_data {
            bar_indicators.insert(key.clone(), values[i]);
        }
        black_box(&bar_indicators);
    }
}

/// Pattern 3: Use SmallVec (stack allocation for small sizes)
fn smallvec_pattern(indicator_data: &HashMap<String, Vec<f64>>, n_candles: usize) {
    use smallvec::SmallVec;

    for i in 0..n_candles {
        let mut bar_indicators: SmallVec<[(String, f64); 16]> = SmallVec::new();
        for (key, values) in indicator_data {
            bar_indicators.push((key.clone(), values[i]));
        }
        black_box(&bar_indicators);
    }
}

/// Pattern 4: Pre-allocated Vec (if we don't need HashMap lookup)
fn vec_pattern(indicator_data: &HashMap<String, Vec<f64>>, n_candles: usize) {
    for i in 0..n_candles {
        let mut bar_indicators = Vec::with_capacity(indicator_data.len());
        for (key, values) in indicator_data {
            bar_indicators.push((key.clone(), values[i]));
        }
        black_box(&bar_indicators);
    }
}

/// Pattern 5: Reuse Vec with clear()
fn reuse_vec_with_clear(indicator_data: &HashMap<String, Vec<f64>>, n_candles: usize) {
    let mut bar_indicators = Vec::with_capacity(indicator_data.len());
    for i in 0..n_candles {
        bar_indicators.clear();
        for (key, values) in indicator_data {
            bar_indicators.push((key.clone(), values[i]));
        }
        black_box(&bar_indicators);
    }
}

/// Pattern 6: Pre-clone keys to avoid allocation in hot loop
fn pre_cloned_keys(indicator_data: &HashMap<String, Vec<f64>>, n_candles: usize) {
    // Pre-clone keys once
    let keys: Vec<String> = indicator_data.keys().cloned().collect();
    let values_vec: Vec<&Vec<f64>> = indicator_data.values().collect();

    let mut bar_indicators = HashMap::with_capacity(keys.len());
    for i in 0..n_candles {
        bar_indicators.clear();
        for (j, key) in keys.iter().enumerate() {
            bar_indicators.insert(key.clone(), values_vec[j][i]);
        }
        black_box(&bar_indicators);
    }
}

/// Pattern 7: Use &str references instead of owned String keys
fn str_references(indicator_data: &HashMap<String, Vec<f64>>, n_candles: usize) {
    for i in 0..n_candles {
        let mut bar_indicators: HashMap<&str, f64> = HashMap::with_capacity(indicator_data.len());
        for (key, values) in indicator_data {
            bar_indicators.insert(key.as_str(), values[i]);
        }
        black_box(&bar_indicators);
    }
}

fn bench_hashmap_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashmap_patterns");

    // Test with different numbers of indicators (typical strategies use 1-5)
    for num_indicators in [1, 3, 5].iter() {
        let indicator_data = generate_indicator_data(*num_indicators);
        let n_candles = 1000; // Typical backtest size from benchmark

        group.bench_with_input(
            BenchmarkId::new("allocate_each_iteration", num_indicators),
            num_indicators,
            |b, _| {
                b.iter(|| allocate_each_iteration(black_box(&indicator_data), black_box(n_candles)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("reuse_with_clear", num_indicators),
            num_indicators,
            |b, _| {
                b.iter(|| reuse_with_clear(black_box(&indicator_data), black_box(n_candles)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("smallvec_pattern", num_indicators),
            num_indicators,
            |b, _| {
                b.iter(|| smallvec_pattern(black_box(&indicator_data), black_box(n_candles)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec_pattern", num_indicators),
            num_indicators,
            |b, _| {
                b.iter(|| vec_pattern(black_box(&indicator_data), black_box(n_candles)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("reuse_vec_with_clear", num_indicators),
            num_indicators,
            |b, _| {
                b.iter(|| reuse_vec_with_clear(black_box(&indicator_data), black_box(n_candles)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pre_cloned_keys", num_indicators),
            num_indicators,
            |b, _| {
                b.iter(|| pre_cloned_keys(black_box(&indicator_data), black_box(n_candles)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("str_references", num_indicators),
            num_indicators,
            |b, _| {
                b.iter(|| str_references(black_box(&indicator_data), black_box(n_candles)))
            },
        );
    }

    group.finish();
}

/// Benchmark HashMap::clear() overhead vs new()
fn bench_clear_vs_new(c: &mut Criterion) {
    let mut group = c.benchmark_group("clear_vs_new");

    for size in [1, 5, 10, 20].iter() {
        // Benchmark new() each iteration
        group.bench_with_input(BenchmarkId::new("new_each_iter", size), size, |b, &size| {
            let data: Vec<(String, f64)> = (0..size)
                .map(|i| (format!("key_{}", i), i as f64))
                .collect();

            b.iter(|| {
                for _ in 0..100 {
                    let mut map = HashMap::new();
                    for (k, v) in &data {
                        map.insert(k.clone(), *v);
                    }
                    black_box(map);
                }
            });
        });

        // Benchmark clear() + reuse
        group.bench_with_input(BenchmarkId::new("clear_reuse", size), size, |b, &size| {
            let data: Vec<(String, f64)> = (0..size)
                .map(|i| (format!("key_{}", i), i as f64))
                .collect();

            b.iter(|| {
                let mut map = HashMap::with_capacity(size);
                for _ in 0..100 {
                    map.clear();
                    for (k, v) in &data {
                        map.insert(k.clone(), *v);
                    }
                    black_box(&map);
                }
            });
        });

        // Benchmark with_capacity() each iteration
        group.bench_with_input(BenchmarkId::new("with_capacity_each_iter", size), size, |b, &size| {
            let data: Vec<(String, f64)> = (0..size)
                .map(|i| (format!("key_{}", i), i as f64))
                .collect();

            b.iter(|| {
                for _ in 0..100 {
                    let mut map = HashMap::with_capacity(size);
                    for (k, v) in &data {
                        map.insert(k.clone(), *v);
                    }
                    black_box(map);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark cache effects: contiguous vs scattered allocations
fn bench_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effects");

    // Simulate 1000 iterations
    let n = 1000;
    let size = 5; // 5 indicators typical

    // Fresh allocations (scattered in memory)
    group.bench_function("scattered_allocations", |b| {
        let data: Vec<(String, f64)> = (0..size)
            .map(|i| (format!("key_{}", i), i as f64))
            .collect();

        b.iter(|| {
            for _ in 0..n {
                let mut map = HashMap::with_capacity(size);
                for (k, v) in &data {
                    map.insert(k.clone(), *v);
                }
                black_box(map);
            }
        });
    });

    // Reused allocation (same memory location)
    group.bench_function("reused_allocation", |b| {
        let data: Vec<(String, f64)> = (0..size)
            .map(|i| (format!("key_{}", i), i as f64))
            .collect();

        b.iter(|| {
            let mut map = HashMap::with_capacity(size);
            for _ in 0..n {
                map.clear();
                for (k, v) in &data {
                    map.insert(k.clone(), *v);
                }
                black_box(&map);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hashmap_patterns,
    bench_clear_vs_new,
    bench_cache_effects
);
criterion_main!(benches);
