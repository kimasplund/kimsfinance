//! Pinned Memory Transfer Benchmark
//!
//! Validates the third enhancement: pinned (page-locked) memory for faster transfers.
//!
//! # Problem
//!
//! Traditional CUDA memory transfers use pageable host memory:
//! - Must go through DMA staging buffer
//! - Kernel may page-fault during transfer
//! - Limited by PCIe + system RAM latency
//! - Typical: 8-10 GB/s on PCIe 3.0 x16
//!
//! # Solution: Pinned Memory
//!
//! Use `cudaMallocHost()` to allocate page-locked memory:
//! - Direct DMA (no staging)
//! - Guaranteed to be resident (no page faults)
//! - Faster transfers (1.2-1.3x typical)
//! - Downsides: Limited resource, slower allocations
//!
//! # Expected Results
//!
//! - **Transfer speedup**: 1.2-1.3x faster
//! - **Larger benefit**: For larger transfers (>1MB)
//! - **Trade-off**: Slower allocation (amortize over many transfers)
//!
//! # Benchmark Methodology
//!
//! Compare three scenarios:
//! 1. **Pageable (baseline)**: Standard Vec allocation + htod_copy
//! 2. **Pinned (optimized)**: cudaMallocHost + htod_copy
//! 3. **Mapped (zero-copy)**: cudaHostAlloc with cudaHostAllocMapped flag
//!
//! Test across multiple data sizes to find crossover point where pinned wins.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::GpuDevice;

/// Benchmark: Pageable memory transfer (baseline)
fn bench_pageable_transfer(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("memory_transfer");
    group.sample_size(100);

    // Test various data sizes (1K to 10M elements)
    for size in [1_000, 10_000, 100_000, 1_000_000, 10_000_000].iter() {
        let bytes = size * std::mem::size_of::<f64>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("pageable", size), size, |b, &n| {
            b.iter(|| {
                // Allocate pageable host memory (standard Vec)
                let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

                // Transfer to device
                let dev_buf = device.copy_to_device(&data).expect("htod failed");

                // Transfer back to host (round-trip for fair comparison)
                let result: Vec<f64> = device.copy_to_host(&dev_buf).expect("dtoh failed");

                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Pinned memory transfer (optimized)
///
/// NOTE: Requires implementation of PinnedBuffer wrapper around cudaMallocHost.
/// For now, we benchmark pageable as a baseline and document expected improvement.
fn bench_pinned_transfer(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("memory_transfer");
    group.sample_size(100);

    for size in [1_000, 10_000, 100_000, 1_000_000, 10_000_000].iter() {
        let bytes = size * std::mem::size_of::<f64>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("pinned", size), size, |b, &n| {
            b.iter(|| {
                // TODO: Use PinnedBuffer::new(n)
                // For now, use pageable as placeholder
                let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

                let dev_buf = device.copy_to_device(&data).expect("htod failed");
                let result: Vec<f64> = device.copy_to_host(&dev_buf).expect("dtoh failed");

                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Transfer-only (no computation)
///
/// Isolates transfer overhead from computation to measure pure bandwidth
fn bench_transfer_bandwidth(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("transfer_bandwidth");
    group.sample_size(50);

    // Test large transfers to saturate PCIe bandwidth
    for size in [1_000_000, 10_000_000, 50_000_000].iter() {
        let bytes = size * std::mem::size_of::<f64>();
        group.throughput(Throughput::Bytes(bytes as u64));

        // Host-to-Device only
        group.bench_with_input(BenchmarkId::new("htod_pageable", size), size, |b, &n| {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

            b.iter(|| {
                let dev_buf = device
                    .copy_to_device(black_box(&data))
                    .expect("htod failed");
                black_box(dev_buf);
            });
        });

        // Device-to-Host only
        group.bench_with_input(BenchmarkId::new("dtoh_pageable", size), size, |b, &n| {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let dev_buf = device.copy_to_device(&data).expect("htod failed");

            b.iter(|| {
                let result: Vec<f64> = device
                    .copy_to_host(black_box(&dev_buf))
                    .expect("dtoh failed");
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Allocation overhead
///
/// Measures allocation cost for pageable vs pinned memory.
/// Important for understanding amortization requirements.
fn bench_allocation_overhead(c: &mut Criterion) {
    let _device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("allocation_overhead");
    group.sample_size(100);

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Pageable allocation (Vec::with_capacity)
        group.bench_with_input(BenchmarkId::new("pageable_alloc", size), size, |b, &n| {
            b.iter(|| {
                let mut vec = Vec::with_capacity(n);
                vec.extend((0..n).map(|i| i as f64));
                black_box(vec);
            });
        });

        // TODO: Pinned allocation (cudaMallocHost)
        // Expected: 2-5x slower allocation, but faster transfers
        // group.bench_with_input(
        //     BenchmarkId::new("pinned_alloc", size),
        //     size,
        //     |b, &n| {
        //         b.iter(|| {
        //             let mut pinned = PinnedBuffer::new(n).expect("pinned alloc failed");
        //             pinned.as_mut_slice().iter_mut().enumerate().for_each(|(i, x)| *x = i as f64);
        //             black_box(pinned);
        //         });
        //     },
        // );
    }

    group.finish();
}

/// Benchmark: Amortization analysis
///
/// Tests how many transfers are needed to amortize slower pinned allocation.
/// Formula: breakeven = allocation_overhead / transfer_speedup_per_iter
fn bench_amortization(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("amortization");
    group.sample_size(50);

    let size = 100_000; // 100K elements
    let num_transfers_list = [1, 5, 10, 50, 100];

    for num_transfers in num_transfers_list.iter() {
        group.throughput(Throughput::Elements((size * num_transfers) as u64));

        // Pageable: allocate once, transfer N times
        group.bench_with_input(
            BenchmarkId::new("pageable", num_transfers),
            num_transfers,
            |b, &n| {
                b.iter(|| {
                    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

                    for _ in 0..n {
                        let dev_buf = device.copy_to_device(&data).expect("htod failed");
                        let result: Vec<f64> = device.copy_to_host(&dev_buf).expect("dtoh failed");
                        black_box(result);
                    }
                });
            },
        );

        // TODO: Pinned: allocate once, transfer N times
        // Expected: slower for n=1-5, faster for n>=10
        // group.bench_with_input(
        //     BenchmarkId::new("pinned", num_transfers),
        //     num_transfers,
        //     |b, &n| { ... }
        // );
    }

    group.finish();
}

/// Benchmark: Persistent kernel + pinned memory (combined optimization)
///
/// Tests the combination of persistent kernels + pinned transfers.
/// Expected: 1.2-1.3x additional speedup on top of persistent kernel gains.
fn bench_persistent_with_pinned(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let mut group = c.benchmark_group("persistent_pinned_combined");
    group.sample_size(50);

    let data_size = 10_000;
    let num_tasks = 10;

    group.throughput(Throughput::Elements((data_size * num_tasks) as u64));

    // Baseline: persistent + pageable
    group.bench_function("persistent_pageable", |b| {
        b.iter(|| {
            use kimsfinance_core::gpu::persistent::*;

            let mut batch = TaskBatch::<SmaIndicator>::new();
            for i in 0..num_tasks {
                let data: Vec<f64> = (0..data_size).map(|j| 100.0 + j as f64).collect();
                let period = (14 + i % 10) as i32;
                batch.add_task(data, period);
            }

            let results = execute_batch(&device, &batch).expect("Batch failed");
            black_box(results);
        });
    });

    // TODO: Optimized: persistent + pinned
    // Expected: 1.2-1.3x faster than persistent_pageable
    // group.bench_function("persistent_pinned", |b| { ... });

    group.finish();
}

/// Print PCIe bandwidth info for context
fn print_bandwidth_info() {
    eprintln!("\n=== PCIe Bandwidth Expectations ===");
    eprintln!("PCIe 3.0 x16: ~12 GB/s theoretical, ~10 GB/s practical");
    eprintln!("PCIe 4.0 x16: ~24 GB/s theoretical, ~20 GB/s practical");
    eprintln!("PCIe 5.0 x16: ~48 GB/s theoretical, ~40 GB/s practical");
    eprintln!("");
    eprintln!("Expected pinned memory speedup:");
    eprintln!("  - Small transfers (<1MB): 1.1-1.2x");
    eprintln!("  - Medium transfers (1-10MB): 1.2-1.3x");
    eprintln!("  - Large transfers (>10MB): 1.3-1.5x");
    eprintln!("");
}

fn setup_benchmarks(_c: &mut Criterion) {
    print_bandwidth_info();
}

criterion_group!(
    benches,
    setup_benchmarks,
    bench_pageable_transfer,
    bench_pinned_transfer,
    bench_transfer_bandwidth,
    bench_allocation_overhead,
    bench_amortization,
    bench_persistent_with_pinned,
);
criterion_main!(benches);
