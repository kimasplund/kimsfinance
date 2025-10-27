//! Benchmark: Pinned Memory vs Pageable Memory Transfer Speed
//!
//! Validates 20-30% transfer speedup from pinned memory.
//!
//! # Expected Results
//!
//! - Pinned H2D: 20-30% faster than pageable
//! - Pinned D2H: 20-30% faster than pageable
//! - Speedup: 1.2-1.3x
//!
//! # Hardware Context
//!
//! - GPU: NVIDIA RTX 3500 Ada (12GB VRAM)
//! - PCIe: Gen 4 (theoretical 32 GB/s)
//! - Driver: CUDA 13.0

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use kimsfinance_core::gpu::{GpuDevice, persistent::pinned_memory::PinnedBuffer};

fn bench_h2d_transfers(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let sizes = vec![
        1_000,      // 1K elements (~8KB)
        10_000,     // 10K elements (~80KB)
        100_000,    // 100K elements (~800KB)
        1_000_000,  // 1M elements (~8MB)
        10_000_000, // 10M elements (~80MB)
    ];

    for &size in &sizes {
        let mut group = c.benchmark_group(format!("H2D_{}", size));

        // Pageable memory baseline
        group.bench_function(BenchmarkId::new("pageable", size), |b| {
            let data = vec![1.0f64; size];
            let mut d_buffer = device.alloc_buffer(size).expect("GPU allocation failed");

            b.iter(|| {
                device
                    .stream
                    .memcpy_htod(black_box(&data), black_box(&mut d_buffer))
                    .expect("H2D copy failed");
                device.synchronize().expect("Sync failed");
            });
        });

        // Pinned memory optimization
        group.bench_function(BenchmarkId::new("pinned", size), |b| {
            let data = vec![1.0f64; size];
            let mut pinned_buffer = PinnedBuffer::new(size).expect("Pinned allocation failed");
            pinned_buffer.copy_from_slice(&data);
            let mut d_buffer = device.alloc_buffer(size).expect("GPU allocation failed");

            b.iter(|| {
                device
                    .stream
                    .memcpy_htod(
                        black_box(pinned_buffer.as_slice()),
                        black_box(&mut d_buffer),
                    )
                    .expect("H2D copy failed");
                device.synchronize().expect("Sync failed");
            });
        });

        group.finish();
    }
}

fn bench_d2h_transfers(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let sizes = vec![
        1_000,      // 1K elements (~8KB)
        10_000,     // 10K elements (~80KB)
        100_000,    // 100K elements (~800KB)
        1_000_000,  // 1M elements (~8MB)
        10_000_000, // 10M elements (~80MB)
    ];

    for &size in &sizes {
        let mut group = c.benchmark_group(format!("D2H_{}", size));

        // Pageable memory baseline
        group.bench_function(BenchmarkId::new("pageable", size), |b| {
            let d_buffer = device
                .copy_to_device(&vec![1.0f64; size])
                .expect("H2D failed");

            b.iter(|| {
                let _result = device
                    .copy_to_host(black_box(&d_buffer))
                    .expect("D2H copy failed");
                device.synchronize().expect("Sync failed");
            });
        });

        // Pinned memory optimization
        group.bench_function(BenchmarkId::new("pinned", size), |b| {
            let d_buffer = device
                .copy_to_device(&vec![1.0f64; size])
                .expect("H2D failed");
            let mut pinned_buffer = PinnedBuffer::new(size).expect("Pinned allocation failed");

            b.iter(|| {
                device
                    .stream
                    .memcpy_dtoh(
                        black_box(&d_buffer),
                        black_box(pinned_buffer.as_mut_slice()),
                    )
                    .expect("D2H copy failed");
                device.synchronize().expect("Sync failed");
            });
        });

        group.finish();
    }
}

fn bench_roundtrip_transfers(c: &mut Criterion) {
    let device = GpuDevice::new().expect("GPU required");

    let sizes = vec![
        10_000,    // 10K elements (~80KB)
        100_000,   // 100K elements (~800KB)
        1_000_000, // 1M elements (~8MB)
    ];

    for &size in &sizes {
        let mut group = c.benchmark_group(format!("Roundtrip_{}", size));

        // Pageable memory baseline
        group.bench_function(BenchmarkId::new("pageable", size), |b| {
            let data = vec![1.0f64; size];
            let mut d_buffer = device.alloc_buffer(size).expect("GPU allocation failed");

            b.iter(|| {
                // H2D
                device
                    .stream
                    .memcpy_htod(black_box(&data), black_box(&mut d_buffer))
                    .expect("H2D copy failed");
                // D2H
                let _result = device
                    .copy_to_host(black_box(&d_buffer))
                    .expect("D2H copy failed");
                device.synchronize().expect("Sync failed");
            });
        });

        // Pinned memory optimization
        group.bench_function(BenchmarkId::new("pinned", size), |b| {
            let data = vec![1.0f64; size];
            let mut pinned_input = PinnedBuffer::new(size).expect("Pinned input allocation failed");
            let mut pinned_output =
                PinnedBuffer::new(size).expect("Pinned output allocation failed");
            pinned_input.copy_from_slice(&data);
            let mut d_buffer = device.alloc_buffer(size).expect("GPU allocation failed");

            b.iter(|| {
                // H2D
                device
                    .stream
                    .memcpy_htod(black_box(pinned_input.as_slice()), black_box(&mut d_buffer))
                    .expect("H2D copy failed");
                // D2H
                device
                    .stream
                    .memcpy_dtoh(
                        black_box(&d_buffer),
                        black_box(pinned_output.as_mut_slice()),
                    )
                    .expect("D2H copy failed");
                device.synchronize().expect("Sync failed");
            });
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_h2d_transfers,
    bench_d2h_transfers,
    bench_roundtrip_transfers
);
criterion_main!(benches);
