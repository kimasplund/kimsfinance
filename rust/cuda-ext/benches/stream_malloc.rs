//! Criterion Benchmark: Stream-Ordered Malloc vs Standard Allocation
//!
//! This benchmark uses Criterion for statistical analysis of allocation performance.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench stream_malloc
//! ```
//!
//! # Expected Results
//!
//! - Standard cudaMalloc: 10-15ms per allocation
//! - cudaMallocAsync: 5-10ms per allocation
//! - Speedup: 1.2-1.5x

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cudarc::driver::CudaContext;
use kimsfinance_cuda_ext::stream_malloc::StreamOrderedAllocator;
use std::sync::Arc;

fn bench_standard_alloc(c: &mut Criterion) {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let stream = context.default_stream();

    let mut group = c.benchmark_group("allocation");

    for size in [1024, 1024 * 1024, 10 * 1024 * 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("standard_cudaMalloc", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let slice = stream.alloc_zeros::<u8>(size).expect("Allocation failed");
                    black_box(slice);
                });
            },
        );
    }

    group.finish();
}

fn bench_async_alloc(c: &mut Criterion) {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    let mut group = c.benchmark_group("allocation");

    for size in [1024, 1024 * 1024, 10 * 1024 * 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("cudaMallocAsync", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let ptr = unsafe {
                        allocator
                            .alloc_async(size, stream.clone())
                            .expect("Allocation failed")
                    };
                    black_box(ptr);
                    unsafe {
                        allocator
                            .free_async(ptr, stream.clone())
                            .expect("Free failed");
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_standard_alloc, bench_async_alloc);
criterion_main!(benches);
