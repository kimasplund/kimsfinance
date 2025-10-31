//! Basic Stream-Ordered Allocation Example
//!
//! This example demonstrates the fundamental usage of stream-ordered memory allocation
//! and compares it with traditional cudaMalloc.
//!
//! # What You'll Learn
//!
//! 1. How to create a StreamOrderedAllocator
//! 2. How to allocate memory asynchronously on a stream
//! 3. How to properly free memory on the same stream
//! 4. Performance comparison with traditional allocation
//!
//! # Expected Output
//!
//! - Traditional allocation time: ~5-15ms for 1000 allocations
//! - Stream-ordered allocation time: ~3-10ms for 1000 allocations
//! - Speedup: 1.2-1.5x
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example stream_allocation_basics
//! ```

use cudarc::driver::CudaContext;
use kimsfinance_cuda_ext::stream_malloc::StreamOrderedAllocator;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║   Stream-Ordered Allocation - Basic Example       ║");
    println!("╚════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize CUDA context
    println!("[1/5] Initializing CUDA context...");
    let context = Arc::new(CudaContext::new(0)?);
    println!("      ✓ Context created for device 0\n");

    // Step 2: Create stream-ordered allocator
    println!("[2/5] Creating stream-ordered allocator...");
    let allocator = StreamOrderedAllocator::new(0)?;
    println!("      ✓ Allocator created");
    println!("      CUDA Version: {}.{}",
             allocator.cuda_version() / 1000,
             (allocator.cuda_version() % 1000) / 10);
    println!("      Device ID: {}\n", allocator.device_id());

    // Step 3: Benchmark traditional allocation
    println!("[3/5] Benchmarking traditional cudaMalloc...");
    let traditional_time = benchmark_traditional_alloc(&context)?;
    println!("      Time: {:?}", traditional_time);
    println!("      Throughput: {:.0} allocs/sec\n",
             1000.0 / traditional_time.as_secs_f64());

    // Step 4: Benchmark stream-ordered allocation
    println!("[4/5] Benchmarking stream-ordered cudaMallocAsync...");
    let async_time = benchmark_async_alloc(&context, &allocator)?;
    println!("      Time: {:?}", async_time);
    println!("      Throughput: {:.0} allocs/sec\n",
             1000.0 / async_time.as_secs_f64());

    // Step 5: Compare results
    println!("[5/5] Performance Analysis:");
    let speedup = traditional_time.as_secs_f64() / async_time.as_secs_f64();
    println!("      Speedup: {:.2}x", speedup);

    if speedup >= 1.5 {
        println!("      Result: ✅ EXCELLENT - Exceeded expected performance!");
    } else if speedup >= 1.2 {
        println!("      Result: ✅ SUCCESS - Achieved expected 1.2-1.5x speedup!");
    } else {
        println!("      Result: ⚠️  BELOW TARGET - Got {:.2}x (expected 1.2x+)", speedup);
        println!("\n      Possible reasons:");
        println!("      • GPU memory caching (second run is cached)");
        println!("      • Small allocation size (overhead dominates)");
        println!("      • Try running again or use larger allocations");
    }

    println!("\n╔════════════════════════════════════════════════════╗");
    println!("║   Example completed successfully!                  ║");
    println!("╚════════════════════════════════════════════════════╝");

    Ok(())
}

/// Benchmark traditional cudaMalloc (via cudarc's alloc_zeros)
fn benchmark_traditional_alloc(context: &Arc<CudaContext>) -> Result<std::time::Duration, Box<dyn std::error::Error>> {
    const NUM_ITERATIONS: usize = 1000;
    const ALLOC_SIZE: usize = 1024; // 1KB per allocation

    let stream = context.default_stream();
    let start = Instant::now();

    // Allocate and immediately free (via RAII drop)
    for _ in 0..NUM_ITERATIONS {
        let _buffer = stream.alloc_zeros::<u8>(ALLOC_SIZE)?;
        // Buffer automatically freed when it goes out of scope
    }

    Ok(start.elapsed())
}

/// Benchmark stream-ordered cudaMallocAsync
fn benchmark_async_alloc(
    context: &Arc<CudaContext>,
    allocator: &StreamOrderedAllocator,
) -> Result<std::time::Duration, Box<dyn std::error::Error>> {
    const NUM_ITERATIONS: usize = 1000;
    const ALLOC_SIZE: usize = 1024; // 1KB per allocation

    let stream = context.default_stream();
    let start = Instant::now();

    // Allocate and immediately free
    for _ in 0..NUM_ITERATIONS {
        unsafe {
            let ptr = allocator.alloc_async(ALLOC_SIZE, stream.clone())?;
            allocator.free_async(ptr, stream.clone())?;
        }
    }

    // Synchronize to ensure all operations completed
    stream.synchronize()?;

    Ok(start.elapsed())
}
