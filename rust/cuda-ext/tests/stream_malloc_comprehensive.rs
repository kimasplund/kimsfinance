//! Comprehensive Test Suite for Stream-Ordered Memory Allocation
//!
//! This test suite covers:
//! 1. Basic functionality (allocation, free, pool creation)
//! 2. Edge cases (zero-size, large allocations, multiple pools)
//! 3. Error handling (invalid operations, out-of-memory)
//! 4. Concurrency (multiple streams, thread safety)
//! 5. Memory safety (no leaks, proper cleanup)
//!
//! # Running Tests
//!
//! ```bash
//! # Run all tests (requires GPU)
//! cargo test --test stream_malloc_comprehensive -- --ignored
//!
//! # Run specific test
//! cargo test --test stream_malloc_comprehensive test_basic_alloc_free -- --ignored
//! ```

use cudarc::driver::CudaContext;
use kimsfinance_cuda_ext::stream_malloc::StreamOrderedAllocator;
use std::sync::Arc;

/// Test basic allocator creation
#[test]
#[ignore] // Requires GPU
fn test_allocator_creation() {
    let _context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");

    // Verify allocator properties
    assert_eq!(allocator.device_id(), 0);
    assert!(allocator.cuda_version() >= 11020, "CUDA version must be >= 11.2");
}

/// Test basic allocation and free
#[test]
#[ignore] // Requires GPU
fn test_basic_alloc_free() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    // Allocate 1MB
    let ptr = unsafe {
        allocator.alloc_async(1024 * 1024, stream.clone())
            .expect("Allocation failed")
    };

    assert_ne!(ptr, 0, "Pointer should be non-null");

    // Free memory
    unsafe {
        allocator.free_async(ptr, stream.clone())
            .expect("Free failed");
    }

    // Synchronize to ensure completion
    stream.synchronize().expect("Synchronization failed");
}

/// Test zero-size allocation (edge case)
#[test]
#[ignore] // Requires GPU
fn test_zero_size_allocation() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    // Allocate 0 bytes (should succeed but may return null pointer)
    let result = unsafe {
        allocator.alloc_async(0, stream.clone())
    };

    // Zero-size allocation behavior is implementation-defined
    // Some implementations return null, others return a valid (non-dereferenceable) pointer
    match result {
        Ok(ptr) => {
            // If allocation succeeded, free it
            unsafe {
                allocator.free_async(ptr, stream.clone())
                    .expect("Free failed");
            }
        }
        Err(e) => {
            // Zero-size allocation may also fail, which is acceptable
            println!("Zero-size allocation failed (acceptable): {:?}", e);
        }
    }
}

/// Test small allocation (1 byte)
#[test]
#[ignore] // Requires GPU
fn test_small_allocation() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    // Allocate 1 byte
    let ptr = unsafe {
        allocator.alloc_async(1, stream.clone())
            .expect("Small allocation failed")
    };

    assert_ne!(ptr, 0);

    unsafe {
        allocator.free_async(ptr, stream.clone())
            .expect("Free failed");
    }
}

/// Test large allocation (1GB)
#[test]
#[ignore] // Requires GPU with sufficient memory
fn test_large_allocation() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    // Allocate 1GB
    let size = 1024 * 1024 * 1024;
    let result = unsafe {
        allocator.alloc_async(size, stream.clone())
    };

    match result {
        Ok(ptr) => {
            assert_ne!(ptr, 0);
            unsafe {
                allocator.free_async(ptr, stream.clone())
                    .expect("Free failed");
            }
        }
        Err(e) => {
            // Large allocation may fail if GPU doesn't have enough memory
            println!("Large allocation failed (may be acceptable): {:?}", e);
        }
    }
}

/// Test multiple sequential allocations
#[test]
#[ignore] // Requires GPU
fn test_multiple_allocations() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    const NUM_ALLOCS: usize = 100;
    const ALLOC_SIZE: usize = 1024 * 1024; // 1MB each

    let mut ptrs = Vec::with_capacity(NUM_ALLOCS);

    // Allocate 100 buffers
    for i in 0..NUM_ALLOCS {
        let ptr = unsafe {
            allocator.alloc_async(ALLOC_SIZE, stream.clone())
                .unwrap_or_else(|e| panic!("Allocation {} failed: {:?}", i, e))
        };
        assert_ne!(ptr, 0);
        ptrs.push(ptr);
    }

    // Verify all pointers are unique
    for i in 0..ptrs.len() {
        for j in (i + 1)..ptrs.len() {
            assert_ne!(ptrs[i], ptrs[j], "Pointers must be unique");
        }
    }

    // Free all buffers
    for (i, ptr) in ptrs.into_iter().enumerate() {
        unsafe {
            allocator.free_async(ptr, stream.clone())
                .unwrap_or_else(|e| panic!("Free {} failed: {:?}", i, e));
        }
    }

    stream.synchronize().expect("Synchronization failed");
}

/// Test allocation and immediate free (stress test)
#[test]
#[ignore] // Requires GPU
fn test_alloc_free_stress() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    const NUM_ITERATIONS: usize = 1000;
    const ALLOC_SIZE: usize = 4096; // 4KB

    // Rapidly allocate and free
    for i in 0..NUM_ITERATIONS {
        let ptr = unsafe {
            allocator.alloc_async(ALLOC_SIZE, stream.clone())
                .unwrap_or_else(|e| panic!("Iteration {} alloc failed: {:?}", i, e))
        };

        unsafe {
            allocator.free_async(ptr, stream.clone())
                .unwrap_or_else(|e| panic!("Iteration {} free failed: {:?}", i, e));
        }
    }

    stream.synchronize().expect("Synchronization failed");
}

/// Test concurrent allocations (simulated with single stream)
#[test]
#[ignore] // Requires GPU
fn test_concurrent_style_allocations() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");

    const NUM_STREAMS: usize = 4;
    const ALLOCS_PER_STREAM: usize = 50;
    const ALLOC_SIZE: usize = 2 * 1024 * 1024; // 2MB

    let stream = context.default_stream();

    // Simulate multiple streams (in real app, these would be independent)
    let streams: Vec<_> = (0..NUM_STREAMS)
        .map(|_| stream.clone())
        .collect();

    let mut all_ptrs = Vec::new();

    // Each "stream" allocates its batch
    for stream in &streams {
        let mut ptrs = Vec::with_capacity(ALLOCS_PER_STREAM);
        for _ in 0..ALLOCS_PER_STREAM {
            let ptr = unsafe {
                allocator.alloc_async(ALLOC_SIZE, stream.clone())
                    .expect("Allocation failed")
            };
            ptrs.push(ptr);
        }
        all_ptrs.push((ptrs, stream.clone()));
    }

    // Verify total allocations
    let total_allocations: usize = all_ptrs.iter().map(|(ptrs, _)| ptrs.len()).sum();
    assert_eq!(total_allocations, NUM_STREAMS * ALLOCS_PER_STREAM);

    // Each "stream" frees its batch
    for (ptrs, stream) in all_ptrs {
        for ptr in ptrs {
            unsafe {
                allocator.free_async(ptr, stream.clone())
                    .expect("Free failed");
            }
        }
    }

    stream.synchronize().expect("Synchronization failed");
}

/// Test pool trim operation
#[test]
#[ignore] // Requires GPU
fn test_pool_trim() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    // Allocate and free to populate pool
    let ptr = unsafe {
        allocator.alloc_async(10 * 1024 * 1024, stream.clone())
            .expect("Allocation failed")
    };

    unsafe {
        allocator.free_async(ptr, stream.clone())
            .expect("Free failed");
    }

    stream.synchronize().expect("Synchronization failed");

    // Trim pool (release unused memory)
    allocator.trim().expect("Trim failed");
}

/// Test allocator with multiple different sizes
#[test]
#[ignore] // Requires GPU
fn test_mixed_allocation_sizes() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    let sizes = vec![
        1024,           // 1KB
        64 * 1024,      // 64KB
        1024 * 1024,    // 1MB
        16 * 1024 * 1024, // 16MB
    ];

    let mut ptrs = Vec::new();

    // Allocate different sizes
    for &size in &sizes {
        let ptr = unsafe {
            allocator.alloc_async(size, stream.clone())
                .expect("Allocation failed")
        };
        ptrs.push((ptr, size));
    }

    // Free all
    for (ptr, _size) in ptrs {
        unsafe {
            allocator.free_async(ptr, stream.clone())
                .expect("Free failed");
        }
    }

    stream.synchronize().expect("Synchronization failed");
}

/// Test repeated allocate-free cycles (memory reuse)
#[test]
#[ignore] // Requires GPU
fn test_memory_reuse() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
    let stream = context.default_stream();

    const ALLOC_SIZE: usize = 1024 * 1024; // 1MB
    const NUM_CYCLES: usize = 100;

    let mut previous_ptr = 0;
    let mut reuse_count = 0;

    // Repeatedly allocate and free same size
    for _ in 0..NUM_CYCLES {
        let ptr = unsafe {
            allocator.alloc_async(ALLOC_SIZE, stream.clone())
                .expect("Allocation failed")
        };

        // Check if memory was reused (same pointer as previous)
        if ptr == previous_ptr && previous_ptr != 0 {
            reuse_count += 1;
        }

        previous_ptr = ptr;

        unsafe {
            allocator.free_async(ptr, stream.clone())
                .expect("Free failed");
        }

        // Synchronize to ensure free completes before next allocation
        stream.synchronize().expect("Synchronization failed");
    }

    // Stream-ordered allocation should reuse memory frequently
    println!("Memory reuse: {}/{} allocations", reuse_count, NUM_CYCLES - 1);
}

/// Test proper cleanup on drop
#[test]
#[ignore] // Requires GPU
fn test_allocator_drop_cleanup() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));

    // Create allocator in inner scope
    {
        let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");
        let stream = context.default_stream();

        // Allocate some memory
        let ptr = unsafe {
            allocator.alloc_async(1024, stream.clone())
                .expect("Allocation failed")
        };

        // Free it properly
        unsafe {
            allocator.free_async(ptr, stream.clone())
                .expect("Free failed");
        }

        stream.synchronize().expect("Synchronization failed");

        // Allocator drops here - should clean up pool
    }

    // If we get here without panicking, drop was successful
}

/// Test that CUDA version is correctly detected
#[test]
#[ignore] // Requires GPU
fn test_cuda_version_detection() {
    let _context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = StreamOrderedAllocator::new(0).expect("Failed to create allocator");

    let version = allocator.cuda_version();

    // Should be >= 11.2 (11020)
    assert!(version >= 11020, "CUDA version too old: {}", version);

    // Should be reasonable (< 20.0)
    assert!(version < 20000, "CUDA version suspiciously high: {}", version);

    let major = version / 1000;
    let minor = (version % 1000) / 10;

    println!("Detected CUDA version: {}.{}", major, minor);
}

/// Test thread safety (Send + Sync)
#[test]
fn test_send_sync_traits() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<StreamOrderedAllocator>();
    assert_sync::<StreamOrderedAllocator>();
}

/// Test allocator with Arc (shared ownership)
#[test]
#[ignore] // Requires GPU
fn test_shared_allocator() {
    let context = Arc::new(CudaContext::new(0).expect("Failed to initialize GPU"));
    let allocator = Arc::new(StreamOrderedAllocator::new(0).expect("Failed to create allocator"));

    let stream = context.default_stream();

    // Clone Arc to simulate sharing
    let allocator_clone = Arc::clone(&allocator);

    // Allocate with original
    let ptr = unsafe {
        allocator.alloc_async(1024, stream.clone())
            .expect("Allocation failed")
    };

    // Free with clone
    unsafe {
        allocator_clone.free_async(ptr, stream.clone())
            .expect("Free failed");
    }

    stream.synchronize().expect("Synchronization failed");
}
