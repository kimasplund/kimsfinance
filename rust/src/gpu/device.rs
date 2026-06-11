//! GPU Device Management
//!
//! Handles CUDA context and stream initialization, memory allocation, and error handling.

use super::async_alloc::AsyncAllocator;
use super::persistent::pinned_memory::PinnedBufferPool;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg,
    result::DriverError,
};
use cudarc::nvrtc::CompileError;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};

// Tunable constants for pinned memory pool
const PINNED_BUFFER_COUNT: usize = 16; // Number of reusable buffers
const PINNED_BUFFER_SIZE: usize = 1_000_000; // 1M f64 elements (~8MB per buffer)

/// GPU device handle with memory management
///
/// Provides safe CUDA stream access for memory operations and kernel execution.
pub struct GpuDevice {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub device_id: usize,
    /// Pool of reusable pinned memory buffers for 20-30% faster async transfers
    pub(crate) pinned_pool: Mutex<PinnedBufferPool<f64>>,
    /// Async memory allocator for 1.2-1.5x faster allocation (CUDA 11.2+)
    pub(crate) async_allocator: Option<Arc<AsyncAllocator>>,
    /// Per-device cache of loaded CUDA modules, keyed by
    /// `compile::kernel_source_hash_u64(kernel_src)` (first 8 bytes of the same
    /// SHA-256 digest that keys the process-wide PTX cache).
    ///
    /// `context.load_module()` (cuModuleLoadData + driver JIT) costs ~0.1-1ms per
    /// call; caching the `Arc<CudaModule>` makes repeat indicator invocations a
    /// lock-free map lookup + cuModuleGetFunction (sub-microsecond). See
    /// `get_or_load_function`.
    pub(crate) module_cache: DashMap<u64, Arc<CudaModule>>,
}

impl GpuDevice {
    /// Initialize GPU device (device 0 by default)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No CUDA-capable GPU found
    /// - CUDA driver not installed
    /// - Context initialization fails
    pub fn new() -> Result<Self, GpuError> {
        Self::with_device_id(0)
    }

    /// Initialize specific GPU device by ID
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device ordinal (0 for first GPU, 1 for second, etc.)
    pub fn with_device_id(device_id: usize) -> Result<Self, GpuError> {
        let context = CudaContext::new(device_id).map_err(|e| {
            GpuError::InitializationError(format!(
                "Failed to initialize CUDA context {}: {:?}",
                device_id, e
            ))
        })?;

        // Get the default stream
        let stream = context.default_stream();

        // Initialize pinned memory pool for faster async transfers
        // Fallback: If pinned allocation fails (e.g., WSL, limited system resources),
        // create an empty pool. Operations will then fall back to pageable memory.
        let pinned_pool = match PinnedBufferPool::new(PINNED_BUFFER_COUNT, PINNED_BUFFER_SIZE) {
            Ok(pool) => pool,
            Err(e) => {
                eprintln!(
                    "Warning: Failed to allocate pinned memory pool: {:?}. \
                     GPU transfers will use slower pageable memory. \
                     This may happen on systems with limited pinned memory support (e.g., WSL1).",
                    e
                );
                // Create an empty pool as a fallback
                PinnedBufferPool::new(0, 0)?
            }
        };

        // Initialize async memory allocator (CUDA 11.2+)
        // Fallback: If CUDA < 11.2 or pool creation fails, async_allocator will be None
        // Note: device_id is inferred from context (always device 0 for now, TODO: multi-GPU)
        let async_allocator = AsyncAllocator::new(stream.clone(), device_id as i32)
            .ok()
            .map(Arc::new);

        Ok(Self {
            context,
            stream,
            device_id,
            pinned_pool: Mutex::new(pinned_pool),
            async_allocator,
            module_cache: DashMap::new(),
        })
    }

    /// Get the process-wide shared GPU device (device 0), initializing it on first call
    ///
    /// Constructing a `GpuDevice` is expensive: CUDA context creation plus eager
    /// allocation of a ~128MB pinned-memory pool. Several call sites currently
    /// construct a fresh device per operation (`batch.rs`, `triple_buffer.rs`,
    /// `persistent/mod.rs`), repeating that cost on every call. This singleton
    /// amortizes it to once per process.
    ///
    /// The first call's outcome (success or failure) is cached: on a host without
    /// a usable GPU, subsequent calls return the same `InitializationError`
    /// without retrying driver initialization.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::InitializationError` if device 0 could not be
    /// initialized (no CUDA GPU, driver missing, context creation failed).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = GpuDevice::global()?;
    /// let rsi = device.get_or_load_function(RSI_KERNEL, "calculate_rsi_kernel")?;
    /// ```
    pub fn global() -> Result<Arc<GpuDevice>, GpuError> {
        static GLOBAL_DEVICE: OnceLock<Result<Arc<GpuDevice>, String>> = OnceLock::new();

        GLOBAL_DEVICE
            .get_or_init(|| GpuDevice::new().map(Arc::new).map_err(|e| e.to_string()))
            .clone()
            .map_err(GpuError::InitializationError)
    }

    /// Get a compiled kernel function, loading and caching its module on first use
    ///
    /// Replaces the per-call pattern
    /// `Arc::unwrap_or_clone(compile_ptx_optimized_cached(src)?)` +
    /// `context.load_module(ptx)` (deep-clones the multi-KB PTX string and pays
    /// cuModuleLoadData + driver JIT, ~0.1-1ms, on **every** invocation) with a
    /// per-device module cache:
    ///
    /// - **First call** per (device, kernel source): NVRTC compile (process-wide
    ///   PTX cache), one PTX clone, one module load. Module cached.
    /// - **Subsequent calls**: SHA-256 of source + lock-free map lookup +
    ///   cuModuleGetFunction (sub-microsecond).
    ///
    /// # Arguments
    ///
    /// * `kernel_src` - CUDA C source string (NVRTC-compatible: no `#include`,
    ///   `extern "C" __global__` entry points)
    /// * `fn_name` - Name of the `extern "C" __global__` function to load
    ///
    /// # Thread Safety
    ///
    /// Safe to call concurrently. If two threads race on the same uncached
    /// kernel, both load a module but only the first insert wins; the loser's
    /// module is dropped (unloaded) and its function is taken from the winner.
    ///
    /// # Errors
    ///
    /// - `GpuError::CompilationError` if NVRTC compilation or module load fails
    /// - `GpuError::ExecutionError` if `fn_name` is not found in the module
    pub fn get_or_load_function(
        &self,
        kernel_src: &str,
        fn_name: &str,
    ) -> Result<CudaFunction, GpuError> {
        let key = super::compile::kernel_source_hash_u64(kernel_src);

        // Fast path: module already loaded on this device
        if let Some(entry) = self.module_cache.get(&key) {
            let module = Arc::clone(entry.value());
            drop(entry); // Release shard lock before driver call
            return module.load_function(fn_name).map_err(|e| {
                GpuError::ExecutionError(format!(
                    "Failed to load kernel function '{}': {:?}",
                    fn_name, e
                ))
            });
        }

        // Slow path: compile (PTX itself is cached process-wide) and load once.
        let ptx_arc = super::compile::compile_ptx_cached_ref(kernel_src).map_err(|e| {
            GpuError::CompilationError(format!("Failed to compile kernel: {:?}", e))
        })?;

        // load_module takes Ptx by value; this clone happens at most once per
        // (device, kernel) thanks to the module cache — unlike the per-call
        // Arc::unwrap_or_clone pattern this API replaces.
        let loaded = self.context.load_module((*ptx_arc).clone()).map_err(|e| {
            GpuError::CompilationError(format!("Failed to load PTX module: {:?}", e))
        })?;

        // entry().or_insert keeps the first-inserted module if another thread won
        // the race; our `loaded` is then dropped (module unloaded) safely because
        // no CudaFunction was created from it.
        let module = Arc::clone(self.module_cache.entry(key).or_insert(loaded).value());

        module.load_function(fn_name).map_err(|e| {
            GpuError::ExecutionError(format!(
                "Failed to load kernel function '{}': {:?}",
                fn_name, e
            ))
        })
    }

    /// Allocate GPU memory buffer (traditional approach)
    ///
    /// # Arguments
    ///
    /// * `len` - Number of f64 elements to allocate
    ///
    /// # Performance
    ///
    /// Uses traditional memory allocation. For faster allocation, consider
    /// `alloc_async()` for 1.2-1.5x improvement (CUDA 11.2+).
    pub fn alloc_buffer(&self, len: usize) -> Result<CudaSlice<f64>, GpuError> {
        self.stream.alloc_zeros::<f64>(len).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate {} elements: {:?}", len, e))
        })
    }

    /// Allocate GPU memory **without** zero-initialization
    ///
    /// `alloc_buffer`/`allocate_device_buffer` use `alloc_zeros`, which issues a
    /// `cudaMemsetAsync` over the entire buffer — a full extra VRAM write pass.
    /// For kernel *output* buffers whose every element is overwritten before any
    /// read, that memset is pure waste; this method skips it.
    ///
    /// # Contract (caller responsibility)
    ///
    /// The buffer contents are **garbage** until written. The caller MUST ensure
    /// every element is overwritten by a kernel (or explicit NaN-fill for warmup
    /// regions) before it is read or copied back to host.
    ///
    /// **Counter-examples — keep `alloc_zeros` for these:**
    /// - `rsi.rs`: `calculate_gains_losses_kernel` writes `gains[idx + 1]` /
    ///   `losses[idx + 1]` only, leaving element 0 untouched — it relies on the
    ///   implicit zero from `alloc_buffer`.
    /// - `ichimoku.rs`: `shift_forward_kernel` writes `output[idx + displacement]`
    ///   only, leaving the first `displacement` elements unwritten (zeroed
    ///   explicitly via `memset_zeros` before launch).
    ///
    /// # Arguments
    ///
    /// * `len` - Number of `T` elements to allocate
    ///
    /// # Errors
    ///
    /// Returns `GpuError::AllocationError` if device allocation fails.
    pub fn alloc_uninit<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, GpuError> {
        // SAFETY: cudarc marks `alloc` unsafe because the memory is unset. The
        // contract above shifts initialization responsibility to the caller; for
        // the POD numeric types used in this crate (f64/f32/i64/i32/i8/u8) any
        // bit pattern is a valid value, so reading garbage yields wrong numbers,
        // not undefined behavior.
        unsafe { self.stream.alloc::<T>(len) }.map_err(|e| {
            GpuError::AllocationError(format!(
                "Failed to allocate {} uninitialized elements: {:?}",
                len, e
            ))
        })
    }

    /// Allocate GPU memory using async allocator (1.2-1.5x faster, CUDA 11.2+)
    ///
    /// # Arguments
    ///
    /// * `len` - Number of f64 elements to allocate
    ///
    /// # Performance
    ///
    /// - **CUDA >= 11.2**: Uses cudaMallocAsync for 1.2-1.5x faster allocation
    /// - **CUDA < 11.2**: Automatically falls back to standard allocation
    ///
    /// # When to Use
    ///
    /// - Allocation-heavy code (frequent alloc/free cycles)
    /// - Batch processing with many temporary buffers
    /// - Multi-stream workloads
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = GpuDevice::new()?;
    ///
    /// // Use async allocation for better performance
    /// let buffer = device.alloc_async(1_000_000)?;
    /// ```
    pub fn alloc_async(&self, len: usize) -> Result<CudaSlice<f64>, GpuError> {
        if let Some(allocator) = &self.async_allocator {
            allocator.alloc(len)
        } else {
            // Fallback to standard allocation
            self.alloc_buffer(len)
        }
    }

    /// Allocate GPU memory for i32 using async allocator (1.2-1.5x faster, CUDA 11.2+)
    ///
    /// # Arguments
    ///
    /// * `len` - Number of i32 elements to allocate
    ///
    /// # Performance
    ///
    /// - **CUDA >= 11.2**: Uses cudaMallocAsync for 1.2-1.5x faster allocation
    /// - **CUDA < 11.2**: Automatically falls back to standard allocation
    ///
    /// # When to Use
    ///
    /// - Allocation-heavy code (frequent alloc/free cycles)
    /// - Batch processing with many temporary buffers
    /// - Multi-stream workloads
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = GpuDevice::new()?;
    ///
    /// // Use async allocation for better performance
    /// let buffer = device.alloc_async_i32(1_000)?;
    /// ```
    pub fn alloc_async_i32(&self, len: usize) -> Result<CudaSlice<i32>, GpuError> {
        if let Some(allocator) = &self.async_allocator {
            allocator.alloc(len)
        } else {
            // Fallback to standard allocation
            self.allocate_device_buffer(len)
        }
    }

    /// Allocate GPU memory for u8 using async allocator (1.2-1.5x faster, CUDA 11.2+)
    ///
    /// # Arguments
    ///
    /// * `len` - Number of u8 elements to allocate
    ///
    /// # Performance
    ///
    /// - **CUDA >= 11.2**: Uses cudaMallocAsync for 1.2-1.5x faster allocation
    /// - **CUDA < 11.2**: Automatically falls back to standard allocation
    ///
    /// # When to Use
    ///
    /// - Allocation-heavy code (frequent alloc/free cycles)
    /// - Batch processing with many temporary buffers
    /// - Multi-stream workloads
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = GpuDevice::new()?;
    ///
    /// // Use async allocation for better performance
    /// let buffer = device.alloc_async_u8(1024)?;
    /// ```
    pub fn alloc_async_u8(&self, len: usize) -> Result<CudaSlice<u8>, GpuError> {
        if let Some(allocator) = &self.async_allocator {
            allocator.alloc(len)
        } else {
            // Fallback to standard allocation
            self.allocate_device_buffer(len)
        }
    }

    /// Check if async allocation is supported
    ///
    /// # Returns
    ///
    /// - `true`: CUDA >= 11.2, async allocation available (1.2-1.5x faster)
    /// - `false`: CUDA < 11.2, standard allocation only
    pub fn supports_async_alloc(&self) -> bool {
        self.async_allocator
            .as_ref()
            .map_or(false, |a| a.supports_async())
    }

    /// Get async allocator statistics
    ///
    /// # Returns
    ///
    /// Statistics if async allocator is available, None otherwise
    pub fn async_alloc_stats(&self) -> Option<super::async_alloc::PoolStats> {
        self.async_allocator.as_ref().map(|a| a.stats())
    }

    /// Trim excess memory from async allocator
    ///
    /// Releases unused memory back to OS. Only effective with CUDA >= 11.2.
    pub fn trim_async_pool(&self) {
        if let Some(allocator) = &self.async_allocator {
            allocator.trim();
        }
    }

    /// Allocate memory from stream-ordered pool (CUDA 13.0 optimization)
    ///
    /// # Arguments
    ///
    /// * `len` - Number of f64 elements to allocate
    ///
    /// # Performance
    ///
    /// **CUDA 13.0 Feature**: Stream-ordered memory allocator provides:
    /// - **10-20% faster** allocation for memory-bound kernels
    /// - **Reduced fragmentation** through stream-specific pools
    /// - **Better concurrency** - allocations don't block other streams
    ///
    /// # When to Use
    ///
    /// ✅ **Use stream-ordered allocation when:**
    /// - Kernel is memory-bound (bandwidth-limited)
    /// - Allocating/freeing frequently (batch processing)
    /// - Using multiple streams (concurrent execution)
    ///
    /// ❌ **Use traditional allocation when:**
    /// - Kernel is compute-bound (allocation overhead negligible)
    /// - Memory lives for long duration (single allocation at startup)
    /// - Single stream workflow (no concurrency benefits)
    ///
    /// # CUDA Version
    ///
    /// - **Required**: CUDA 11.2+ for basic stream-ordered malloc
    /// - **Recommended**: CUDA 13.0+ for improved pool management (10-20% faster)
    /// - **Current Driver**: 13.0 ✅ (fully optimized)
    ///
    /// # Implementation Status
    ///
    /// **PLACEHOLDER**: This is a design document for stream-ordered allocation.
    /// Full implementation requires:
    /// - cudarc stream-ordered malloc API (tracking issue pending)
    /// - OR direct CUDA driver API via unsafe FFI (`cudaMallocAsync`, `cudaFreeAsync`)
    ///
    /// Currently falls back to traditional allocation (no performance change).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = GpuDevice::new()?;
    ///
    /// // Traditional allocation (current behavior)
    /// let buffer1 = device.alloc_buffer(10_000)?;
    ///
    /// // Stream-ordered allocation (future CUDA 13.0 optimization)
    /// let buffer2 = device.alloc_stream_ordered(10_000)?;
    /// // ↑ 10-20% faster for memory-bound kernels
    /// ```
    ///
    /// # References
    ///
    /// - CUDA Stream-Ordered Memory Guide: https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
    /// - CUDA 13.0 improvements: Enhanced pool management, reduced overhead
    /// - cudarc tracking: Stream-ordered malloc API pending
    #[allow(dead_code)]
    pub fn alloc_stream_ordered(&self, len: usize) -> Result<CudaSlice<f64>, GpuError> {
        // TODO: When cudarc adds stream-ordered malloc, use:
        // self.stream.alloc_zeros_async::<f64>(len).map_err(|e| {
        //     GpuError::AllocationError(format!("Failed to allocate {} elements: {:?}", len, e))
        // })

        // PLACEHOLDER: Fall back to traditional allocation
        // This maintains API compatibility for future optimization
        self.alloc_buffer(len)
    }

    /// Free stream-ordered memory (CUDA 13.0 optimization)
    ///
    /// # Arguments
    ///
    /// * `buffer` - GPU buffer allocated with `alloc_stream_ordered()`
    ///
    /// # Performance
    ///
    /// Stream-ordered free is asynchronous and doesn't block the host thread.
    /// This enables better overlap between CPU and GPU work.
    ///
    /// # Safety
    ///
    /// Memory is not actually freed until:
    /// 1. All kernels using this memory on this stream complete
    /// 2. Stream synchronization occurs
    ///
    /// This is automatic and safe - the CUDA driver manages lifetime.
    ///
    /// # Implementation Status
    ///
    /// **PLACEHOLDER**: Currently no-op (cudarc handles deallocation automatically).
    /// Future implementation will use `cudaFreeAsync()` for true asynchronous free.
    #[allow(dead_code)]
    pub fn free_stream_ordered(&self, _buffer: CudaSlice<f64>) -> Result<(), GpuError> {
        // TODO: When cudarc adds stream-ordered malloc, use:
        // self.stream.free_async(buffer)?;

        // PLACEHOLDER: cudarc handles deallocation via RAII (Drop trait)
        // No explicit free needed - buffer is freed when dropped
        Ok(())
    }

    /// Copy data from host to device
    ///
    /// # Arguments
    ///
    /// * `data` - Host data slice
    pub fn copy_to_device(&self, data: &[f64]) -> Result<CudaSlice<f64>, GpuError> {
        // Allocate device buffer
        let mut buffer = self.stream.alloc_zeros::<f64>(data.len()).map_err(|e| {
            GpuError::AllocationError(format!(
                "Failed to allocate {} elements: {:?}",
                data.len(),
                e
            ))
        })?;

        // Copy data into buffer
        self.stream.memcpy_htod(data, &mut buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!(
                "Failed to copy {} elements to device: {:?}",
                data.len(),
                e
            ))
        })?;

        Ok(buffer)
    }

    /// Copy i32 data from host to device
    ///
    /// # Arguments
    ///
    /// * `data` - Host data slice of i32 values
    pub fn copy_to_device_i32(&self, data: &[i32]) -> Result<CudaSlice<i32>, GpuError> {
        // Allocate device buffer
        let mut buffer = self.stream.alloc_zeros::<i32>(data.len()).map_err(|e| {
            GpuError::AllocationError(format!(
                "Failed to allocate {} i32 elements: {:?}",
                data.len(),
                e
            ))
        })?;

        // Copy data into buffer
        self.stream.memcpy_htod(data, &mut buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!(
                "Failed to copy {} i32 elements to device: {:?}",
                data.len(),
                e
            ))
        })?;

        Ok(buffer)
    }

    /// Copy i64 data from host to device
    ///
    /// # Arguments
    ///
    /// * `data` - Host data slice of i64 values
    pub fn copy_to_device_i64(&self, data: &[i64]) -> Result<CudaSlice<i64>, GpuError> {
        // Allocate device buffer
        let mut buffer = self.stream.alloc_zeros::<i64>(data.len()).map_err(|e| {
            GpuError::AllocationError(format!(
                "Failed to allocate {} i64 elements: {:?}",
                data.len(),
                e
            ))
        })?;

        // Copy data into buffer
        self.stream.memcpy_htod(data, &mut buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!(
                "Failed to copy {} i64 elements to device: {:?}",
                data.len(),
                e
            ))
        })?;

        Ok(buffer)
    }

    /// Copy f32 data from host to device
    ///
    /// # Arguments
    ///
    /// * `data` - Host data slice of f32 values
    pub fn copy_to_device_f32(&self, data: &[f32]) -> Result<CudaSlice<f32>, GpuError> {
        // Allocate device buffer
        let mut buffer = self.stream.alloc_zeros::<f32>(data.len()).map_err(|e| {
            GpuError::AllocationError(format!(
                "Failed to allocate {} f32 elements: {:?}",
                data.len(),
                e
            ))
        })?;

        // Copy data into buffer
        self.stream.memcpy_htod(data, &mut buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!(
                "Failed to copy {} f32 elements to device: {:?}",
                data.len(),
                e
            ))
        })?;

        Ok(buffer)
    }

    /// Copy f32 data from device to host
    ///
    /// # Arguments
    ///
    /// * `buffer` - GPU buffer to copy from
    pub fn copy_to_host_f32(&self, buffer: &CudaSlice<f32>) -> Result<Vec<f32>, GpuError> {
        self.stream.memcpy_dtov(buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy f32 from device: {:?}", e))
        })
    }

    /// Copy i8 data from host to device
    ///
    /// # Arguments
    ///
    /// * `data` - Host data slice of i8 values
    pub fn copy_to_device_i8(&self, data: &[i8]) -> Result<CudaSlice<i8>, GpuError> {
        // Allocate device buffer
        let mut buffer = self.stream.alloc_zeros::<i8>(data.len()).map_err(|e| {
            GpuError::AllocationError(format!(
                "Failed to allocate {} i8 elements: {:?}",
                data.len(),
                e
            ))
        })?;

        // Copy data into buffer
        self.stream.memcpy_htod(data, &mut buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!(
                "Failed to copy {} i8 elements to device: {:?}",
                data.len(),
                e
            ))
        })?;

        Ok(buffer)
    }

    /// Copy i8 data from device to host
    ///
    /// # Arguments
    ///
    /// * `buffer` - GPU buffer to copy from
    pub fn copy_to_host_i8(&self, buffer: &CudaSlice<i8>) -> Result<Vec<i8>, GpuError> {
        self.stream.memcpy_dtov(buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy i8 from device: {:?}", e))
        })
    }

    /// Copy u8 data from device to host
    ///
    /// # Arguments
    ///
    /// * `buffer` - GPU buffer to copy from
    pub fn copy_to_host_u8(&self, buffer: &CudaSlice<u8>) -> Result<Vec<u8>, GpuError> {
        self.stream.memcpy_dtov(buffer).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy u8 from device: {:?}", e))
        })
    }

    /// Copy data from device to host
    ///
    /// # Arguments
    ///
    /// * `buffer` - GPU buffer to copy from
    pub fn copy_to_host(&self, buffer: &CudaSlice<f64>) -> Result<Vec<f64>, GpuError> {
        self.stream
            .memcpy_dtov(buffer)
            .map_err(|e| GpuError::MemoryCopyError(format!("Failed to copy from device: {:?}", e)))
    }

    /// Allocate device buffer (traditional approach)
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements to allocate
    ///
    /// # Performance
    ///
    /// This is an alias for `alloc_buffer()` to maintain API consistency
    /// with pinned memory operations.
    pub fn allocate_device_buffer<
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    >(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, GpuError> {
        self.stream.alloc_zeros::<T>(len).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate {} elements: {:?}", len, e))
        })
    }

    /// Copy data from pinned memory to device (20-30% faster than pageable)
    ///
    /// # Arguments
    ///
    /// * `pinned` - Pinned host buffer
    /// * `dst` - Device buffer to copy into
    ///
    /// # Performance
    ///
    /// Pinned memory enables direct DMA transfers without intermediate page-locking,
    /// resulting in 20-30% faster H2D transfers compared to pageable memory.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kimsfinance_core::gpu::persistent::PinnedBuffer;
    ///
    /// let mut pinned = PinnedBuffer::new(1000)?;
    /// pinned.copy_from_slice(&data);
    ///
    /// let mut d_buffer = device.allocate_device_buffer(1000)?;
    /// device.htod_pinned(&pinned, &mut d_buffer)?;
    /// ```
    pub fn htod_pinned<T: cudarc::driver::DeviceRepr>(
        &self,
        pinned: &crate::gpu::persistent::PinnedBuffer<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), GpuError> {
        self.stream
            .memcpy_htod(pinned.as_slice(), dst)
            .map_err(Into::into)
    }

    /// Copy data from device to pinned memory (20-30% faster than pageable)
    ///
    /// # Arguments
    ///
    /// * `src` - Device buffer to copy from
    /// * `pinned` - Pinned host buffer to copy into
    ///
    /// # Performance
    ///
    /// Pinned memory enables direct DMA transfers without intermediate page-locking,
    /// resulting in 20-30% faster D2H transfers compared to pageable memory.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kimsfinance_core::gpu::persistent::PinnedBuffer;
    ///
    /// let d_buffer = device.copy_to_device(&data)?;
    /// let mut pinned = PinnedBuffer::new(1000)?;
    /// device.dtoh_pinned(&d_buffer, &mut pinned)?;
    /// ```
    pub fn dtoh_pinned<T: cudarc::driver::DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        pinned: &mut crate::gpu::persistent::PinnedBuffer<T>,
    ) -> Result<(), GpuError> {
        self.stream
            .memcpy_dtoh(src, pinned.as_mut_slice())
            .map_err(Into::into)
    }

    /// Synchronize device (wait for all kernels to complete)
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.stream.synchronize().map_err(|e| {
            GpuError::SynchronizationError(format!("Device synchronization failed: {:?}", e))
        })
    }

    /// Get reference to CUDA context
    ///
    /// Required for loading PTX modules via `context.load_module()`
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get reference to CUDA stream
    ///
    /// Required for launching custom kernels or custom stream operations
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Query kernel occupancy to determine optimal grid size
    ///
    /// Uses CUDA occupancy API to calculate maximum active blocks per multiprocessor
    /// for the given kernel configuration.
    ///
    /// # Arguments
    ///
    /// * `func` - Compiled CUDA kernel function
    /// * `block_size` - Thread block size (e.g., 256)
    /// * `shared_mem` - Dynamic shared memory per block in bytes
    ///
    /// # Returns
    ///
    /// Number of blocks that can be active per SM. Multiply by SM count to get total grid size.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = GpuDevice::new()?;
    /// let func = compile_kernel(&device)?;
    ///
    /// let blocks_per_sm = device.query_occupancy(&func, 256, 0)?;
    /// let sm_count = 80; // RTX 3500 Ada
    /// let optimal_grid = blocks_per_sm * sm_count;
    /// ```
    pub fn query_occupancy(
        &self,
        func: &cudarc::driver::CudaFunction,
        block_size: u32,
        shared_mem: usize,
    ) -> Result<u32, GpuError> {
        func.occupancy_max_active_blocks_per_multiprocessor(block_size, shared_mem, None)
            .map_err(|e| {
                GpuError::ExecutionError(format!("Failed to query kernel occupancy: {:?}", e))
            })
    }

    /// Get GPU compute capability
    ///
    /// Returns the compute capability of the GPU (e.g., (8, 9) for RTX 3500 Ada).
    ///
    /// # Returns
    ///
    /// Tuple of (major, minor) compute capability version
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = GpuDevice::new()?;
    /// let (major, minor) = device.compute_capability();
    /// println!("GPU Compute Capability: {}.{}", major, minor);
    ///
    /// // Check for FP8 support (requires 8.9+)
    /// if major >= 8 && minor >= 9 {
    ///     println!("FP8 tensor cores supported!");
    /// }
    /// ```
    pub fn compute_capability(&self) -> (u32, u32) {
        use cudarc::driver::sys;

        unsafe {
            let mut major = 0i32;
            let mut minor = 0i32;

            // Query compute capability from device
            let result_major = sys::cuDeviceGetAttribute(
                &mut major,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.context.cu_device(),
            );

            let result_minor = sys::cuDeviceGetAttribute(
                &mut minor,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.context.cu_device(),
            );

            // Check if both results succeeded
            if result_major == sys::cudaError_enum::CUDA_SUCCESS
                && result_minor == sys::cudaError_enum::CUDA_SUCCESS
            {
                (major as u32, minor as u32)
            } else {
                // Fallback: assume compute capability 7.0 (Volta) if query fails
                eprintln!("Warning: Failed to query compute capability, assuming 7.0");
                (7, 0)
            }
        }
    }
}

/// Query GPU compute capability via the CUDA driver, without creating a context
///
/// Standalone variant of `GpuDevice::compute_capability` for use before any
/// `GpuDevice` exists — notably by `compile::detect_gpu_arch()` to pick the
/// NVRTC target architecture. Performs `cuInit` (idempotent, cheap after the
/// first call) + `cuDeviceGet` + `cuDeviceGetAttribute`; this replaces the
/// previous nvidia-smi subprocess query (~10-50ms process spawn + CSV parsing).
///
/// # Arguments
///
/// * `device_ordinal` - CUDA device ordinal (0 for first GPU)
///
/// # Returns
///
/// `Some((major, minor))` on success (e.g. `(8, 9)` for RTX 3500 Ada), or
/// `None` if the driver is unavailable, the ordinal is invalid, or any query
/// fails. Callers choose their own fallback (e.g. compute_89).
pub fn query_compute_capability(device_ordinal: i32) -> Option<(u32, u32)> {
    use cudarc::driver::{result, sys};

    result::init().ok()?;
    let dev = result::device::get(device_ordinal).ok()?;

    // SAFETY: `dev` was just returned by cuDeviceGet (result::device::get).
    let major = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .ok()?;

    // SAFETY: same `dev` as above.
    let minor = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .ok()?;

    Some((major as u32, minor as u32))
}

/// GPU operation errors
#[derive(Debug)]
pub enum GpuError {
    /// GPU context initialization failed
    InitializationError(String),

    /// Memory allocation failed
    AllocationError(String),

    /// Memory copy failed
    MemoryCopyError(String),

    /// Kernel compilation failed
    CompilationError(String),

    /// Kernel execution failed
    ExecutionError(String),

    /// Device synchronization failed
    SynchronizationError(String),

    /// Invalid parameters (dynamic message)
    InvalidParameter(String),

    /// cudarc driver error
    DriverError(DriverError),

    /// Computation error (dynamic message)
    ComputationError(String),

    // ===== Static Error Variants (Zero-Allocation) =====
    /// Empty OHLCV data provided
    EmptyOhlcvData,

    /// OHLCV arrays have mismatched lengths
    OhlcvLengthMismatch,

    /// Parameter grid is empty
    EmptyParameterGrid,

    /// Invalid input data provided
    InvalidInput(String),

    /// Backtesting error
    BacktestError(String),

    /// Invalid parameter with static message
    InvalidParameterStatic(&'static str),

    /// Computation error with static message
    ComputationErrorStatic(&'static str),

    /// Insufficient compute capability for operation
    InsufficientComputeCapability { required: String, found: String },

    /// Invalid matrix dimensions
    InvalidDimensions { expected: usize, found: usize },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::InitializationError(msg) => write!(f, "GPU initialization error: {}", msg),
            GpuError::AllocationError(msg) => write!(f, "GPU allocation error: {}", msg),
            GpuError::MemoryCopyError(msg) => write!(f, "GPU memory copy error: {}", msg),
            GpuError::CompilationError(msg) => write!(f, "CUDA kernel compilation error: {}", msg),
            GpuError::ExecutionError(msg) => write!(f, "CUDA kernel execution error: {}", msg),
            GpuError::SynchronizationError(msg) => write!(f, "GPU synchronization error: {}", msg),
            GpuError::InvalidParameter(msg) => write!(f, "Invalid GPU parameter: {}", msg),
            GpuError::DriverError(e) => write!(f, "CUDA driver error: {:?}", e),
            GpuError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            // Static error variants (zero-allocation)
            GpuError::EmptyOhlcvData => write!(f, "Empty OHLCV data"),
            GpuError::OhlcvLengthMismatch => write!(f, "OHLCV arrays must have same length"),
            GpuError::EmptyParameterGrid => write!(f, "Parameter grid is empty"),
            GpuError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            GpuError::BacktestError(msg) => write!(f, "Backtest error: {}", msg),
            GpuError::InvalidParameterStatic(msg) => write!(f, "Invalid GPU parameter: {}", msg),
            GpuError::ComputationErrorStatic(msg) => write!(f, "Computation error: {}", msg),
            GpuError::InsufficientComputeCapability { required, found } => {
                write!(
                    f,
                    "Insufficient compute capability: required {}, found {}",
                    required, found
                )
            }
            GpuError::InvalidDimensions { expected, found } => {
                write!(
                    f,
                    "Invalid dimensions: expected {}, found {}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError::DriverError(e)
    }
}

impl From<CompileError> for GpuError {
    fn from(e: CompileError) -> Self {
        GpuError::CompilationError(format!("{:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_device_initialization() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        println!("GPU context initialized");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_operations() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        // Copy to device
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gpu_buffer = device
            .copy_to_device(&data)
            .expect("Failed to copy to device");

        // Copy back to host
        let result = device
            .copy_to_host(&gpu_buffer)
            .expect("Failed to copy to host");

        assert_eq!(data, result);
    }

    /// NVRTC-compatible test kernel: no #include, extern "C" __global__ entry point
    const MODULE_CACHE_TEST_KERNEL: &str = r#"
    extern "C" __global__ void module_cache_test_kernel(double* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (double)idx;
        }
    }
    "#;

    #[test]
    fn test_module_cache_test_kernel_is_nvrtc_compatible() {
        // Host-side guard (no GPU needed): the test kernel source must stay
        // NVRTC-JIT-compatible.
        assert!(
            !MODULE_CACHE_TEST_KERNEL.contains("#include"),
            "NVRTC kernels must not use #include directives"
        );
        assert!(
            MODULE_CACHE_TEST_KERNEL.contains("extern \"C\" __global__"),
            "NVRTC kernels must use extern \"C\" __global__ entry points"
        );
    }

    #[test]
    fn test_global_device_consistent_across_calls() {
        // Runs with or without a GPU: the OnceLock must cache the first outcome,
        // so two calls always agree (same Arc on success, error again on failure).
        let first = GpuDevice::global();
        let second = GpuDevice::global();

        match (first, second) {
            (Ok(a), Ok(b)) => {
                assert!(
                    Arc::ptr_eq(&a, &b),
                    "global() must return the same Arc<GpuDevice> on every call"
                );
            }
            (Err(_), Err(_)) => {
                // Cached initialization failure (no GPU on this host) — consistent.
            }
            _ => panic!("global() must be consistent across calls (Ok/Ok or Err/Err)"),
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_global_device_singleton() {
        let a = GpuDevice::global().expect("Failed to initialize global GPU device");
        let b = GpuDevice::global().expect("Failed to get global GPU device");
        assert!(Arc::ptr_eq(&a, &b), "global() must return the same Arc");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_get_or_load_function_caches_module() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        assert_eq!(device.module_cache.len(), 0, "Module cache starts empty");

        let _f1 = device
            .get_or_load_function(MODULE_CACHE_TEST_KERNEL, "module_cache_test_kernel")
            .expect("First load should compile, load, and cache the module");
        assert_eq!(
            device.module_cache.len(),
            1,
            "Module should be cached after first load"
        );

        let _f2 = device
            .get_or_load_function(MODULE_CACHE_TEST_KERNEL, "module_cache_test_kernel")
            .expect("Second load should hit the module cache");
        assert_eq!(
            device.module_cache.len(),
            1,
            "Repeat load of the same source must reuse the cached module"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_get_or_load_function_unknown_name_fails() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");
        let result = device.get_or_load_function(MODULE_CACHE_TEST_KERNEL, "no_such_kernel");
        assert!(
            result.is_err(),
            "Loading a non-existent function name must fail"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_alloc_uninit_roundtrip() {
        let device = GpuDevice::new().expect("Failed to initialize GPU");

        let data: Vec<f64> = (0..1024).map(|i| i as f64).collect();
        let mut buffer = device
            .alloc_uninit::<f64>(data.len())
            .expect("alloc_uninit failed");
        assert_eq!(buffer.len(), data.len());

        // Fulfill the alloc_uninit contract: overwrite every element before reading.
        device
            .stream
            .memcpy_htod(&data, &mut buffer)
            .expect("H2D copy failed");
        let result = device.copy_to_host(&buffer).expect("D2H copy failed");
        assert_eq!(result, data);
    }
}
