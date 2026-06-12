//! Page-Locked (Pinned) Memory for GPU Transfers
//!
//! Provides 20-30% faster H2D/D2H transfers using CUDA pinned memory.
//!
//! # Problem
//!
//! Regular pageable memory requires the CUDA driver to:
//! 1. Page-lock the memory region during transfer
//! 2. Perform the DMA transfer
//! 3. Unpin the memory after transfer
//!
//! This adds 20-30% overhead to every transfer operation.
//!
//! # Solution: Pinned Memory
//!
//! Pre-allocate page-locked memory that:
//! - Cannot be paged out by OS
//! - Allows direct DMA without intermediate copy
//! - Provides 2-3x faster transfers than pageable memory
//!
//! # Performance
//!
//! - **H2D transfers**: 20-30% faster
//! - **D2H transfers**: 20-30% faster
//! - **Memory overhead**: Limited to ~50% of system RAM
//!
//! # Architecture
//!
//! ```text
//! Pageable Memory (traditional):
//!   Host Vec → [OS page-lock] → DMA → GPU
//!   Overhead: 20-30%
//!
//! Pinned Memory (optimized):
//!   Host Vec → Pinned Buffer → DMA → GPU
//!   Overhead: 0% (pre-locked)
//! ```
//!
//! # Pool Hardening (Safety Semantics)
//!
//! The pool provides three release paths with different safety guarantees:
//!
//! 1. [`PinnedBufferPool::release`] — **requires a prior stream synchronization**.
//!    Returning a buffer while an async DMA from/to it is still in flight lets
//!    the next [`PinnedBufferPool::acquire`] hand the same memory to another
//!    transfer, silently corrupting data.
//! 2. [`PinnedBufferPool::release_with_event`] — safe for async transfers.
//!    The buffer is parked on a pending list together with a [`CudaEvent`]
//!    recorded after the transfer; `acquire()` sweeps the pending list and only
//!    recycles buffers whose event has completed.
//! 3. [`PinnedGuard`] — RAII wrapper returned by
//!    [`PinnedBufferPool::acquire_guard`]. Dropping the guard returns the
//!    buffer to the pool even on `?` early-return paths, fixing permanent pool
//!    drain on error paths.
//!
//! Requests larger than the largest pooled buffer no longer hard-error:
//! `acquire()` falls back to a one-off pinned allocation of exactly the
//! requested size, and `release()` drops such oversize buffers instead of
//! pooling them. This restores >1M-element sweep entry points
//! (e.g. `rsi_sweep_3d_gpu`, `sma_sweep_3d_gpu`, `sharpe_reduction_gpu`).
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::persistent::pinned_memory::PinnedBuffer;
//!
//! // Allocate pinned buffer
//! let mut buffer = PinnedBuffer::new(1000)?;
//!
//! // Copy from regular Vec
//! let data = vec![1.0, 2.0, 3.0];
//! buffer.copy_from_slice(&data);
//!
//! // DMA transfer to GPU (20-30% faster!)
//! device.htod_copy_into(buffer.as_slice(), &mut d_buffer)?;
//! ```

use cudarc::driver::{CudaEvent, sys};
use parking_lot::Mutex;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::gpu::GpuError;

/// Small pool tier: covers typical single-indicator inputs (8 MB for `f64`).
pub const PINNED_TIER_SMALL: usize = 1_000_000;

/// Large pool tier: covers multi-million-element sweep slabs (64 MB for `f64`).
pub const PINNED_TIER_LARGE: usize = 8_000_000;

/// Default tiered pool layout as `(buffer_count, buffer_size)` pairs.
///
/// For `f64` this allocates 12 × 8 MB + 2 × 64 MB = 224 MB of pinned memory,
/// comparable to the legacy 16 × 8 MB layout while also serving 1M-8M element
/// requests from the pool instead of the one-off oversize fallback.
pub const DEFAULT_PINNED_TIERS: [(usize, usize); 2] =
    [(12, PINNED_TIER_SMALL), (2, PINNED_TIER_LARGE)];

/// Page-locked (pinned) host memory buffer
///
/// Provides faster GPU transfers by pre-locking memory pages.
///
/// # Memory Allocation Flags
///
/// - `CU_MEMHOSTALLOC_DEVICEMAP`: Enables zero-copy access from GPU
/// - `CU_MEMHOSTALLOC_PORTABLE`: Allows use with multiple CUDA contexts
///
/// # Safety
///
/// - Memory is properly aligned for DMA transfers
/// - Automatically freed via RAII (Drop trait)
/// - Send + Sync safe (pinned memory is thread-safe)
pub struct PinnedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

impl<T> PinnedBuffer<T> {
    /// Allocate pinned memory using cuMemHostAlloc
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements to allocate
    ///
    /// # Errors
    ///
    /// Returns `GpuError::AllocationError` if:
    /// - Pinned memory limit exceeded (~50% of system RAM)
    /// - CUDA driver not initialized
    /// - Invalid size (0 or too large)
    ///
    /// # Performance
    ///
    /// Pinned memory is a limited resource. On allocation failure,
    /// caller should gracefully fall back to pageable memory.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let buffer = match PinnedBuffer::new(1000) {
    ///     Ok(buf) => buf,
    ///     Err(_) => {
    ///         eprintln!("Pinned allocation failed, using pageable memory");
    ///         return allocate_pageable_fallback(1000);
    ///     }
    /// };
    /// ```
    pub fn new(len: usize) -> Result<Self, GpuError> {
        if len == 0 {
            return Err(GpuError::AllocationError(
                "Cannot allocate zero-length pinned buffer".to_string(),
            ));
        }

        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let size = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                GpuError::AllocationError(format!(
                    "Size overflow: {} elements of {} bytes",
                    len,
                    std::mem::size_of::<T>()
                ))
            })?;

            // Allocate pinned memory with flags:
            // - DEVICEMAP: Enable zero-copy GPU access
            // - PORTABLE: Allow use with multiple CUDA contexts
            let result = sys::cuMemHostAlloc(
                &mut ptr,
                size,
                sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
            );

            result.result().map_err(|e| {
                GpuError::AllocationError(format!(
                    "Failed to allocate {} bytes of pinned memory: {:?}. \
                     Pinned memory is limited to ~50% of system RAM. \
                     Consider using pageable memory fallback.",
                    size, e
                ))
            })?;

            let non_null = NonNull::new(ptr as *mut T).ok_or_else(|| {
                GpuError::AllocationError("cuMemHostAlloc returned null pointer".to_string())
            })?;

            Ok(Self {
                ptr: non_null,
                len,
                _marker: PhantomData,
            })
        }
    }

    /// Get immutable slice of pinned memory
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get mutable slice of pinned memory
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Copy from regular slice into pinned buffer
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != self.len()`. Use with pre-validated buffers.
    #[inline]
    pub fn copy_from_slice(&mut self, data: &[T])
    where
        T: Copy,
    {
        assert_eq!(
            data.len(),
            self.len,
            "Source slice length {} does not match buffer length {}",
            data.len(),
            self.len
        );
        self.as_mut_slice().copy_from_slice(data);
    }

    /// Get buffer length
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw pointer (for CUDA operations)
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - Pointer is not used after buffer is dropped
    /// - Pointer is not dereferenced on CPU if GPU is modifying it
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer (for CUDA operations)
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - Pointer is not used after buffer is dropped
    /// - No data races between CPU and GPU access
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            // Free pinned memory
            let _ = sys::cuMemFreeHost(self.ptr.as_ptr() as *mut std::ffi::c_void);
        }
    }
}

/// Abstraction over CUDA event completion queries.
///
/// Allows the pool's pending-list sweep logic to be unit-tested host-side
/// (without a GPU) using mock events, while production code uses
/// [`CudaEvent::is_complete`] (`cuEventQuery`).
pub trait CompletionQuery {
    /// Returns `true` once all GPU work recorded before this event has finished.
    fn is_complete(&self) -> bool;
}

impl CompletionQuery for CudaEvent {
    #[inline]
    fn is_complete(&self) -> bool {
        // Inherent method (cuEventQuery); takes precedence over this trait method.
        CudaEvent::is_complete(self)
    }
}

/// Move completed entries out of a pending list, preserving FIFO order of the
/// still-pending remainder.
///
/// Each `(buffer, event)` pair whose event reports complete is handed to
/// `recycle`; incomplete pairs are returned in their original relative order.
/// Pure host-side logic — exercised by unit tests with mock events.
fn partition_completed<B, E: CompletionQuery>(
    pending: Vec<(B, E)>,
    mut recycle: impl FnMut(B),
) -> Vec<(B, E)> {
    let mut still_pending = Vec::with_capacity(pending.len());
    for (buffer, event) in pending {
        if event.is_complete() {
            recycle(buffer);
        } else {
            still_pending.push((buffer, event));
        }
    }
    still_pending
}

/// Index of the **smallest** buffer with `len >= size` in a descending-sorted
/// list, or `None` if nothing fits.
///
/// In a descending list every fitting buffer forms a prefix, so the rightmost
/// fitting element is the smallest fit (true best-fit).
fn best_fit_index<B>(bufs: &[B], len_of: impl Fn(&B) -> usize, size: usize) -> Option<usize> {
    bufs.iter().rposition(|b| len_of(b) >= size)
}

/// Insertion index that keeps a descending-sorted (by length) list sorted.
fn descending_insert_index<B>(bufs: &[B], len_of: impl Fn(&B) -> usize, len: usize) -> usize {
    // For a descending-sorted slice, `target.cmp(probe)` is monotonically
    // non-decreasing along the slice, satisfying binary_search's contract.
    bufs.binary_search_by(|probe| len.cmp(&len_of(probe)))
        .unwrap_or_else(|pos| pos)
}

/// Pool of reusable pinned buffers
///
/// Avoids repeated allocation overhead by reusing buffers.
///
/// # Architecture
///
/// ```text
/// PinnedBufferPool
///   ├── Available: [buf1, buf2, buf3]          (sorted by size, largest first)
///   ├── Pending:   [(buf4, ev4), (buf5, ev5)]  (DMA in flight, event-gated)
///   └── In-use:    [buf6, buf7]                (currently borrowed by callers)
/// ```
///
/// # Invariants
///
/// - `available` is sorted by length, largest first, and only contains
///   buffers with `len <= standard_size` (oversize one-offs are dropped on
///   release, never pooled).
/// - `pending` holds buffers whose async transfer may still be in flight;
///   they are only moved to `available` once their recorded event completes.
///
/// # Example
///
/// ```rust,ignore
/// let mut pool = PinnedBufferPool::new(5, 1000)?;
///
/// // Acquire buffer from pool
/// let mut buffer = pool.acquire(1000)?;
/// buffer.copy_from_slice(&data);
///
/// // Async transfer to GPU, then event-gated release (no sync required)
/// stream.memcpy_htod(buffer.as_slice(), &mut d_buffer)?;
/// let event = context.new_event(None)?;
/// event.record(&stream)?;
/// pool.release_with_event(buffer, event);
///
/// // Or, with RAII (buffer returns to the pool even on `?` early returns):
/// let pool = Arc::new(Mutex::new(PinnedBufferPool::new(5, 1000)?));
/// let mut guard = PinnedBufferPool::acquire_guard(&pool, 1000)?;
/// guard.copy_from_slice(&data);
/// stream.memcpy_htod(guard.as_slice(), &mut d_buffer)?;
/// stream.synchronize()?; // plain drop-release requires a sync
/// drop(guard);
/// ```
pub struct PinnedBufferPool<T> {
    /// Available buffers sorted by size (largest first)
    available: Vec<PinnedBuffer<T>>,
    /// Buffers with potentially in-flight DMA, gated on event completion
    pending: Vec<(PinnedBuffer<T>, CudaEvent)>,
    /// Largest pooled buffer size; requests above this use one-off allocations
    standard_size: usize,
    /// Whether the oversize-fallback debug message has been emitted
    oversize_logged: bool,
}

impl<T> PinnedBufferPool<T> {
    /// Create new pinned buffer pool
    ///
    /// # Arguments
    ///
    /// * `buffer_count` - Number of buffers to pre-allocate
    /// * `buffer_size` - Size of each buffer (in elements)
    ///
    /// # Errors
    ///
    /// Returns error if pinned allocation fails for any buffer.
    /// Caller should handle gracefully with pageable fallback.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Pre-allocate 10 buffers of 1000 elements each
    /// let pool = PinnedBufferPool::<f64>::new(10, 1000)?;
    /// ```
    pub fn new(buffer_count: usize, buffer_size: usize) -> Result<Self, GpuError> {
        let mut available = Vec::with_capacity(buffer_count);

        for i in 0..buffer_count {
            match PinnedBuffer::new(buffer_size) {
                Ok(buffer) => available.push(buffer),
                Err(e) => {
                    return Err(GpuError::AllocationError(format!(
                        "Failed to allocate pinned buffer {} of {}: {}. \
                         Successfully allocated {} buffers before failure.",
                        i + 1,
                        buffer_count,
                        e,
                        i
                    )));
                }
            }
        }

        Ok(Self {
            available,
            pending: Vec::new(),
            standard_size: buffer_size,
            oversize_logged: false,
        })
    }

    /// Create a pool with multiple size tiers
    ///
    /// # Arguments
    ///
    /// * `tiers` - `(buffer_count, buffer_size)` pairs, e.g.
    ///   [`DEFAULT_PINNED_TIERS`] (12 × 1M + 2 × 8M elements)
    ///
    /// The pool's `standard_size` becomes the largest tier size; requests up
    /// to that size are served from the pool (best-fit picks the smallest
    /// fitting tier), larger requests use the one-off oversize fallback.
    ///
    /// # Errors
    ///
    /// Returns error if pinned allocation fails for any buffer.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pool = PinnedBufferPool::<f64>::with_tiers(&DEFAULT_PINNED_TIERS)?;
    /// ```
    pub fn with_tiers(tiers: &[(usize, usize)]) -> Result<Self, GpuError> {
        let mut available = Vec::with_capacity(tiers.iter().map(|&(count, _)| count).sum());
        let mut standard_size = 0;

        for &(buffer_count, buffer_size) in tiers {
            for i in 0..buffer_count {
                match PinnedBuffer::new(buffer_size) {
                    Ok(buffer) => available.push(buffer),
                    Err(e) => {
                        return Err(GpuError::AllocationError(format!(
                            "Failed to allocate pinned tier buffer {} of {} \
                             (size {} elements): {}. \
                             Successfully allocated {} buffers before failure.",
                            i + 1,
                            buffer_count,
                            buffer_size,
                            e,
                            available.len()
                        )));
                    }
                }
            }
            standard_size = standard_size.max(buffer_size);
        }

        // Maintain the descending-by-length invariant
        available.sort_by_key(|buf| std::cmp::Reverse(buf.len()));

        Ok(Self {
            available,
            pending: Vec::new(),
            standard_size,
            oversize_logged: false,
        })
    }

    /// Create a pool with the default size tiers ([`DEFAULT_PINNED_TIERS`])
    pub fn with_default_tiers() -> Result<Self, GpuError> {
        Self::with_tiers(&DEFAULT_PINNED_TIERS)
    }

    /// Acquire buffer from pool
    ///
    /// # Arguments
    ///
    /// * `size` - Required buffer size (in elements)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No pooled buffer fits and `size <= standard_size` (all in use)
    /// - `size > standard_size` and the one-off pinned allocation fails
    ///
    /// # Strategy
    ///
    /// 1. Sweep the pending list, recycling buffers whose events completed
    /// 2. If `size > standard_size`: allocate a one-off pinned buffer of
    ///    exactly `size` (it is dropped, not pooled, on release)
    /// 3. Otherwise: best-fit — the smallest available buffer with
    ///    `len >= size`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut buffer = pool.acquire(500)?;
    /// // Use buffer...
    /// pool.release(buffer);
    /// ```
    pub fn acquire(&mut self, size: usize) -> Result<PinnedBuffer<T>, GpuError> {
        // Reclaim buffers whose in-flight DMA has completed
        self.sweep_pending();

        // Oversize fallback: one-off allocation outside the pool. Restores
        // >standard_size entry points (3D sweeps, Sharpe reductions) that
        // previously hard-errored here.
        if size > self.standard_size {
            if !self.oversize_logged {
                self.oversize_logged = true;
                eprintln!(
                    "PinnedBufferPool: oversize request ({} elements > standard size {}), \
                     allocating one-off pinned buffer. Further oversize messages suppressed.",
                    size, self.standard_size
                );
            }
            return PinnedBuffer::new(size);
        }

        let idx = best_fit_index(&self.available, |b| b.len(), size).ok_or_else(|| {
            GpuError::AllocationError(format!(
                "No pinned buffer available for size {}. \
                 Pool has {} buffers available, {} pending DMA completion, \
                 standard size {}.",
                size,
                self.available.len(),
                self.pending.len(),
                self.standard_size
            ))
        })?;

        // Ordered remove preserves the descending-sort invariant
        // (the previous swap_remove silently broke it).
        Ok(self.available.remove(idx))
    }

    /// Acquire a buffer wrapped in an RAII guard
    ///
    /// The guard returns the buffer to the pool when dropped, including on
    /// `?` early-return paths — without this, any error between `acquire` and
    /// `release` permanently drains the pool.
    ///
    /// # Arguments
    ///
    /// * `pool` - Shared handle to the pool (the guard keeps a clone)
    /// * `size` - Required buffer size (in elements)
    ///
    /// # Errors
    ///
    /// Same as [`PinnedBufferPool::acquire`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pool = Arc::new(Mutex::new(PinnedBufferPool::<f64>::new(8, 1_000_000)?));
    /// let mut guard = PinnedBufferPool::acquire_guard(&pool, n)?;
    /// guard.as_mut_slice()[..n].copy_from_slice(&host_data);
    /// stream.memcpy_htod(&guard.as_slice()[..n], &mut d_buf)?;
    /// stream.synchronize()?; // required before plain drop-release
    /// // guard drops here → buffer returns to the pool
    /// ```
    pub fn acquire_guard(
        pool: &Arc<Mutex<Self>>,
        size: usize,
    ) -> Result<PinnedGuard<T>, GpuError> {
        let buffer = pool.lock().acquire(size)?;
        Ok(PinnedGuard {
            buffer: Some(buffer),
            pool: Arc::clone(pool),
        })
    }

    /// Release buffer back to pool
    ///
    /// # Warning: synchronization required
    ///
    /// The caller **must** have synchronized the stream that last touched this
    /// buffer (e.g. `stream.synchronize()`) before calling `release()`. The
    /// buffer becomes immediately available to the next `acquire()`; releasing
    /// while an async `memcpy_htod`/`memcpy_dtoh` is still in flight lets two
    /// transfers share the same pinned memory and silently corrupts data. For
    /// async pipelines use [`PinnedBufferPool::release_with_event`] instead.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer to return to pool
    ///
    /// # Oversize buffers
    ///
    /// Buffers larger than `standard_size` (from the oversize fallback) are
    /// dropped here, freeing their pinned memory, instead of being pooled.
    pub fn release(&mut self, buffer: PinnedBuffer<T>) {
        self.recycle(buffer);
    }

    /// Release a buffer gated on a CUDA event
    ///
    /// The buffer is parked on the pending list and only becomes available to
    /// `acquire()` once `event` reports completion, making it safe to call
    /// immediately after enqueueing an async transfer — no stream sync needed.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer with potentially in-flight DMA
    /// * `event` - Event recorded on the transfer's stream **after** the last
    ///   operation touching `buffer` (e.g. `event.record(stream)` right after
    ///   `memcpy_htod`)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// stream.memcpy_htod(pinned.as_slice(), &mut d_buf)?;
    /// let event = context.new_event(None)?;
    /// event.record(&stream)?;
    /// pool.release_with_event(pinned, event); // safe: reuse gated on event
    /// ```
    pub fn release_with_event(&mut self, buffer: PinnedBuffer<T>, event: CudaEvent) {
        self.pending.push((buffer, event));
    }

    /// Move pending buffers whose events completed back to the free list
    fn sweep_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        let mut freed: Vec<PinnedBuffer<T>> = Vec::new();
        self.pending = partition_completed(pending, |buffer| freed.push(buffer));
        for buffer in freed {
            self.recycle(buffer);
        }
    }

    /// Return a buffer to the free list (or drop it if oversize)
    fn recycle(&mut self, buffer: PinnedBuffer<T>) {
        // Oversize one-offs are never pooled: dropping frees the pinned
        // allocation and keeps the pool's footprint bounded.
        if buffer.len() > self.standard_size {
            drop(buffer);
            return;
        }

        // Insert maintaining sorted order (largest first)
        let insert_pos = descending_insert_index(&self.available, |b| b.len(), buffer.len());
        self.available.insert(insert_pos, buffer);
    }

    /// Get number of available buffers (excludes pending event-gated buffers)
    #[inline]
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Get number of buffers awaiting event completion
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if pool is empty (all buffers in use or pending)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    /// Get standard buffer size (largest pooled tier)
    #[inline]
    pub fn standard_size(&self) -> usize {
        self.standard_size
    }
}

/// RAII guard returning a pinned buffer to its pool on drop
///
/// Created via [`PinnedBufferPool::acquire_guard`]. Dereferences to
/// [`PinnedBuffer`], so existing buffer methods work unchanged.
///
/// # Drop semantics
///
/// Dropping the guard performs a **plain** release (see the synchronization
/// warning on [`PinnedBufferPool::release`]): the caller must have
/// synchronized the stream that last touched the buffer. For async pipelines,
/// consume the guard with [`PinnedGuard::release_with_event`] instead.
///
/// # Deadlock
///
/// `Drop` locks the pool mutex — do not drop a guard while holding the pool
/// lock on the same thread (parking_lot mutexes are not reentrant).
pub struct PinnedGuard<T> {
    /// `Some` until the buffer is returned (drop or `release_with_event`)
    buffer: Option<PinnedBuffer<T>>,
    pool: Arc<Mutex<PinnedBufferPool<T>>>,
}

impl<T> PinnedGuard<T> {
    /// Consume the guard, releasing the buffer gated on `event`
    ///
    /// Safe to call immediately after enqueueing an async transfer; see
    /// [`PinnedBufferPool::release_with_event`].
    pub fn release_with_event(mut self, event: CudaEvent) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.lock().release_with_event(buffer, event);
        }
    }
}

impl<T> Deref for PinnedGuard<T> {
    type Target = PinnedBuffer<T>;

    #[inline]
    fn deref(&self) -> &PinnedBuffer<T> {
        // Invariant: Some until consumed by release_with_event (which takes
        // self by value) or Drop, so this cannot fail on live guards.
        self.buffer
            .as_ref()
            .expect("PinnedGuard buffer already released")
    }
}

impl<T> DerefMut for PinnedGuard<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut PinnedBuffer<T> {
        self.buffer
            .as_mut()
            .expect("PinnedGuard buffer already released")
    }
}

impl<T> Drop for PinnedGuard<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.lock().release(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuDevice;

    // ==================== Host-side tests (no GPU required) ====================

    /// Mock event for testing pending-sweep logic without a GPU
    struct MockEvent {
        complete: bool,
    }

    impl CompletionQuery for MockEvent {
        fn is_complete(&self) -> bool {
            self.complete
        }
    }

    #[test]
    fn test_partition_completed_moves_only_complete() {
        let pending = vec![
            (1usize, MockEvent { complete: true }),
            (2, MockEvent { complete: false }),
            (3, MockEvent { complete: true }),
            (4, MockEvent { complete: false }),
        ];

        let mut recycled = Vec::new();
        let still_pending = partition_completed(pending, |buf| recycled.push(buf));

        // Completed buffers recycled in scan order
        assert_eq!(recycled, vec![1, 3]);
        // Incomplete entries keep their original FIFO order
        let remaining: Vec<usize> = still_pending.iter().map(|(b, _)| *b).collect();
        assert_eq!(remaining, vec![2, 4]);
    }

    #[test]
    fn test_partition_completed_all_complete() {
        let pending = vec![
            (10usize, MockEvent { complete: true }),
            (20, MockEvent { complete: true }),
        ];

        let mut recycled = Vec::new();
        let still_pending = partition_completed(pending, |buf| recycled.push(buf));

        assert_eq!(recycled, vec![10, 20]);
        assert!(still_pending.is_empty());
    }

    #[test]
    fn test_partition_completed_none_complete() {
        let pending = vec![
            (10usize, MockEvent { complete: false }),
            (20, MockEvent { complete: false }),
        ];

        let mut recycled = Vec::new();
        let still_pending = partition_completed(pending, |buf| recycled.push(buf));

        assert!(recycled.is_empty());
        let remaining: Vec<usize> = still_pending.iter().map(|(b, _)| *b).collect();
        assert_eq!(remaining, vec![10, 20]);
    }

    #[test]
    fn test_partition_completed_empty() {
        let pending: Vec<(usize, MockEvent)> = Vec::new();
        let mut recycled = Vec::new();
        let still_pending = partition_completed(pending, |buf: usize| recycled.push(buf));
        assert!(recycled.is_empty());
        assert!(still_pending.is_empty());
    }

    #[test]
    fn test_best_fit_picks_smallest_fitting_buffer() {
        // Descending-sorted lengths (the pool invariant)
        let lens: Vec<usize> = vec![8_000_000, 1_000_000, 1_000_000, 1_000];

        // Tiny request → smallest buffer (rightmost fit), not the largest
        assert_eq!(best_fit_index(&lens, |&l| l, 500), Some(3));
        // 1M request → smallest 1M buffer
        assert_eq!(best_fit_index(&lens, |&l| l, 1_000_000), Some(2));
        // 2M request → only the 8M buffer fits
        assert_eq!(best_fit_index(&lens, |&l| l, 2_000_000), Some(0));
        // Nothing fits
        assert_eq!(best_fit_index(&lens, |&l| l, 9_000_000), None);
        // Empty pool
        assert_eq!(best_fit_index(&Vec::<usize>::new(), |&l| l, 1), None);
    }

    #[test]
    fn test_descending_insert_index_keeps_order() {
        let mut lens: Vec<usize> = vec![8_000_000, 1_000_000];

        // Larger than everything → front
        let pos = descending_insert_index(&lens, |&l| l, 10_000_000);
        assert_eq!(pos, 0);
        lens.insert(pos, 10_000_000);

        // Middle value
        let pos = descending_insert_index(&lens, |&l| l, 4_000_000);
        lens.insert(pos, 4_000_000);

        // Duplicate of an existing length
        let pos = descending_insert_index(&lens, |&l| l, 1_000_000);
        lens.insert(pos, 1_000_000);

        // Smaller than everything → back
        let pos = descending_insert_index(&lens, |&l| l, 500);
        assert_eq!(pos, lens.len());
        lens.insert(pos, 500);

        // Whole list stays descending after every insert
        assert!(lens.windows(2).all(|w| w[0] >= w[1]), "not descending: {:?}", lens);
        assert_eq!(
            lens,
            vec![10_000_000, 8_000_000, 4_000_000, 1_000_000, 1_000_000, 500]
        );
    }

    #[test]
    fn test_best_fit_and_insert_round_trip() {
        // Simulate acquire/release cycles on plain lengths to validate that
        // the two helpers preserve each other's invariants.
        let mut lens: Vec<usize> = vec![8_000_000, 1_000_000, 1_000_000];

        // Acquire best fit for a small request → smallest buffer leaves
        let idx = best_fit_index(&lens, |&l| l, 500).expect("fit");
        let taken = lens.remove(idx);
        assert_eq!(taken, 1_000_000);
        assert_eq!(lens, vec![8_000_000, 1_000_000]);

        // Release it back → descending order restored
        let pos = descending_insert_index(&lens, |&l| l, taken);
        lens.insert(pos, taken);
        assert_eq!(lens, vec![8_000_000, 1_000_000, 1_000_000]);
    }

    #[test]
    fn test_default_tiers_are_sane() {
        assert!(
            DEFAULT_PINNED_TIERS.iter().all(|&(count, size)| count > 0 && size > 0),
            "tiers must have non-zero counts and sizes"
        );
        assert_eq!(PINNED_TIER_SMALL, 1_000_000);
        assert_eq!(PINNED_TIER_LARGE, 8_000_000);
        assert!(PINNED_TIER_LARGE > PINNED_TIER_SMALL);
    }

    #[test]
    fn test_pinned_buffer_zero_length() {
        let result = PinnedBuffer::<f64>::new(0);
        assert!(result.is_err());
        match result {
            Err(GpuError::AllocationError(msg)) => {
                assert!(msg.contains("zero-length"));
            }
            _ => panic!("Expected AllocationError for zero length"),
        }
    }

    // ==================== GPU tests (require device) ====================

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_buffer_allocation() {
        let buffer = PinnedBuffer::<f64>::new(1000).expect("Pinned allocation failed");
        assert_eq!(buffer.len(), 1000);
        assert!(!buffer.is_empty());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_buffer_copy() {
        let mut buffer = PinnedBuffer::<f64>::new(5).expect("Pinned allocation failed");

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.copy_from_slice(&data);

        let slice = buffer.as_slice();
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[4], 5.0);
    }

    #[test]
    #[ignore] // Requires GPU
    #[should_panic(expected = "does not match")]
    fn test_pinned_buffer_copy_size_mismatch() {
        let mut buffer = PinnedBuffer::<f64>::new(5).expect("Pinned allocation failed");
        let data = vec![1.0, 2.0, 3.0]; // Wrong size
        buffer.copy_from_slice(&data); // Should panic
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_buffer_mutability() {
        let mut buffer = PinnedBuffer::<f64>::new(3).expect("Pinned allocation failed");

        // Modify via mutable slice
        {
            let slice = buffer.as_mut_slice();
            slice[0] = 10.0;
            slice[1] = 20.0;
            slice[2] = 30.0;
        }

        // Verify changes
        let slice = buffer.as_slice();
        assert_eq!(slice[0], 10.0);
        assert_eq!(slice[1], 20.0);
        assert_eq!(slice[2], 30.0);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_buffer_pool_creation() {
        let pool = PinnedBufferPool::<f64>::new(5, 1000).expect("Pool creation failed");
        assert_eq!(pool.available_count(), 5);
        assert_eq!(pool.pending_count(), 0);
        assert_eq!(pool.standard_size(), 1000);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_buffer_pool_tiers() {
        let mut pool =
            PinnedBufferPool::<f64>::with_tiers(&[(2, 1_000), (1, 8_000)]).expect("Pool creation");
        assert_eq!(pool.available_count(), 3);
        assert_eq!(pool.standard_size(), 8_000);

        // Best-fit: small request gets the small tier, leaving the large free
        let small = pool.acquire(500).expect("small tier acquire");
        assert_eq!(small.len(), 1_000);

        let large = pool.acquire(4_000).expect("large tier acquire");
        assert_eq!(large.len(), 8_000);

        pool.release(small);
        pool.release(large);
        assert_eq!(pool.available_count(), 3);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_buffer_pool_acquire_release() {
        let mut pool = PinnedBufferPool::<f64>::new(3, 1000).expect("Pool creation failed");

        // Acquire buffer
        let buffer = pool.acquire(500).expect("Acquire failed");
        assert_eq!(pool.available_count(), 2);

        // Release buffer
        pool.release(buffer);
        assert_eq!(pool.available_count(), 3);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_buffer_pool_exhaust() {
        let mut pool = PinnedBufferPool::<f64>::new(2, 1000).expect("Pool creation failed");

        // Acquire all buffers
        let buf1 = pool.acquire(500).expect("Acquire 1 failed");
        let buf2 = pool.acquire(500).expect("Acquire 2 failed");
        assert!(pool.is_empty());

        // Try to acquire when pool is empty (size <= standard, no fallback)
        let result = pool.acquire(500);
        assert!(result.is_err());

        // Release and try again
        pool.release(buf1);
        let _buf3 = pool
            .acquire(500)
            .expect("Acquire after release should succeed");

        // Cleanup
        pool.release(buf2);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_buffer_pool_oversize_fallback() {
        let mut pool = PinnedBufferPool::<f64>::new(2, 1000).expect("Pool creation failed");

        // Request smaller size (pooled path)
        let buffer_small = pool.acquire(500).expect("Small size should work");
        assert!(buffer_small.len() >= 500);
        assert!(buffer_small.len() <= 1000);

        // Request larger than standard size → one-off pinned allocation of
        // exactly the requested length (previously a hard error)
        let buffer_oversize = pool.acquire(5000).expect("Oversize fallback should work");
        assert_eq!(buffer_oversize.len(), 5000);
        // Pool inventory untouched by the one-off allocation
        assert_eq!(pool.available_count(), 1);

        // Releasing the oversize buffer drops it instead of pooling it
        pool.release(buffer_oversize);
        assert_eq!(pool.available_count(), 1);

        // Pooled buffer round-trips normally
        pool.release(buffer_small);
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pinned_guard_drop_returns_buffer() {
        let pool = Arc::new(Mutex::new(
            PinnedBufferPool::<f64>::new(2, 1000).expect("Pool creation failed"),
        ));

        // Normal scope exit returns the buffer
        {
            let mut guard = PinnedBufferPool::acquire_guard(&pool, 500).expect("Acquire failed");
            guard.as_mut_slice()[0] = 1.0;
            assert_eq!(pool.lock().available_count(), 1);
        }
        assert_eq!(pool.lock().available_count(), 2);

        // `?` early-return between acquire and release must not drain the pool
        fn faulty(pool: &Arc<Mutex<PinnedBufferPool<f64>>>) -> Result<(), GpuError> {
            let _guard = PinnedBufferPool::acquire_guard(pool, 500)?;
            Err(GpuError::ComputationError(
                "simulated mid-pipeline failure".to_string(),
            ))
        }
        assert!(faulty(&pool).is_err());
        assert_eq!(pool.lock().available_count(), 2);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_release_with_event_gated_reuse() {
        let device = GpuDevice::new().expect("GPU required");
        let mut pool = PinnedBufferPool::<f64>::new(2, 1000).expect("Pool creation failed");

        // Enqueue an async H2D transfer from a pinned buffer
        let mut pinned = pool.acquire(1000).expect("Acquire failed");
        pinned.as_mut_slice().fill(42.0);
        let mut d_buf = device.alloc_buffer(1000).expect("GPU allocation failed");
        device
            .stream
            .memcpy_htod(pinned.as_slice(), &mut d_buf)
            .expect("H2D copy failed");

        // Record completion event AFTER the transfer, then release immediately
        let event = device
            .context()
            .new_event(None)
            .expect("Event creation failed");
        event.record(&device.stream).expect("Event record failed");
        pool.release_with_event(pinned, event);
        assert_eq!(pool.pending_count(), 1);
        assert_eq!(pool.available_count(), 1);

        // After the stream drains, the next acquire sweeps the pending list
        device.stream.synchronize().expect("Sync failed");
        let reused = pool.acquire(1000).expect("Reacquire after event completion");
        assert_eq!(pool.pending_count(), 0);

        pool.release(reused);
        assert_eq!(pool.available_count(), 2);
    }

    // ==================== Transfer Speed Benchmark ====================
    // NOTE: This test validates pinned memory is faster than pageable memory.
    // Requires GPU and should show 20-30% improvement.

    #[test]
    #[ignore] // Requires GPU and manual verification
    fn test_pinned_vs_pageable_transfer_speed() {
        use std::time::Instant;

        let device = GpuDevice::new().expect("GPU required");
        let size = 10_000_000; // 10M elements (~80MB)
        let iterations = 100;

        // Allocate GPU buffer
        let mut d_buffer = device.alloc_buffer(size).expect("GPU allocation failed");

        // Test 1: Pageable memory
        let pageable_data = vec![1.0f64; size];
        let start = Instant::now();
        for _ in 0..iterations {
            device
                .stream
                .memcpy_htod(&pageable_data, &mut d_buffer)
                .expect("H2D copy failed");
        }
        device.synchronize().expect("Sync failed");
        let pageable_time = start.elapsed();

        // Test 2: Pinned memory
        let mut pinned_buffer = PinnedBuffer::<f64>::new(size).expect("Pinned allocation failed");
        pinned_buffer.copy_from_slice(&pageable_data);
        let start = Instant::now();
        for _ in 0..iterations {
            device
                .stream
                .memcpy_htod(pinned_buffer.as_slice(), &mut d_buffer)
                .expect("H2D copy failed");
        }
        device.synchronize().expect("Sync failed");
        let pinned_time = start.elapsed();

        // Calculate speedup
        let speedup = pageable_time.as_secs_f64() / pinned_time.as_secs_f64();

        println!("Transfer Speed Comparison:");
        println!(
            "  Pageable: {:?} ({:.2} GB/s)",
            pageable_time,
            calculate_bandwidth(size, iterations, pageable_time)
        );
        println!(
            "  Pinned:   {:?} ({:.2} GB/s)",
            pinned_time,
            calculate_bandwidth(size, iterations, pinned_time)
        );
        println!("  Speedup:  {:.2}x", speedup);

        // Expect 1.2-1.3x speedup (20-30% improvement)
        assert!(
            speedup >= 1.15,
            "Pinned memory should be at least 1.15x faster, got {:.2}x",
            speedup
        );
    }

    fn calculate_bandwidth(
        elements: usize,
        iterations: usize,
        duration: std::time::Duration,
    ) -> f64 {
        let bytes = elements * std::mem::size_of::<f64>() * iterations;
        let gb = bytes as f64 / 1e9;
        gb / duration.as_secs_f64()
    }
}
