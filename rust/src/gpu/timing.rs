//! GPU Kernel Timing Infrastructure using CUDA Events
//!
//! Provides precise GPU-only timing measurements for kernel execution,
//! separating pure GPU performance from CPU overhead (memory allocation, transfers, etc.).
//!
//! # Why GPU-Only Timing Matters
//!
//! End-to-end timing (CPU clock) includes:
//! - Memory allocation (~1-2ms)
//! - H2D transfers (~25μs)
//! - **GPU kernel execution** (target measurement)
//! - D2H transfers (~25μs)
//! - CPU overhead (~1-2ms)
//!
//! For ATR example:
//! - End-to-end: ~1.36ms (measured with `Instant::now()`)
//! - GPU-only: ~145μs (measured with CUDA events)
//! - **9.4x difference!** CPU overhead dominates.
//!
//! # CUDA Event Timing Pattern
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::timing::GpuTimer;
//!
//! let timer = GpuTimer::new(&device)?;
//!
//! // Record start event before kernel
//! timer.start()?;
//!
//! // Launch GPU kernel(s)
//! indicator_gpu(&device, &data, period, None)?;
//!
//! // Record end event and get elapsed time
//! let gpu_time_us = timer.stop_micros()?;
//!
//! println!("GPU-only kernel time: {} μs", gpu_time_us);
//! ```
//!
//! # Use Cases
//!
//! 1. **Validate optimization claims**: Measure GPU speedup without CPU noise
//! 2. **Profile bottlenecks**: Identify slow kernels vs slow CPU code
//! 3. **Compare implementations**: Pure GPU timing for apples-to-apples comparison
//! 4. **Multi-phase timing**: Break down complex operations (H2D → Kernel → D2H)
//!
//! # Performance
//!
//! - Event creation: ~10-20ns
//! - Event recording: ~5-10ns (non-blocking)
//! - Elapsed time query: ~50-100ns
//! - **Negligible overhead** for μs-scale kernel timing

use super::async_transfers::CudaEvent;
use super::device::{GpuDevice, GpuError};
use cudarc::driver::sys;
use std::sync::Arc;

/// GPU timer for precise kernel timing using CUDA events
///
/// Measures GPU-only execution time, excluding CPU overhead.
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::timing::GpuTimer;
/// use kimsfinance_core::gpu::device::GpuDevice;
/// use kimsfinance_core::gpu::atr::atr_gpu;
///
/// let device = GpuDevice::new()?;
/// let timer = GpuTimer::new(&device)?;
///
/// // Warm up (exclude compilation)
/// for _ in 0..5 {
///     let _ = atr_gpu(&device, &high, &low, &close, 14, None)?;
/// }
/// device.synchronize()?;
///
/// // Measure GPU-only time
/// timer.start()?;
/// let _ = atr_gpu(&device, &high, &low, &close, 14, None)?;
/// let gpu_us = timer.stop_micros()?;
///
/// println!("GPU kernel time: {} μs", gpu_us);
/// ```
pub struct GpuTimer {
    device: Arc<GpuDevice>,
    start_event: CudaEvent,
    end_event: CudaEvent,
}

impl GpuTimer {
    /// Create a new GPU timer
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    ///
    /// # Errors
    ///
    /// Returns error if CUDA event creation fails.
    ///
    /// # Performance
    ///
    /// Creation overhead: ~20-40ns (2 events)
    pub fn new(device: &GpuDevice) -> Result<Self, GpuError> {
        Ok(Self {
            device: Arc::new(GpuDevice {
                context: device.context.clone(),
                stream: device.stream.clone(),
                device_id: device.device_id,
                pinned_pool: parking_lot::Mutex::new(
                    super::persistent::pinned_memory::PinnedBufferPool::new(0, 0)?,
                ),
                async_allocator: device.async_allocator.clone(),
                module_cache: dashmap::DashMap::new(),
            }),
            start_event: CudaEvent::new()?,
            end_event: CudaEvent::new()?,
        })
    }

    /// Record start event on the device stream
    ///
    /// Call this immediately before the GPU work you want to time.
    ///
    /// # Errors
    ///
    /// Returns error if event recording fails.
    ///
    /// # Performance
    ///
    /// Recording overhead: ~5-10ns (non-blocking)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// timer.start()?;
    /// // GPU work here
    /// let elapsed = timer.stop_micros()?;
    /// ```
    pub fn start(&self) -> Result<(), GpuError> {
        self.start_event.record(&self.device.stream)
    }

    /// Record end event and calculate elapsed time in microseconds
    ///
    /// Blocks until GPU work completes, then returns precise elapsed time.
    ///
    /// # Returns
    ///
    /// GPU-only execution time in microseconds (μs).
    ///
    /// # Errors
    ///
    /// Returns error if event recording or synchronization fails.
    ///
    /// # Performance
    ///
    /// - Recording: ~5-10ns
    /// - Synchronization: blocks until GPU finishes
    /// - Elapsed query: ~50-100ns
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// timer.start()?;
    /// indicator_gpu(&device, &data, period, None)?;
    /// let gpu_us = timer.stop_micros()?;
    /// println!("GPU time: {} μs", gpu_us);
    /// ```
    pub fn stop_micros(&self) -> Result<u64, GpuError> {
        // Record end event
        self.end_event.record(&self.device.stream)?;

        // Wait for GPU to finish
        self.end_event.synchronize()?;

        // Calculate elapsed time in milliseconds
        let elapsed_ms = self.elapsed_time_ms()?;

        // Convert to microseconds
        Ok((elapsed_ms * 1000.0) as u64)
    }

    /// Record end event and calculate elapsed time in milliseconds
    ///
    /// Blocks until GPU work completes, then returns precise elapsed time.
    ///
    /// # Returns
    ///
    /// GPU-only execution time in milliseconds (ms).
    ///
    /// # Errors
    ///
    /// Returns error if event recording or synchronization fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// timer.start()?;
    /// heavy_kernel(&device)?;
    /// let gpu_ms = timer.stop_millis()?;
    /// println!("GPU time: {:.2} ms", gpu_ms);
    /// ```
    pub fn stop_millis(&self) -> Result<f32, GpuError> {
        // Record end event
        self.end_event.record(&self.device.stream)?;

        // Wait for GPU to finish
        self.end_event.synchronize()?;

        // Calculate elapsed time
        self.elapsed_time_ms()
    }

    /// Get elapsed time between start and end events (internal)
    ///
    /// # Returns
    ///
    /// Elapsed time in milliseconds (with microsecond precision).
    ///
    /// # Errors
    ///
    /// Returns error if elapsed time query fails.
    fn elapsed_time_ms(&self) -> Result<f32, GpuError> {
        unsafe {
            let mut ms = 0.0f32;
            sys::cuEventElapsedTime_v2(
                &mut ms,
                self.start_event.raw_event(),
                self.end_event.raw_event(),
            )
            .result()
            .map_err(|e| {
                GpuError::ExecutionError(format!("Failed to get elapsed time: {:?}", e))
            })?;
            Ok(ms)
        }
    }

    /// Reset timer for reuse
    ///
    /// Not strictly necessary (can just call `start()` again), but provided
    /// for API clarity.
    pub fn reset(&self) -> Result<(), GpuError> {
        // Events are automatically reused when record() is called again
        Ok(())
    }
}

/// Multi-phase GPU timer for detailed breakdowns
///
/// Records multiple events to measure different phases of GPU execution:
/// - H2D transfer time
/// - Kernel execution time
/// - D2H transfer time
///
/// # Example
///
/// ```rust,ignore
/// use kimsfinance_core::gpu::timing::MultiPhaseTimer;
///
/// let timer = MultiPhaseTimer::new(&device)?;
///
/// timer.record_start()?;
///
/// // Phase 1: H2D transfer
/// device.copy_to_device(&data)?;
/// timer.record_h2d_done()?;
///
/// // Phase 2: Kernel execution
/// launch_kernel(&device)?;
/// timer.record_kernel_done()?;
///
/// // Phase 3: D2H transfer
/// let result = device.copy_to_host(&device_buffer)?;
/// timer.record_d2h_done()?;
///
/// // Get breakdown
/// let breakdown = timer.get_breakdown()?;
/// println!("H2D:    {:.2} μs", breakdown.h2d_us);
/// println!("Kernel: {:.2} μs", breakdown.kernel_us);
/// println!("D2H:    {:.2} μs", breakdown.d2h_us);
/// println!("Total:  {:.2} μs", breakdown.total_us);
/// ```
pub struct MultiPhaseTimer {
    device: Arc<GpuDevice>,
    start_event: CudaEvent,
    h2d_event: CudaEvent,
    kernel_event: CudaEvent,
    d2h_event: CudaEvent,
}

/// Timing breakdown for multi-phase execution
#[derive(Debug, Clone, Copy)]
pub struct TimingBreakdown {
    /// H2D transfer time (microseconds)
    pub h2d_us: f32,
    /// Kernel execution time (microseconds)
    pub kernel_us: f32,
    /// D2H transfer time (microseconds)
    pub d2h_us: f32,
    /// Total GPU time (microseconds)
    pub total_us: f32,
}

impl TimingBreakdown {
    /// Calculate transfer overhead percentage
    pub fn transfer_overhead_pct(&self) -> f32 {
        ((self.h2d_us + self.d2h_us) / self.total_us) * 100.0
    }

    /// Calculate kernel percentage of total time
    pub fn kernel_pct(&self) -> f32 {
        (self.kernel_us / self.total_us) * 100.0
    }

    /// Print formatted breakdown report
    pub fn print_report(&self, name: &str) {
        println!("\n╔════════════════════════════════════════════╗");
        println!("║  GPU Timing Breakdown: {:<20} ║", name);
        println!("╠════════════════════════════════════════════╣");
        println!("║  Phase          Time (μs)    % of Total    ║");
        println!("╟────────────────────────────────────────────╢");
        println!(
            "║  H2D Transfer   {:>8.2}       {:>5.1}%       ║",
            self.h2d_us,
            (self.h2d_us / self.total_us) * 100.0
        );
        println!(
            "║  Kernel Exec    {:>8.2}       {:>5.1}%       ║",
            self.kernel_us,
            self.kernel_pct()
        );
        println!(
            "║  D2H Transfer   {:>8.2}       {:>5.1}%       ║",
            self.d2h_us,
            (self.d2h_us / self.total_us) * 100.0
        );
        println!("╟────────────────────────────────────────────╢");
        println!(
            "║  Total GPU      {:>8.2}       100.0%       ║",
            self.total_us
        );
        println!("╠════════════════════════════════════════════╣");
        println!(
            "║  Transfer Overhead: {:.1}%                  ║",
            self.transfer_overhead_pct()
        );
        println!("╚════════════════════════════════════════════╝");
    }
}

impl MultiPhaseTimer {
    /// Create a new multi-phase timer
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    pub fn new(device: &GpuDevice) -> Result<Self, GpuError> {
        Ok(Self {
            device: Arc::new(GpuDevice {
                context: device.context.clone(),
                stream: device.stream.clone(),
                device_id: device.device_id,
                pinned_pool: parking_lot::Mutex::new(
                    super::persistent::pinned_memory::PinnedBufferPool::new(0, 0)?,
                ),
                async_allocator: device.async_allocator.clone(),
                module_cache: dashmap::DashMap::new(),
            }),
            start_event: CudaEvent::new()?,
            h2d_event: CudaEvent::new()?,
            kernel_event: CudaEvent::new()?,
            d2h_event: CudaEvent::new()?,
        })
    }

    /// Record start of timing
    pub fn record_start(&self) -> Result<(), GpuError> {
        self.start_event.record(&self.device.stream)
    }

    /// Record completion of H2D transfer
    pub fn record_h2d_done(&self) -> Result<(), GpuError> {
        self.h2d_event.record(&self.device.stream)
    }

    /// Record completion of kernel execution
    pub fn record_kernel_done(&self) -> Result<(), GpuError> {
        self.kernel_event.record(&self.device.stream)
    }

    /// Record completion of D2H transfer
    pub fn record_d2h_done(&self) -> Result<(), GpuError> {
        self.d2h_event.record(&self.device.stream)
    }

    /// Get timing breakdown for all phases
    ///
    /// Blocks until all GPU work completes.
    ///
    /// # Returns
    ///
    /// Detailed timing breakdown with per-phase measurements.
    pub fn get_breakdown(&self) -> Result<TimingBreakdown, GpuError> {
        // Wait for all work to complete
        self.d2h_event.synchronize()?;

        unsafe {
            let mut h2d_ms = 0.0f32;
            let mut kernel_ms = 0.0f32;
            let mut d2h_ms = 0.0f32;
            let mut total_ms = 0.0f32;

            // H2D time: start → h2d_event
            sys::cuEventElapsedTime_v2(
                &mut h2d_ms,
                self.start_event.raw_event(),
                self.h2d_event.raw_event(),
            )
            .result()
            .map_err(|e| GpuError::ExecutionError(format!("Failed to get H2D time: {:?}", e)))?;

            // Kernel time: h2d_event → kernel_event
            sys::cuEventElapsedTime_v2(
                &mut kernel_ms,
                self.h2d_event.raw_event(),
                self.kernel_event.raw_event(),
            )
            .result()
            .map_err(|e| GpuError::ExecutionError(format!("Failed to get kernel time: {:?}", e)))?;

            // D2H time: kernel_event → d2h_event
            sys::cuEventElapsedTime_v2(
                &mut d2h_ms,
                self.kernel_event.raw_event(),
                self.d2h_event.raw_event(),
            )
            .result()
            .map_err(|e| GpuError::ExecutionError(format!("Failed to get D2H time: {:?}", e)))?;

            // Total time: start → d2h_event
            sys::cuEventElapsedTime_v2(
                &mut total_ms,
                self.start_event.raw_event(),
                self.d2h_event.raw_event(),
            )
            .result()
            .map_err(|e| GpuError::ExecutionError(format!("Failed to get total time: {:?}", e)))?;

            Ok(TimingBreakdown {
                h2d_us: h2d_ms * 1000.0,
                kernel_us: kernel_ms * 1000.0,
                d2h_us: d2h_ms * 1000.0,
                total_us: total_ms * 1000.0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_timer() {
        let device = GpuDevice::new().expect("GPU required");
        let timer = GpuTimer::new(&device).expect("Failed to create timer");

        // Simple timing test
        timer.start().expect("Failed to start timer");

        // Allocate and copy some data (minimal GPU work)
        let data = vec![1.0f64; 1000];
        let _ = device.copy_to_device(&data).expect("Failed to copy data");

        let elapsed_us = timer.stop_micros().expect("Failed to stop timer");

        println!("GPU time: {} μs", elapsed_us);
        assert!(elapsed_us > 0);
        assert!(elapsed_us < 10_000); // Should be < 10ms for simple allocation
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_multi_phase_timer() {
        let device = GpuDevice::new().expect("GPU required");
        let timer = MultiPhaseTimer::new(&device).expect("Failed to create timer");

        let data = vec![1.0f64; 10_000];

        timer.record_start().expect("Failed to record start");

        // H2D
        let device_buf = device
            .copy_to_device(&data)
            .expect("Failed to copy to device");
        timer.record_h2d_done().expect("Failed to record H2D");

        // No kernel (just testing timing infrastructure)
        timer.record_kernel_done().expect("Failed to record kernel");

        // D2H
        let _ = device
            .copy_to_host(&device_buf)
            .expect("Failed to copy to host");
        timer.record_d2h_done().expect("Failed to record D2H");

        let breakdown = timer.get_breakdown().expect("Failed to get breakdown");

        println!("\nMulti-phase timing test:");
        breakdown.print_report("Test");

        assert!(breakdown.total_us > 0.0);
        assert!(breakdown.h2d_us >= 0.0);
        assert!(breakdown.d2h_us >= 0.0);
    }
}
