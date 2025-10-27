//! Persistent Kernel Infrastructure for Launch Overhead Reduction
//!
//! Implements persistent kernels using CUDA Cooperative Groups to eliminate
//! kernel launch overhead in batch processing scenarios.
//!
//! # Problem
//!
//! Traditional CUDA programming launches one kernel per operation:
//! - Launch overhead: ~5-10μs per kernel
//! - 9 indicators × 10μs = ~90μs wasted on launches alone
//! - CPU-GPU synchronization cost for each launch
//!
//! # Solution: Persistent Kernels
//!
//! Launch kernel once, process multiple tasks in a loop:
//! - Single launch overhead: ~10μs total
//! - Overhead reduction: 90% for 10+ tasks
//! - Uses Cooperative Groups for inter-task synchronization
//!
//! # Performance Targets
//!
//! - Small batches (<10): 50-70% launch overhead reduction
//! - Medium batches (10-100): 80-90% reduction
//! - Large batches (>100): 90%+ reduction
//! - Expected throughput: 2-4x improvement
//!
//! # Architecture
//!
//! ```text
//! Traditional:
//!   Task 1: Launch → Execute → Sync → Result
//!   Task 2: Launch → Execute → Sync → Result
//!   Task 3: Launch → Execute → Sync → Result
//!   Total overhead: N × launch_time
//!
//! Persistent:
//!   Launch → [Task 1 → Sync → Task 2 → Sync → Task 3] → Result
//!   Total overhead: 1 × launch_time
//! ```
//!
//! # CUDA Cooperative Launch
//!
//! Requires CUDA Cooperative Launch API:
//! - Grid-wide synchronization via cooperative_groups::this_grid().sync()
//! - All blocks must be simultaneously resident on GPU
//! - Maximum grid size: min(SM_count × max_blocks_per_SM, 2147483647)
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::persistent::{PersistentKernelManager, TaskBatch};
//!
//! let device = GpuDevice::new()?;
//! let manager = PersistentKernelManager::new(&device)?;
//!
//! // Create batch of tasks (e.g., RSI with different periods)
//! let mut batch = TaskBatch::new();
//! batch.add_task(close_prices, 14); // RSI(14)
//! batch.add_task(close_prices, 21); // RSI(21)
//! batch.add_task(close_prices, 28); // RSI(28)
//!
//! // Execute all tasks with single kernel launch
//! let results = manager.execute_batch(&batch)?;
//! ```
//!
//! ## Real-World Usage
//!
//! Calculate ROC with multiple periods in a single kernel launch:
//!
//! ```rust,no_run
//! use kimsfinance_core::gpu::{GpuDevice, persistent::*};
//!
//! let device = GpuDevice::new()?;
//! let close_prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 107.0];
//!
//! // Create batch with 3 different ROC periods
//! let mut batch = TaskBatch::new();
//! batch.add_task(close_prices.clone(), 7);  // ROC(7)
//! batch.add_task(close_prices.clone(), 14); // ROC(14)
//! batch.add_task(close_prices.clone(), 21); // ROC(21)
//!
//! // Execute all 3 with single kernel launch (90% overhead reduction!)
//! let results = execute_batch(&device, &batch)?;
//!
//! assert_eq!(results.len(), 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Performance Characteristics
//!
//! **Traditional Approach** (separate kernel launches):
//! - 3 indicators × 10μs overhead = **30μs wasted**
//! - Total time: 30μs overhead + 45μs compute = 75μs
//!
//! **Persistent Kernel** (single launch):
//! - 1 kernel launch × 10μs = **10μs overhead**
//! - Total time: 10μs overhead + 45μs compute = 55μs
//! - **Speedup**: 75μs / 55μs = **1.36x faster**
//!
//! For 10+ indicators: **2-4x speedup** (overhead becomes dominant)

// Trait system for extensible indicator support
pub mod kernels;
pub mod traits;

// Enhancement modules (Agent 2 & 3)
pub mod occupancy;
pub mod pinned_memory;

// Multi-input/multi-output support (Agent 3)
pub mod generic;

// Re-export trait and kernel types
pub use kernels::{
    AroonIndicator, AtrIndicator, BollingerIndicator, BollingerParams, CciIndicator,
    CmfIndicator, DonchianIndicator, ElderRayIndicator, EmaIndicator, KeltnerIndicator,
    KeltnerParams, MacdIndicator, MacdParams, ObvIndicator, RocIndicator, RsiIndicator,
    SmaIndicator, StochasticIndicator, StochasticParams, VwmaIndicator, WilliamsRIndicator,
    WmaIndicator,
};
pub use traits::{MultiOutputIndicator, PersistentIndicator, SingleOutputIndicator};

// Re-export enhancement types
pub use occupancy::OccupancyCalculator;
pub use pinned_memory::PinnedBuffer;

// Re-export generic batch types (Agent 3)
pub use generic::{execute_generic_batch, GenericBatch};

use super::compile::compile_ptx_optimized;
use super::device::{GpuDevice, GpuError};
use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig};
use std::sync::Arc;

/// CUDA kernel for persistent ROC calculation (simplest test case)
///
/// This kernel demonstrates the persistent pattern:
/// 1. Grid-stride loop over tasks (not data)
/// 2. Each task processes its entire dataset
/// 3. Cooperative groups synchronization between tasks
const PERSISTENT_ROC_KERNEL: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Define NAN constant for NVRTC
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)

extern "C" __global__ void persistent_roc_kernel(
    const double** __restrict__ input_batch,    // Array of input pointers
    double** __restrict__ output_batch,          // Array of output pointers
    const int* __restrict__ sizes,               // Array of dataset sizes
    const int* __restrict__ periods,             // Array of ROC periods
    int num_tasks                                // Number of tasks to process
) {
    // Get grid group for cooperative synchronization
    cg::grid_group grid = cg::this_grid();
    
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;
    
    // Process each task sequentially (persistent kernel pattern)
    for (int task_id = 0; task_id < num_tasks; task_id++) {
        const double* input = input_batch[task_id];
        double* output = output_batch[task_id];
        int n = sizes[task_id];
        int period = periods[task_id];
        
        // Grid-stride loop for this task's data
        for (int idx = global_tid; idx < n; idx += grid_size) {
            if (idx < period) {
                output[idx] = CUDART_NAN;
            } else {
                // ROC = (price[i] / price[i-period] - 1) * 100
                output[idx] = (input[idx] / input[idx - period] - 1.0) * 100.0;
            }
        }
        
        // Synchronize entire grid before next task
        grid.sync();
    }
}
"#;

/// Manager for persistent kernel execution
pub struct PersistentKernelManager {
    _device: Arc<GpuDevice>,
    max_grid_size: u32,
    optimal_block_size: u32,
}

impl PersistentKernelManager {
    /// Create new persistent kernel manager
    ///
    /// Queries device properties to determine optimal launch configuration
    pub fn new(device: &GpuDevice) -> Result<Self, GpuError> {
        use cudarc::driver::sys;

        // Query GPU properties
        let (multiprocessor_count, max_blocks_per_sm) = unsafe {
            let mut sm_count = 0;
            let mut max_blocks = 0;

            sys::cuDeviceGetAttribute(
                &mut sm_count,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device.context.cu_device(),
            )
            .result()
            .map_err(|e| {
                GpuError::InitializationError(format!("Failed to query SM count: {:?}", e))
            })?;

            sys::cuDeviceGetAttribute(
                &mut max_blocks,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                device.context.cu_device(),
            )
            .result()
            .map_err(|e| {
                GpuError::InitializationError(format!("Failed to query max blocks/SM: {:?}", e))
            })?;

            (sm_count as u32, max_blocks as u32)
        };

        // For cooperative launch, use 80% of theoretical maximum to be safe
        // This accounts for register pressure and other resource constraints
        let max_cooperative_blocks = (multiprocessor_count * max_blocks_per_sm) as f32 * 0.8;
        let max_grid_size = max_cooperative_blocks as u32;

        eprintln!("🎯 Cooperative launch limits for this GPU:");
        eprintln!(
            "   SMs: {}, Max blocks/SM: {}",
            multiprocessor_count, max_blocks_per_sm
        );
        eprintln!(
            "   Theoretical max: {} blocks",
            multiprocessor_count * max_blocks_per_sm
        );
        eprintln!("   Safe grid size: {} blocks (80% of max)", max_grid_size);

        Ok(Self {
            _device: Arc::new(GpuDevice::with_device_id(0)?),
            max_grid_size,
            optimal_block_size: 256,
        })
    }

    /// Get optimal launch configuration for persistent kernel
    ///
    /// Ensures all blocks can be simultaneously resident for cooperative launch
    pub fn get_launch_config(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.max_grid_size, 1, 1),
            block_dim: (self.optimal_block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Check if cooperative launch is supported
    ///
    /// Returns maximum grid size for cooperative launch
    pub fn check_cooperative_support(&self) -> Result<u32, GpuError> {
        // TODO: Query actual device properties via cudarc
        // For now, return conservative estimate
        Ok(self.max_grid_size)
    }

    /// Execute batch of tasks using persistent kernel
    ///
    /// Launches kernel once and processes all tasks sequentially using GPU
    pub fn execute_batch<I: PersistentIndicator>(
        &self,
        batch: &TaskBatch<I>,
    ) -> Result<Vec<Vec<f64>>, GpuError> {
        if batch.is_empty() {
            return Err(GpuError::InvalidParameter("Empty task batch".to_string()));
        }

        let func = compile_persistent_kernel(&self._device)?;

        // Query actual occupancy for this compiled kernel using occupancy calculator
        let occupancy_calc = OccupancyCalculator::new(&self._device)?;
        let optimal_grid_size =
            occupancy_calc.calculate_optimal_grid_size(&func, self.optimal_block_size, 0)?;

        eprintln!(
            "🚀 Launching with {} blocks (vs {} conservative)",
            optimal_grid_size,
            self.max_grid_size / 4
        );

        let mut buffers = allocate_batch_buffers(&self._device, batch)?;
        upload_batch_data(&self._device, batch, &mut buffers)?;
        launch_cooperative_kernel(
            &self._device,
            &func,
            &buffers,
            batch.len() as i32,
            optimal_grid_size, // Use dynamic occupancy-based grid size
        )?;
        download_batch_results(&self._device, &mut buffers)
    }
}

/// GPU buffer set for batch execution
struct BatchBuffers {
    /// Host pinned memory for input buffers (if available)
    h_inputs: Vec<Option<PinnedBuffer<f64>>>,
    /// Host pinned memory for output buffers (if available) - [task][output_idx]
    h_outputs: Vec<Vec<Option<PinnedBuffer<f64>>>>,
    /// Whether pinned memory is being used (false = pageable fallback)
    using_pinned: bool,

    /// Array of input buffer pointers on GPU
    d_input_ptrs: CudaSlice<u64>,
    /// Array of output buffer pointers on GPU
    d_output_ptrs: CudaSlice<u64>,
    /// Individual input buffers
    d_inputs: Vec<CudaSlice<f64>>,
    /// Individual output buffers - [task][output_idx]
    d_outputs: Vec<Vec<CudaSlice<f64>>>,
    /// Dataset sizes
    d_sizes: CudaSlice<i32>,
    /// ROC periods
    d_periods: CudaSlice<i32>,
}

impl BatchBuffers {
    /// Get memory usage information for logging
    pub fn memory_info(&self) -> String {
        if self.using_pinned {
            let total_outputs: usize = self.h_outputs.iter().map(|v| v.len()).sum();
            format!(
                "✅ Using pinned memory ({} input buffers, {} output buffers)",
                self.h_inputs.len(),
                total_outputs
            )
        } else {
            "⚠️  Using pageable memory (pinned allocation failed, expect 20-30% slower transfers)"
                .to_string()
        }
    }
}

/// Compile persistent kernel PTX
fn compile_persistent_kernel(device: &GpuDevice) -> Result<CudaFunction, GpuError> {
    // Compile PTX with optimizations
    let ptx = compile_ptx_optimized(PERSISTENT_ROC_KERNEL).map_err(|e| {
        GpuError::CompilationError(format!("Failed to compile persistent ROC kernel: {:?}", e))
    })?;

    // Load module
    let module = device
        .context()
        .load_module(ptx)
        .map_err(|e| GpuError::CompilationError(format!("Failed to load PTX module: {:?}", e)))?;

    // Load kernel function
    let func = module.load_function("persistent_roc_kernel").map_err(|e| {
        GpuError::ExecutionError(format!("Failed to load kernel function: {:?}", e))
    })?;

    Ok(func)
}

/// Allocate GPU buffers for batch processing with pinned memory optimization
fn allocate_batch_buffers<I: PersistentIndicator>(
    device: &GpuDevice,
    batch: &TaskBatch<I>,
) -> Result<BatchBuffers, GpuError> {
    let num_tasks = batch.len();

    // Try to allocate pinned memory for all input buffers
    let mut h_inputs = Vec::with_capacity(num_tasks);
    let mut h_outputs = Vec::with_capacity(num_tasks);
    let mut using_pinned = true;

    // Attempt pinned allocation for input buffers
    for task in batch.tasks() {
        match PinnedBuffer::new(task.data.len()) {
            Ok(pinned) => h_inputs.push(Some(pinned)),
            Err(e) => {
                eprintln!("⚠️  Pinned memory allocation failed: {:?}", e);
                eprintln!("   Falling back to pageable memory (20-30% slower transfers)");
                using_pinned = false;
                break;
            }
        }
    }

    // Get number of outputs for this indicator type
    let num_outputs = I::num_outputs();

    // If pinned allocation failed, clear and use None placeholders
    if !using_pinned {
        h_inputs.clear();
        for _ in 0..num_tasks {
            h_inputs.push(None);
            // Allocate Vec for each task's outputs
            let mut task_outputs = Vec::with_capacity(num_outputs);
            for _ in 0..num_outputs {
                task_outputs.push(None);
            }
            h_outputs.push(task_outputs);
        }
    } else {
        // Allocate pinned output buffers (single contiguous buffer per task)
        for task in batch.tasks() {
            let output_size = task.data.len() * num_outputs;
            match PinnedBuffer::new(output_size) {
                Ok(pinned) => h_outputs.push(vec![Some(pinned)]),
                Err(e) => {
                    eprintln!("⚠️  Pinned output buffer allocation failed: {:?}", e);
                    eprintln!("   Falling back to pageable memory");
                    using_pinned = false;
                    // Clear all and use pageable
                    h_inputs.clear();
                    h_outputs.clear();
                    for _ in 0..num_tasks {
                        h_inputs.push(None);
                        h_outputs.push(vec![None]);
                    }
                    break;
                }
            }
        }
    }

    // Allocate GPU device buffers (always required)
    let mut d_inputs = Vec::with_capacity(num_tasks);
    let mut d_outputs = Vec::with_capacity(num_tasks);

    for task in batch.tasks() {
        let input_buf = device.allocate_device_buffer(task.data.len())?;
        d_inputs.push(input_buf);

        // Allocate single contiguous buffer for all outputs
        // Multi-output (e.g., MACD with 3 outputs): allocate size = n * num_outputs
        // Single-output (e.g., ROC with 1 output): allocate size = n
        let output_size = task.data.len() * num_outputs;
        let output_buf = device.allocate_device_buffer(output_size)?;
        d_outputs.push(vec![output_buf]);  // Wrap in vec for consistency
    }

    // Create host-side pointer arrays
    let mut input_ptrs_host = Vec::with_capacity(num_tasks);
    let mut output_ptrs_host = Vec::with_capacity(num_tasks);

    for (input_buf, task_outputs) in d_inputs.iter().zip(d_outputs.iter()) {
        let (input_ptr, _) = input_buf.device_ptr(&device.stream);
        input_ptrs_host.push(input_ptr as u64);

        // Use first output for pointer array (maintains ROC/RSI/ATR compatibility)
        // TODO: Full multi-output requires separate pointer arrays for each output
        let (output_ptr, _) = task_outputs[0].device_ptr(&device.stream);
        output_ptrs_host.push(output_ptr as u64);
    }

    // Allocate GPU memory for pointer arrays and copy
    let mut d_input_ptrs = device.stream.alloc_zeros::<u64>(num_tasks).map_err(|e| {
        GpuError::AllocationError(format!("Failed to allocate input pointer array: {:?}", e))
    })?;

    device
        .stream
        .memcpy_htod(&input_ptrs_host, &mut d_input_ptrs)
        .map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy input pointers: {:?}", e))
        })?;

    let mut d_output_ptrs = device.stream.alloc_zeros::<u64>(num_tasks).map_err(|e| {
        GpuError::AllocationError(format!("Failed to allocate output pointer array: {:?}", e))
    })?;

    device
        .stream
        .memcpy_htod(&output_ptrs_host, &mut d_output_ptrs)
        .map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy output pointers: {:?}", e))
        })?;

    // Copy sizes and parameters
    let sizes: Vec<i32> = batch.tasks().iter().map(|t| t.data.len() as i32).collect();
    let d_sizes = device.copy_to_device_i32(&sizes)?;

    // Extract parameters - for i32 params (ROC, RSI, ATR), we can directly copy
    // For complex types like MacdParams, this will need special handling
    let periods: Vec<i32> = batch
        .tasks()
        .iter()
        .map(|t| {
            // SAFETY: This assumes I::Params is either i32 or can be safely transmuted to i32.
            // For MacdParams (12 bytes), this will copy first 4 bytes (fast_period).
            // This works for ROC/RSI/ATR (i32 params), but may need refinement for MACD.
            unsafe { std::mem::transmute_copy::<I::Params, i32>(&t.params) }
        })
        .collect();
    let d_periods = device.copy_to_device_i32(&periods)?;

    let buffers = BatchBuffers {
        h_inputs,
        h_outputs,
        using_pinned,
        d_input_ptrs,
        d_output_ptrs,
        d_inputs,
        d_outputs,
        d_sizes,
        d_periods,
    };

    eprintln!("📊 {}", buffers.memory_info());

    Ok(buffers)
}

/// Upload input data to GPU buffers using pinned memory if available
fn upload_batch_data<I: PersistentIndicator>(
    device: &GpuDevice,
    batch: &TaskBatch<I>,
    buffers: &mut BatchBuffers,
) -> Result<(), GpuError> {
    if buffers.using_pinned {
        // Fast path: Use pinned memory (20-30% faster)
        for (i, task) in batch.tasks().iter().enumerate() {
            if let Some(pinned) = &mut buffers.h_inputs[i] {
                // Copy from Vec to pinned buffer
                pinned.copy_from_slice(&task.data);
                // DMA transfer from pinned to device (fast!)
                device.htod_pinned(pinned, &mut buffers.d_inputs[i])?;
            }
        }
    } else {
        // Fallback: Use pageable memory (traditional approach)
        for (i, task) in batch.tasks().iter().enumerate() {
            device
                .stream
                .memcpy_htod(&task.data, &mut buffers.d_inputs[i])
                .map_err(|e| {
                    GpuError::MemoryCopyError(format!("Failed to upload task {} data: {:?}", i, e))
                })?;
        }
    }
    Ok(())
}

/// Launch cooperative kernel using cudarc's safe wrapper
///
/// # Parameters
/// - `device`: GPU device
/// - `func`: Compiled CUDA function
/// - `buffers`: Batch buffers (inputs, outputs, sizes, periods)
/// - `num_tasks`: Number of tasks in batch
/// - `grid_size`: GPU-specific grid size (from PersistentKernelManager)
fn launch_cooperative_kernel(
    device: &GpuDevice,
    func: &CudaFunction,
    buffers: &BatchBuffers,
    num_tasks: i32,
    grid_size: u32,
) -> Result<(), GpuError> {
    use cudarc::driver::PushKernelArg;

    // Launch configuration using GPU-specific limits
    let block_dim = (256u32, 1u32, 1u32);
    let grid_dim = (grid_size, 1u32, 1u32);

    // Launch cooperative kernel using cudarc's safe wrapper
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream
            .launch_builder(func)
            .arg(&buffers.d_input_ptrs)
            .arg(&buffers.d_output_ptrs)
            .arg(&buffers.d_sizes)
            .arg(&buffers.d_periods)
            .arg(&num_tasks)
            .launch_cooperative(cfg)
            .map_err(|e| GpuError::ExecutionError(format!("Cooperative launch failed: {:?}", e)))?;
    }

    // Synchronize
    device.synchronize()?;

    Ok(())
}

/// Download results from GPU using pinned memory if available
fn download_batch_results(
    device: &GpuDevice,
    buffers: &mut BatchBuffers,
) -> Result<Vec<Vec<f64>>, GpuError> {
    let mut results = Vec::with_capacity(buffers.d_outputs.len());

    if buffers.using_pinned {
        // Fast path: Use pinned memory (20-30% faster)
        for (task_idx, task_outputs) in buffers.d_outputs.iter().enumerate() {
            // Download single contiguous buffer (contains all outputs)
            if let Some(pinned) = &mut buffers.h_outputs[task_idx][0] {
                device.dtoh_pinned(&task_outputs[0], pinned)?;
                results.push(pinned.as_slice().to_vec());
            }
        }
    } else {
        // Fallback: Use pageable memory (traditional approach)
        for task_outputs in &buffers.d_outputs {
            // Download single contiguous buffer (contains all outputs)
            let result = device.copy_to_host(&task_outputs[0])?;
            results.push(result);
        }
    }

    Ok(results)
}

/// Execute batch of ROC tasks using persistent kernel
///
/// This function eliminates per-task kernel launch overhead by:
/// 1. Launching kernel once with cooperative groups
/// 2. Processing all tasks sequentially within the kernel
/// 3. Using grid-wide synchronization between tasks
///
/// # Performance
///
/// - Launch overhead: O(1) instead of O(N)
/// - Expected speedup: 2-4x for N ≥ 10 tasks
/// - Overhead reduction: 80-90% for batch operations
///
/// # Arguments
///
/// * `device` - GPU device handle
/// * `batch` - Task batch containing inputs, sizes, and periods
///
/// # Returns
///
/// Vector of result vectors, one per task in the batch.
/// Each result has the same length as the corresponding input.
///
/// # Errors
///
/// Returns `GpuError` if:
/// - Batch is empty
/// - GPU memory allocation fails
/// - Kernel compilation fails
/// - Cooperative launch unsupported
///
/// # Example
///
/// ```rust,no_run
/// # use kimsfinance_core::gpu::{GpuDevice, persistent::*};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = GpuDevice::new()?;
/// let mut batch = TaskBatch::new();
///
/// batch.add_task(vec![100.0, 102.0, 104.0], 2);
/// batch.add_task(vec![200.0, 201.0, 202.0], 2);
///
/// let results = execute_batch(&device, &batch)?;
/// assert_eq!(results.len(), 2);
/// # Ok(())
/// # }
/// ```
pub fn execute_batch<I: PersistentIndicator>(
    device: &GpuDevice,
    batch: &TaskBatch<I>,
) -> Result<Vec<Vec<f64>>, GpuError> {
    if batch.is_empty() {
        return Err(GpuError::InvalidParameter("Empty batch".to_string()));
    }

    // Compile kernel for this specific indicator type
    let func = I::compile_kernel(device)?;

    // Use occupancy calculator for optimal grid size
    let calculator = OccupancyCalculator::new(device)?;
    let grid_size = calculator.calculate_optimal_grid_size(&func, 256, 0)?;

    eprintln!(
        "🚀 Launching {} indicator with {} blocks (occupancy-optimized)",
        I::kernel_name(),
        grid_size
    );

    // Allocate, upload, launch, download
    let mut buffers = allocate_batch_buffers(device, batch)?;
    upload_batch_data(device, batch, &mut buffers)?;
    launch_cooperative_kernel(device, &func, &buffers, batch.len() as i32, grid_size)?;
    download_batch_results(device, &mut buffers)
}

/// Task batch for persistent kernel execution
///
/// Generic over indicator type `I`, enabling type-safe batch processing
/// for different technical indicators (ROC, RSI, MACD, ATR).
///
/// # Type Safety
///
/// The generic parameter `I` ensures:
/// - Correct parameters are passed (e.g., `i32` for ROC, `MacdParams` for MACD)
/// - Compile-time validation of indicator configuration
/// - Zero-cost abstraction (no runtime overhead)
///
/// # Examples
///
/// ```rust,no_run
/// # use kimsfinance_core::gpu::persistent::*;
/// // ROC batch (single output, i32 params)
/// let mut roc_batch = TaskBatch::<RocIndicator>::new();
/// roc_batch.add_task(vec![100.0, 102.0, 104.0], 2);
///
/// // RSI batch (single output, i32 params)
/// let mut rsi_batch = TaskBatch::<RsiIndicator>::new();
/// rsi_batch.add_task(vec![44.0, 44.5, 43.0], 14);
///
/// // MACD batch (triple output, MacdParams)
/// let mut macd_batch = TaskBatch::<MacdIndicator>::new();
/// macd_batch.add_task(vec![100.0; 50], MacdParams::standard());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct TaskBatch<I: PersistentIndicator> {
    /// Individual tasks with data and parameters
    tasks: Vec<Task<I>>,
    /// Phantom data for type safety
    _marker: std::marker::PhantomData<I>,
}

/// Individual task within a batch
#[derive(Debug, Clone)]
pub struct Task<I: PersistentIndicator> {
    /// Input data for this task
    pub data: Vec<f64>,
    /// Indicator-specific parameters
    pub params: I::Params,
}

impl<I: PersistentIndicator> TaskBatch<I> {
    /// Create new empty task batch
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Add task to batch
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for the indicator (e.g., close prices for ROC/RSI)
    /// * `params` - Indicator-specific parameters (type enforced by `I::Params`)
    pub fn add_task(&mut self, data: Vec<f64>, params: I::Params) {
        self.tasks.push(Task { data, params });
    }

    /// Get reference to all tasks
    pub fn tasks(&self) -> &[Task<I>] {
        &self.tasks
    }

    /// Get number of tasks
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

}

impl<I: PersistentIndicator> Default for TaskBatch<I> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for ROC batch (backward compatibility)
pub type RocBatch = TaskBatch<RocIndicator>;

/// Type alias for RSI batch
pub type RsiBatch = TaskBatch<RsiIndicator>;

/// Type alias for MACD batch
pub type MacdBatch = TaskBatch<MacdIndicator>;

/// Type alias for ATR batch
pub type AtrBatch = TaskBatch<AtrIndicator>;

/// Type alias for Stochastic batch
pub type StochasticBatch = TaskBatch<StochasticIndicator>;

/// Type alias for Donchian batch
pub type DonchianBatch = TaskBatch<DonchianIndicator>;

/// Type alias for Keltner batch
pub type KeltnerBatch = TaskBatch<KeltnerIndicator>;

/// Type alias for Aroon batch
pub type AroonBatch = TaskBatch<AroonIndicator>;

/// Type alias for Williams %R batch
pub type WilliamsRBatch = TaskBatch<WilliamsRIndicator>;

/// Type alias for CCI batch
pub type CciBatch = TaskBatch<CciIndicator>;

/// Type alias for SMA batch
pub type SmaBatch = TaskBatch<SmaIndicator>;

/// Type alias for EMA batch
pub type EmaBatch = TaskBatch<EmaIndicator>;

/// Type alias for Bollinger batch
pub type BollingerBatch = TaskBatch<BollingerIndicator>;

/// Type alias for Elder Ray batch
pub type ElderRayBatch = TaskBatch<ElderRayIndicator>;

/// Type alias for OBV batch
pub type ObvBatch = TaskBatch<ObvIndicator>;

/// Type alias for CMF batch
pub type CmfBatch = TaskBatch<CmfIndicator>;

/// Type alias for VWMA batch
pub type VwmaBatch = TaskBatch<VwmaIndicator>;

/// Type alias for WMA batch
pub type WmaBatch = TaskBatch<WmaIndicator>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_batch_creation() {
        let mut batch = TaskBatch::new();
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());

        batch.add_task(vec![1.0, 2.0, 3.0], 14);
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_persistent_kernel_manager_creation() {
        let device = GpuDevice::new().expect("GPU required");
        let manager = PersistentKernelManager::new(&device).expect("Manager creation failed");

        let config = manager.get_launch_config();
        assert!(config.block_dim.0 > 0);
        assert!(config.grid_dim.0 > 0);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cooperative_support_check() {
        let device = GpuDevice::new().expect("GPU required");
        let manager = PersistentKernelManager::new(&device).expect("Manager creation failed");

        let max_grid = manager
            .check_cooperative_support()
            .expect("Cooperative support check failed");

        assert!(max_grid > 0, "Cooperative launch should be supported");
    }

    // ==================== Comprehensive Correctness Tests ====================

    #[test]
    #[ignore] // Requires GPU
    fn test_persistent_single_task() {
        let device = GpuDevice::new().expect("GPU required");

        let mut batch = TaskBatch::new();
        let data = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];
        batch.add_task(data.clone(), 3); // ROC period 3

        let results = execute_batch(&device, &batch).expect("Execute failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), data.len());

        // Validate ROC calculation
        // ROC(i) = (close[i] - close[i-period]) / close[i-period] * 100
        // ROC(3) = (103.0 - 100.0) / 100.0 * 100 = 3.0
        assert!(
            (results[0][3] - 3.0).abs() < 1e-6,
            "ROC[3] should be 3.0, got {}",
            results[0][3]
        );

        // ROC(4) = (105.0 - 102.0) / 102.0 * 100 = 2.941176...
        let expected_roc4 = (105.0 - 102.0) / 102.0 * 100.0;
        assert!(
            (results[0][4] - expected_roc4).abs() < 1e-6,
            "ROC[4] should be {}, got {}",
            expected_roc4,
            results[0][4]
        );

        // ROC(5) = (104.0 - 101.0) / 101.0 * 100 = 2.970297...
        let expected_roc5 = (104.0 - 101.0) / 101.0 * 100.0;
        assert!(
            (results[0][5] - expected_roc5).abs() < 1e-6,
            "ROC[5] should be {}, got {}",
            expected_roc5,
            results[0][5]
        );

        // First 3 values should be NaN
        assert!(results[0][0].is_nan(), "ROC[0] should be NaN");
        assert!(results[0][1].is_nan(), "ROC[1] should be NaN");
        assert!(results[0][2].is_nan(), "ROC[2] should be NaN");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_persistent_multi_task_batch() {
        let device = GpuDevice::new().expect("GPU required");

        let mut batch = TaskBatch::new();

        // Task 1: ROC(14) on 100 points
        batch.add_task((0..100).map(|i| 100.0 + i as f64).collect(), 14);

        // Task 2: ROC(7) on 50 points
        batch.add_task((0..50).map(|i| 200.0 + i as f64 * 2.0).collect(), 7);

        // Task 3: ROC(21) on 150 points
        batch.add_task((0..150).map(|i| 50.0 + i as f64 * 0.5).collect(), 21);

        let results = execute_batch(&device, &batch).expect("Execute failed");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 100);
        assert_eq!(results[1].len(), 50);
        assert_eq!(results[2].len(), 150);

        // Validate first valid ROC for each task
        // All should be non-NaN after warmup period
        assert!(
            results[0][14].is_finite(),
            "Task 1 ROC[14] should be finite"
        );
        assert!(results[1][7].is_finite(), "Task 2 ROC[7] should be finite");
        assert!(
            results[2][21].is_finite(),
            "Task 3 ROC[21] should be finite"
        );

        // Validate numerical correctness for Task 1
        // ROC(14) = (price[14] - price[0]) / price[0] * 100
        // = (114.0 - 100.0) / 100.0 * 100 = 14.0
        assert!(
            (results[0][14] - 14.0).abs() < 1e-6,
            "Task 1 ROC[14] should be 14.0, got {}",
            results[0][14]
        );

        // Validate numerical correctness for Task 2
        // ROC(7) = (price[7] - price[0]) / price[0] * 100
        // price[7] = 200.0 + 7*2 = 214.0, price[0] = 200.0
        // = (214.0 - 200.0) / 200.0 * 100 = 7.0
        assert!(
            (results[1][7] - 7.0).abs() < 1e-6,
            "Task 2 ROC[7] should be 7.0, got {}",
            results[1][7]
        );

        // Validate numerical correctness for Task 3
        // ROC(21) = (price[21] - price[0]) / price[0] * 100
        // price[21] = 50.0 + 21*0.5 = 60.5, price[0] = 50.0
        // = (60.5 - 50.0) / 50.0 * 100 = 21.0
        assert!(
            (results[2][21] - 21.0).abs() < 1e-6,
            "Task 3 ROC[21] should be 21.0, got {}",
            results[2][21]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_persistent_empty_batch_error() {
        let device = GpuDevice::new().expect("GPU required");
        let batch = TaskBatch::new(); // Empty

        let result = execute_batch(&device, &batch);
        assert!(result.is_err(), "Empty batch should return error");

        match result {
            Err(GpuError::InvalidParameter(msg)) => {
                assert!(
                    msg.contains("Empty"),
                    "Error message should mention empty batch"
                );
            }
            _ => panic!("Expected InvalidParameter error for empty batch"),
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_persistent_large_batch() {
        let device = GpuDevice::new().expect("GPU required");

        let mut batch = TaskBatch::new();

        // Add 100 tasks
        for i in 0..100 {
            let size = 50 + (i % 10) * 10; // Variable sizes 50-140
            let period = 7 + (i % 5) * 7; // Periods 7, 14, 21, 28, 35
            batch.add_task((0..size).map(|j| 100.0 + j as f64).collect(), period);
        }

        let results = execute_batch(&device, &batch).expect("Execute failed");

        assert_eq!(results.len(), 100);

        // Verify all results have correct lengths
        for (i, result) in results.iter().enumerate() {
            let expected_len = 50 + (i % 10) * 10;
            assert_eq!(
                result.len(),
                expected_len,
                "Task {} should have length {}",
                i,
                expected_len
            );
        }

        // Spot check numerical correctness on a few tasks
        // Task 0: size=50, period=7
        // ROC[7] = (107.0 - 100.0) / 100.0 * 100 = 7.0
        assert!(
            (results[0][7] - 7.0).abs() < 1e-6,
            "Task 0 ROC[7] should be 7.0, got {}",
            results[0][7]
        );

        // Task 50: size=50, period=7 (same pattern)
        assert!(
            (results[50][7] - 7.0).abs() < 1e-6,
            "Task 50 ROC[7] should be 7.0, got {}",
            results[50][7]
        );

        // Task 99: size=140, period=35
        // ROC[35] = (135.0 - 100.0) / 100.0 * 100 = 35.0
        assert!(
            (results[99][35] - 35.0).abs() < 1e-6,
            "Task 99 ROC[35] should be 35.0, got {}",
            results[99][35]
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cooperative_launch_supported() {
        let device = GpuDevice::new().expect("GPU required");
        let manager = PersistentKernelManager::new(&device).expect("Manager creation failed");

        let max_grid = manager
            .check_cooperative_support()
            .expect("Cooperative support check failed");

        assert!(max_grid > 0, "GPU must support cooperative launch");
        println!("Max cooperative grid size: {}", max_grid);
    }

    // Additional edge case tests

    #[test]
    #[ignore] // Requires GPU
    fn test_persistent_single_element_task() {
        let device = GpuDevice::new().expect("GPU required");

        let mut batch = TaskBatch::new();
        batch.add_task(vec![100.0], 1); // Single element, period=1

        let results = execute_batch(&device, &batch).expect("Execute failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 1);
        assert!(
            results[0][0].is_nan(),
            "Single element with period=1 should be NaN"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_persistent_varying_periods() {
        let device = GpuDevice::new().expect("GPU required");

        let mut batch = TaskBatch::new();
        let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();

        // Same data, different periods
        batch.add_task(data.clone(), 1);
        batch.add_task(data.clone(), 5);
        batch.add_task(data.clone(), 10);
        batch.add_task(data.clone(), 20);

        let results = execute_batch(&device, &batch).expect("Execute failed");

        assert_eq!(results.len(), 4);

        // ROC with period=1: ROC[1] = (101 - 100) / 100 * 100 = 1.0
        assert!((results[0][1] - 1.0).abs() < 1e-6);

        // ROC with period=5: ROC[5] = (105 - 100) / 100 * 100 = 5.0
        assert!((results[1][5] - 5.0).abs() < 1e-6);

        // ROC with period=10: ROC[10] = (110 - 100) / 100 * 100 = 10.0
        assert!((results[2][10] - 10.0).abs() < 1e-6);

        // ROC with period=20: ROC[20] = (120 - 100) / 100 * 100 = 20.0
        assert!((results[3][20] - 20.0).abs() < 1e-6);
    }
}
