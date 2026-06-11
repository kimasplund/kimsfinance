//! Generic multi-input/multi-output batch execution for persistent kernels
//!
//! This module provides a type-safe, generic interface for executing persistent kernels
//! with arbitrary numbers of inputs and outputs.
//!
//! # Status: execution not implemented
//!
//! The batch/task containers below work, but [`execute_generic_batch`]
//! currently returns an error because the generic kernel launch was never
//! implemented (see its documentation). The examples in this header describe
//! the intended API once a launch path exists.
//!
//! # Key Features
//!
//! - **Type-safe**: Generic over `PersistentIndicator` trait
//! - **Multi-input**: Supports 1-N input arrays (e.g., ATR with high/low/close)
//! - **Multi-output**: Supports 1-N output arrays (e.g., MACD with 3 outputs)
//! - **Backward compatible**: Single-input/single-output works seamlessly
//!
//! # Example: Single-Input/Single-Output (ROC)
//!
//! ```rust,no_run
//! use kimsfinance_core::gpu::{GpuDevice, persistent::*};
//!
//! let device = GpuDevice::new()?;
//! let mut batch = GenericBatch::<RocIndicator>::new();
//!
//! batch.add_task(vec![vec![100.0, 102.0, 104.0]], 3); // Single input array
//!
//! let results = execute_generic_batch(&device, &batch)?;
//! // results[0] = [output_array]
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Example: Multi-Input/Single-Output (ATR)
//!
//! ```rust,no_run
//! use kimsfinance_core::gpu::{GpuDevice, persistent::*};
//!
//! let device = GpuDevice::new()?;
//! let mut batch = GenericBatch::<AtrIndicator>::new();
//!
//! let high = vec![10.0, 11.0, 12.0];
//! let low = vec![9.0, 10.0, 10.5];
//! let close = vec![9.5, 10.5, 11.5];
//!
//! batch.add_task(vec![high, low, close], 14); // Three input arrays
//!
//! let results = execute_generic_batch(&device, &batch)?;
//! // results[0] = [atr_array]
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Example: Single-Input/Multi-Output (MACD)
//!
//! ```rust,no_run
//! use kimsfinance_core::gpu::{GpuDevice, persistent::*};
//!
//! let device = GpuDevice::new()?;
//! let mut batch = GenericBatch::<MacdIndicator>::new();
//!
//! batch.add_task(vec![vec![44.0, 44.5, 43.0]], MacdParams::standard());
//!
//! let results = execute_generic_batch(&device, &batch)?;
//! // results[0] = [macd_line, signal_line, histogram]
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::PinnedBuffer;
use super::traits::PersistentIndicator;
use crate::gpu::device::{GpuDevice, GpuError};
use cudarc::driver::{CudaSlice, DevicePtr};
use std::marker::PhantomData;

/// Generic task for persistent kernel execution
///
/// Type-safe wrapper around multi-dimensional input/output arrays
pub struct Task<I: PersistentIndicator>
where
    I::Params: std::fmt::Debug,
{
    /// Input arrays (length = I::num_inputs())
    pub inputs: Vec<Vec<f64>>,
    /// Parameters for this task
    pub params: I::Params,
}

impl<I: PersistentIndicator> Task<I> {
    /// Create task with multiple input arrays
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len()` != `I::num_inputs()`
    pub fn new(inputs: Vec<Vec<f64>>, params: I::Params) -> Self {
        assert_eq!(
            inputs.len(),
            I::num_inputs(),
            "Expected {} inputs, got {}",
            I::num_inputs(),
            inputs.len()
        );
        Self { inputs, params }
    }

    /// Create task with single input array (convenience method)
    ///
    /// # Panics
    ///
    /// Panics if `I::num_inputs()` != 1
    pub fn new_single(data: Vec<f64>, params: I::Params) -> Self {
        assert_eq!(
            I::num_inputs(),
            1,
            "Cannot use new_single for multi-input indicator"
        );
        Self::new(vec![data], params)
    }

    /// Get size of first input (assumes all inputs same length)
    pub fn size(&self) -> usize {
        self.inputs[0].len()
    }
}

/// Generic batch of tasks for persistent kernel execution
pub struct GenericBatch<I: PersistentIndicator> {
    /// Tasks to execute
    tasks: Vec<Task<I>>,
    /// Phantom data for indicator type
    _phantom: PhantomData<I>,
}

impl<I: PersistentIndicator> GenericBatch<I> {
    /// Create new empty batch
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add task to batch
    pub fn add_task(&mut self, inputs: Vec<Vec<f64>>, params: I::Params) {
        self.tasks.push(Task::new(inputs, params));
    }

    /// Add single-input task (convenience method)
    pub fn add_single_input_task(&mut self, data: Vec<f64>, params: I::Params) {
        self.tasks.push(Task::new_single(data, params));
    }

    /// Get number of tasks
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Get reference to tasks
    pub fn tasks(&self) -> &[Task<I>] {
        &self.tasks
    }
}

impl<I: PersistentIndicator> Default for GenericBatch<I> {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU buffer set for multi-input/multi-output batch execution
///
/// Retained for the future generic launch implementation; currently unused
/// because `execute_generic_batch` returns an error before allocating
/// (see that function's documentation).
#[allow(dead_code)]
struct GenericBatchBuffers<I: PersistentIndicator> {
    /// Multi-input support: [task_idx][input_idx] -> buffer
    d_inputs: Vec<Vec<CudaSlice<f64>>>,
    /// Host-side pinned buffers for inputs (optional, for performance)
    h_inputs: Vec<Vec<Option<PinnedBuffer<f64>>>>,

    /// Multi-output support: [task_idx][output_idx] -> buffer
    d_outputs: Vec<Vec<CudaSlice<f64>>>,
    /// Host-side pinned buffers for outputs (optional, for performance)
    h_outputs: Vec<Vec<Option<PinnedBuffer<f64>>>>,

    /// Pointer arrays on GPU: [input_idx] -> array of pointers for that input dimension
    d_input_ptr_arrays: Vec<CudaSlice<u64>>,
    /// Pointer arrays on GPU: [output_idx] -> array of pointers for that output dimension
    d_output_ptr_arrays: Vec<CudaSlice<u64>>,

    /// Dataset sizes
    d_sizes: CudaSlice<i32>,
    /// Parameter buffer (generic byte array)
    d_params: Vec<u8>, // Will be copied to GPU as raw bytes

    /// Whether pinned memory is being used
    using_pinned: bool,

    /// Phantom data
    _phantom: PhantomData<I>,
}

/// Allocate GPU buffers for generic multi-input/multi-output batch
///
/// Retained for the future generic launch implementation (currently unused).
#[allow(dead_code)]
fn allocate_generic_buffers<I: PersistentIndicator>(
    device: &GpuDevice,
    batch: &GenericBatch<I>,
) -> Result<GenericBatchBuffers<I>, GpuError> {
    let num_tasks = batch.len();
    let num_inputs = I::num_inputs();
    let num_outputs = I::num_outputs();

    // Allocate input buffers: [task][input_idx]
    let mut d_inputs = Vec::with_capacity(num_tasks);
    let mut h_inputs = Vec::with_capacity(num_tasks);

    for task in batch.tasks() {
        let mut task_d_inputs = Vec::with_capacity(num_inputs);
        let mut task_h_inputs = Vec::with_capacity(num_inputs);

        for input_data in &task.inputs {
            // Try pinned allocation for performance
            let h_buf = PinnedBuffer::new(input_data.len()).ok();
            task_h_inputs.push(h_buf);

            // Allocate GPU buffer
            task_d_inputs.push(device.alloc_buffer(input_data.len())?);
        }

        d_inputs.push(task_d_inputs);
        h_inputs.push(task_h_inputs);
    }

    // Allocate output buffers: [task][output_idx]
    let mut d_outputs = Vec::with_capacity(num_tasks);
    let mut h_outputs = Vec::with_capacity(num_tasks);

    for task in batch.tasks() {
        let mut task_d_outputs = Vec::with_capacity(num_outputs);
        let mut task_h_outputs = Vec::with_capacity(num_outputs);

        for _ in 0..num_outputs {
            let output_size = task.size(); // Assuming same size as input
            let h_buf = PinnedBuffer::new(output_size).ok();
            task_h_outputs.push(h_buf);
            task_d_outputs.push(device.alloc_buffer(output_size)?);
        }

        d_outputs.push(task_d_outputs);
        h_outputs.push(task_h_outputs);
    }

    // Create pointer arrays for each input dimension
    let mut d_input_ptr_arrays = Vec::with_capacity(num_inputs);
    for input_idx in 0..num_inputs {
        let mut ptrs_host = Vec::with_capacity(num_tasks);
        for task_inputs in &d_inputs {
            let (ptr, _) = task_inputs[input_idx].device_ptr(&device.stream);
            ptrs_host.push(ptr as u64);
        }

        // Copy to GPU
        let mut d_ptrs = device.stream.alloc_zeros::<u64>(num_tasks).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate input pointer array: {:?}", e))
        })?;

        device
            .stream
            .memcpy_htod(&ptrs_host, &mut d_ptrs)
            .map_err(|e| {
                GpuError::MemoryCopyError(format!("Failed to copy input pointers: {:?}", e))
            })?;

        d_input_ptr_arrays.push(d_ptrs);
    }

    // Create pointer arrays for each output dimension
    let mut d_output_ptr_arrays = Vec::with_capacity(num_outputs);
    for output_idx in 0..num_outputs {
        let mut ptrs_host = Vec::with_capacity(num_tasks);
        for task_outputs in &d_outputs {
            let (ptr, _) = task_outputs[output_idx].device_ptr(&device.stream);
            ptrs_host.push(ptr as u64);
        }

        // Copy to GPU
        let mut d_ptrs = device.stream.alloc_zeros::<u64>(num_tasks).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate output pointer array: {:?}", e))
        })?;

        device
            .stream
            .memcpy_htod(&ptrs_host, &mut d_ptrs)
            .map_err(|e| {
                GpuError::MemoryCopyError(format!("Failed to copy output pointers: {:?}", e))
            })?;

        d_output_ptr_arrays.push(d_ptrs);
    }

    // Copy sizes
    let sizes: Vec<i32> = batch.tasks().iter().map(|t| t.size() as i32).collect();
    let d_sizes = device.copy_to_device_i32(&sizes)?;

    // Copy parameters (as raw bytes)
    // This is generic - params can be any Copy type
    let d_params = Vec::new(); // Will be handled per-indicator type

    let using_pinned = h_inputs.iter().any(|task| task.iter().any(|h| h.is_some()));

    Ok(GenericBatchBuffers {
        d_inputs,
        h_inputs,
        d_outputs,
        h_outputs,
        d_input_ptr_arrays,
        d_output_ptr_arrays,
        d_sizes,
        d_params,
        using_pinned,
        _phantom: PhantomData,
    })
}

/// Upload input data to GPU buffers
///
/// Retained for the future generic launch implementation (currently unused).
#[allow(dead_code)]
fn upload_generic_data<I: PersistentIndicator>(
    device: &GpuDevice,
    batch: &GenericBatch<I>,
    buffers: &mut GenericBatchBuffers<I>,
) -> Result<(), GpuError> {
    for (task_idx, task) in batch.tasks().iter().enumerate() {
        for (input_idx, input_data) in task.inputs.iter().enumerate() {
            // Use pinned buffer if available
            if let Some(ref mut h_buf) = buffers.h_inputs[task_idx][input_idx] {
                h_buf.copy_from_slice(input_data);
                device
                    .stream
                    .memcpy_htod(h_buf.as_slice(), &mut buffers.d_inputs[task_idx][input_idx])
                    .map_err(|e| {
                        GpuError::MemoryCopyError(format!(
                            "Failed to upload task {} input {} (pinned): {:?}",
                            task_idx, input_idx, e
                        ))
                    })?;
            } else {
                device
                    .stream
                    .memcpy_htod(input_data, &mut buffers.d_inputs[task_idx][input_idx])
                    .map_err(|e| {
                        GpuError::MemoryCopyError(format!(
                            "Failed to upload task {} input {}: {:?}",
                            task_idx, input_idx, e
                        ))
                    })?;
            }
        }
    }
    Ok(())
}

/// Download results from GPU
///
/// Retained for the future generic launch implementation (currently unused).
#[allow(dead_code)]
fn download_generic_results<I: PersistentIndicator>(
    device: &GpuDevice,
    buffers: &GenericBatchBuffers<I>,
) -> Result<Vec<Vec<Vec<f64>>>, GpuError> {
    let mut results = Vec::with_capacity(buffers.d_outputs.len());

    for (task_idx, task_outputs) in buffers.d_outputs.iter().enumerate() {
        let mut task_results = Vec::with_capacity(task_outputs.len());

        for (output_idx, output_buf) in task_outputs.iter().enumerate() {
            // Use pinned buffer if available
            let result = if let Some(ref h_buf) = buffers.h_outputs[task_idx][output_idx] {
                let mut host_buf = vec![0.0; h_buf.len()];
                device
                    .stream
                    .memcpy_dtoh(output_buf, &mut host_buf)
                    .map_err(|e| {
                        GpuError::MemoryCopyError(format!(
                            "Failed to download task {} output {} (pinned): {:?}",
                            task_idx, output_idx, e
                        ))
                    })?;
                host_buf
            } else {
                device.copy_to_host(output_buf)?
            };

            task_results.push(result);
        }

        results.push(task_results);
    }

    Ok(results)
}

/// Error message returned while the generic kernel launch is unimplemented.
pub(crate) const GENERIC_BATCH_NOT_IMPLEMENTED_MSG: &str = "execute_generic_batch is not \
implemented: the generic multi-input/multi-output kernel launch was never wired up, so the \
previous version uploaded inputs, launched nothing, and returned zero-filled output buffers \
presented as results. Use gpu::persistent::execute_batch for single-output indicators.";

/// Execute generic batch using persistent kernel - NOT IMPLEMENTED
///
/// # Status
///
/// The generic kernel launch has never been implemented. The previous version
/// of this function allocated buffers, uploaded inputs, skipped the launch
/// entirely, and then downloaded the (zero-initialized) output buffers,
/// returning all-zero arrays as if they were indicator results. It now
/// returns an honest error instead of silently-wrong data.
///
/// Use [`crate::gpu::persistent::execute_batch`] for the supported
/// single-pointer-array indicator path.
///
/// # Errors
///
/// - `InvalidParameter` for an empty batch
/// - `ComputationErrorStatic` (always, after validation) until the generic
///   launch is implemented
pub fn execute_generic_batch<I: PersistentIndicator>(
    _device: &GpuDevice,
    batch: &GenericBatch<I>,
) -> Result<Vec<Vec<Vec<f64>>>, GpuError> {
    if batch.is_empty() {
        return Err(GpuError::InvalidParameter("Empty task batch".to_string()));
    }

    // Fail before any compilation, allocation, or transfer work: there is no
    // kernel launch to feed, so any GPU work here would be pure waste.
    Err(GpuError::ComputationErrorStatic(
        GENERIC_BATCH_NOT_IMPLEMENTED_MSG,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_batch_creation() {
        use crate::gpu::persistent::RocIndicator;

        let mut batch = GenericBatch::<RocIndicator>::new();
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());

        batch.add_single_input_task(vec![1.0, 2.0, 3.0], 14);
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    #[should_panic(expected = "Expected 1 inputs, got 3")]
    fn test_wrong_input_count_panics() {
        use crate::gpu::persistent::RocIndicator;

        let mut batch = GenericBatch::<RocIndicator>::new();
        // ROC expects 1 input, but we're giving 3 - should panic
        batch.add_task(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]], 14);
    }

    #[test]
    fn test_atr_multi_input() {
        use crate::gpu::persistent::AtrIndicator;

        let mut batch = GenericBatch::<AtrIndicator>::new();

        // ATR expects 3 inputs: high, low, close
        let high = vec![10.0, 11.0, 12.0];
        let low = vec![9.0, 10.0, 10.5];
        let close = vec![9.5, 10.5, 11.5];

        batch.add_task(vec![high, low, close], 14);
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_generic_batch_error_message_mentions_alternative() {
        // Host-side sanity check: the not-implemented error must point users
        // at the supported batch API.
        assert!(GENERIC_BATCH_NOT_IMPLEMENTED_MSG.contains("execute_batch"));
        assert!(GENERIC_BATCH_NOT_IMPLEMENTED_MSG.contains("not"));
    }

    #[test]
    fn test_macd_multi_output() {
        use crate::gpu::persistent::MacdIndicator;

        let mut batch = GenericBatch::<MacdIndicator>::new();

        // MACD expects 1 input but produces 3 outputs
        batch.add_single_input_task(
            vec![44.0, 44.5, 43.0],
            crate::gpu::persistent::MacdParams::standard(),
        );
        assert_eq!(batch.len(), 1);
        assert_eq!(MacdIndicator::num_outputs(), 3);
    }
}
