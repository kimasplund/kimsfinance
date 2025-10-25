//! GPU Device Management
//!
//! Handles CUDA context and stream initialization, memory allocation, and error handling.

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, result::DriverError};
use std::sync::Arc;

/// GPU device handle with memory management
///
/// Provides safe CUDA stream access for memory operations and kernel execution.
pub struct GpuDevice {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
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

        Ok(Self { context, stream })
    }

    /// Allocate GPU memory buffer
    ///
    /// # Arguments
    ///
    /// * `len` - Number of f64 elements to allocate
    pub fn alloc_buffer(&self, len: usize) -> Result<CudaSlice<f64>, GpuError> {
        self.stream.alloc_zeros::<f64>(len).map_err(|e| {
            GpuError::AllocationError(format!("Failed to allocate {} elements: {:?}", len, e))
        })
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

    /// Invalid parameters
    InvalidParameter(String),

    /// cudarc driver error
    DriverError(DriverError),
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
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError::DriverError(e)
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
}
