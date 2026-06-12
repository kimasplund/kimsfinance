//! FP8 GEMM using CUTLASS 3.5.0 for Ada Lovelace (sm_89) - UNAVAILABLE
//!
//! # Status: construction always fails
//!
//! The CUTLASS GEMM kernel source (`src/gpu/kernels/fp8_gemm_cutlass.cu`)
//! `#include`s CUTLASS 3.5 template headers. This crate JIT-compiles kernels
//! at runtime through NVRTC (`compile_ptx_optimized_cached`), which cannot
//! resolve header includes — so the previous implementation of
//! [`FP8GemmCutlass::new`] failed on every call with a confusing NVRTC
//! compilation error after attempting the doomed compile. It now fails fast
//! with an honest explanation instead.
//!
//! Making this module work requires a host-side nvcc/CUTLASS build step that
//! produces a CUBIN, plus a CUBIN loading path (`load_module` from file) —
//! neither exists today. No performance figures are quoted here: none were
//! ever measured, because the kernels never compiled at runtime.
//!
//! For a working tensor core matmul use
//! [`crate::gpu::fp8_wmma::FP8TensorCore::matmul_fp16`].

use crate::gpu::{GpuDevice, GpuError};
use cudarc::driver::{CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Error message returned by [`FP8GemmCutlass::new`].
pub(crate) const CUTLASS_REQUIRES_HOST_BUILD_MSG: &str = "FP8GemmCutlass is unavailable: \
kernels/fp8_gemm_cutlass.cu depends on CUTLASS headers (#include \"cutlass/...\") that NVRTC \
cannot compile at runtime, and no pre-compiled CUBIN loading path is implemented. A host-side \
nvcc/CUTLASS build would be required. Use FP8TensorCore::matmul_fp16 instead.";

/// FP8 GEMM kernel manager using CUTLASS 3.5.0 - currently unavailable
///
/// [`Self::new`] always returns an error (see module documentation); the
/// remaining methods are retained for API stability and for a future
/// host-side CUTLASS build integration.
pub struct FP8GemmCutlass {
    /// PTX module containing compiled kernels
    module: Arc<CudaModule>,
}

impl FP8GemmCutlass {
    /// Load compiled FP8 GEMM kernels - ALWAYS FAILS
    ///
    /// # Status
    ///
    /// The CUTLASS kernel source can never be compiled by the runtime NVRTC
    /// pipeline (it requires CUTLASS headers), and there is no CUBIN loading
    /// path. This constructor fails fast with an explanatory error instead
    /// of attempting (and burying the reason for) a doomed NVRTC compile.
    ///
    /// # Errors
    ///
    /// - `InsufficientComputeCapability` if the GPU is older than sm_89
    ///   (the operation could never work on such hardware anyway)
    /// - `ComputationErrorStatic` ([`CUTLASS_REQUIRES_HOST_BUILD_MSG`])
    ///   otherwise
    pub fn new(device: &GpuDevice) -> Result<Self, GpuError> {
        // Check GPU compute capability (must be 8.9 for Ada FP8)
        let (major, minor) = device.compute_capability();
        if major < 8 || (major == 8 && minor < 9) {
            return Err(GpuError::InsufficientComputeCapability {
                required: "8.9".to_string(),
                found: format!("{}.{}", major, minor),
            });
        }

        // Fail fast: NVRTC can never compile the CUTLASS-header-dependent
        // kernel source, so there is nothing to load.
        Err(GpuError::ComputationErrorStatic(
            CUTLASS_REQUIRES_HOST_BUILD_MSG,
        ))
    }

    /// Convert FP32 array to FP8 E4M3 on GPU
    ///
    /// Performs: FP32 → FP16 → FP8 with saturation.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - FP32 input array (device memory)
    ///
    /// # Returns
    ///
    /// FP8 E4M3 array (device memory, opaque pointer)
    pub fn fp32_to_fp8(
        &self,
        device: &GpuDevice,
        input: &CudaSlice<f32>,
    ) -> Result<CudaSlice<u8>, GpuError> {
        let n = input.len();

        // Allocate FP8 output (1 byte per element)
        let output = device.allocate_device_buffer::<u8>(n)?;

        // Load kernel
        let kernel = self
            .module
            .load_function("fp32_to_fp8_e4m3_cutlass")
            .map_err(|e| {
                GpuError::ExecutionError(format!(
                    "Failed to load fp32_to_fp8_e4m3_cutlass kernel: {:?}",
                    e
                ))
            })?;

        // Launch configuration
        let block_size = 256;
        let num_blocks = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let n_i32 = n as i32;
        let mut builder = device.stream.launch_builder(&kernel);
        builder.arg(input);
        builder.arg(&output);
        builder.arg(&n_i32);
        unsafe {
            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("FP32→FP8 conversion failed: {:?}", e))
            })?;
        }

        device.synchronize()?;

        Ok(output)
    }

    /// Convert FP8 E4M3 array to FP32 on GPU
    ///
    /// Performs: FP8 → FP16 → FP32 (exact inverse of `fp32_to_fp8`).
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - FP8 E4M3 input array (device memory, opaque)
    ///
    /// # Returns
    ///
    /// FP32 array (device memory)
    pub fn fp8_to_fp32(
        &self,
        device: &GpuDevice,
        input: &CudaSlice<u8>,
    ) -> Result<CudaSlice<f32>, GpuError> {
        let n = input.len();

        // Allocate FP32 output
        let mut output = device.allocate_device_buffer::<f32>(n)?;

        // Load kernel
        let kernel = self
            .module
            .load_function("fp8_e4m3_to_fp32_cutlass")
            .map_err(|e| {
                GpuError::ExecutionError(format!(
                    "Failed to load fp8_e4m3_to_fp32_cutlass kernel: {:?}",
                    e
                ))
            })?;

        // Launch configuration
        let block_size = 256;
        let num_blocks = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let n_i32 = n as i32;
        let mut builder = device.stream.launch_builder(&kernel);
        builder.arg(input);
        builder.arg(&mut output);
        builder.arg(&n_i32);
        unsafe {
            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("FP8→FP32 conversion failed: {:?}", e))
            })?;
        }

        device.synchronize()?;

        Ok(output)
    }

    /// FP8 GEMM: C = alpha * (A @ B) + beta * C
    ///
    /// Automatically selects optimal tile size based on matrix dimensions.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `a` - FP8 E4M3 matrix A (m × k, row-major)
    /// * `b` - FP8 E4M3 matrix B (k × n, row-major)
    /// * `m` - Rows of A and C
    /// * `n` - Columns of B and C
    /// * `k` - Columns of A, rows of B
    /// * `alpha` - Scaling factor for A*B (default: 1.0)
    /// * `beta` - Scaling factor for C (default: 0.0)
    ///
    /// # Returns
    ///
    /// FP32 matrix C (m × n, row-major)
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        device: &GpuDevice,
        a: &CudaSlice<u8>,
        b: &CudaSlice<u8>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<CudaSlice<f32>, GpuError> {
        // Validate dimensions
        if a.len() != m * k {
            return Err(GpuError::InvalidDimensions {
                expected: m * k,
                found: a.len(),
            });
        }
        if b.len() != k * n {
            return Err(GpuError::InvalidDimensions {
                expected: k * n,
                found: b.len(),
            });
        }

        // Allocate output (FP32)
        let mut c = device.allocate_device_buffer::<f32>(m * n)?;

        // Load auto-select kernel
        let kernel = self.module.load_function("fp8_gemm_auto").map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load fp8_gemm_auto kernel: {:?}", e))
        })?;

        // Launch configuration (CUTLASS manages parallelism internally)
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;

        let mut builder = device.stream.launch_builder(&kernel);
        builder.arg(a);
        builder.arg(b);
        builder.arg(&mut c);
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        builder.arg(&k_i32);
        builder.arg(&alpha);
        builder.arg(&beta);
        unsafe {
            builder
                .launch(config)
                .map_err(|e| GpuError::ExecutionError(format!("FP8 GEMM failed: {:?}", e)))?;
        }

        device.synchronize()?;

        Ok(c)
    }

    /// FP8 matrix multiplication: C = A @ B
    ///
    /// Convenience wrapper for `gemm` with alpha=1.0, beta=0.0.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let c = gemm.matmul(&device, &a_fp8, &b_fp8, m, n, k)?;
    /// ```
    pub fn matmul(
        &self,
        device: &GpuDevice,
        a: &CudaSlice<u8>,
        b: &CudaSlice<u8>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaSlice<f32>, GpuError> {
        self.gemm(device, a, b, m, n, k, 1.0, 0.0)
    }

    /// Batched FP8 GEMM: C[i] = A[i] @ B[i] for all i
    ///
    /// Performs multiple independent GEMMs in parallel.
    /// All matrices must have the same dimensions.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `a_batch` - Batched FP8 matrices A (batch_size × m × k)
    /// * `b_batch` - Batched FP8 matrices B (batch_size × k × n)
    /// * `batch_size` - Number of matrices
    /// * `m`, `n`, `k` - Dimensions (same for all matrices)
    ///
    /// # Returns
    ///
    /// Batched FP32 matrices C (batch_size × m × n)
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched(
        &self,
        device: &GpuDevice,
        a_batch: &CudaSlice<u8>,
        b_batch: &CudaSlice<u8>,
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<CudaSlice<f32>, GpuError> {
        // Validate dimensions
        if a_batch.len() != batch_size * m * k {
            return Err(GpuError::InvalidDimensions {
                expected: batch_size * m * k,
                found: a_batch.len(),
            });
        }
        if b_batch.len() != batch_size * k * n {
            return Err(GpuError::InvalidDimensions {
                expected: batch_size * k * n,
                found: b_batch.len(),
            });
        }

        // Allocate output
        let mut c_batch = device.allocate_device_buffer::<f32>(batch_size * m * n)?;

        // Load batched kernel
        let kernel = self.module.load_function("fp8_gemm_batched").map_err(|e| {
            GpuError::ExecutionError(format!("Failed to load fp8_gemm_batched kernel: {:?}", e))
        })?;

        // Launch configuration (one block per batch element)
        let config = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let batch_size_i32 = batch_size as i32;
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;

        let mut builder = device.stream.launch_builder(&kernel);
        builder.arg(a_batch);
        builder.arg(b_batch);
        builder.arg(&mut c_batch);
        builder.arg(&batch_size_i32);
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        builder.arg(&k_i32);
        builder.arg(&alpha);
        builder.arg(&beta);
        unsafe {
            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Batched FP8 GEMM failed: {:?}", e))
            })?;
        }

        device.synchronize()?;

        Ok(c_batch)
    }

    /// Test FP8 GEMM functionality
    ///
    /// Runs a simple 4x4 GEMM to verify:
    /// - FP8 conversion works
    /// - CUTLASS kernels execute
    /// - Numerical accuracy is acceptable
    ///
    /// # Returns
    ///
    /// - `Ok(())` if test passed
    /// - `Err(GpuError)` if test failed
    pub fn test(&self, device: &GpuDevice) -> Result<(), GpuError> {
        // Allocate test result buffer
        let mut test_result = device.allocate_device_buffer::<f32>(1)?;

        // Load test kernel
        let kernel = self
            .module
            .load_function("test_fp8_gemm_cutlass")
            .map_err(|e| {
                GpuError::ExecutionError(format!(
                    "Failed to load test_fp8_gemm_cutlass kernel: {:?}",
                    e
                ))
            })?;

        // Launch test kernel
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = device.stream.launch_builder(&kernel);
        builder.arg(&mut test_result);
        unsafe {
            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("FP8 test kernel failed: {:?}", e))
            })?;
        }

        device.synchronize()?;

        // Check result (copy f32 buffer to host manually)
        let result_host: Vec<f32> = device.stream.memcpy_dtov(&test_result).map_err(|e| {
            GpuError::MemoryCopyError(format!("Failed to copy test result: {:?}", e))
        })?;

        if result_host[0] == 1.0 {
            Ok(())
        } else {
            Err(GpuError::ExecutionError(
                "FP8 GEMM test failed: kernel returned error".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unavailable_message_is_honest() {
        // The fail-fast error must explain the root cause (CUTLASS headers
        // vs NVRTC) and point at the working alternative.
        assert!(CUTLASS_REQUIRES_HOST_BUILD_MSG.contains("CUTLASS"));
        assert!(CUTLASS_REQUIRES_HOST_BUILD_MSG.contains("NVRTC"));
        assert!(CUTLASS_REQUIRES_HOST_BUILD_MSG.contains("matmul_fp16"));
    }

    #[test]
    fn test_cutlass_kernel_source_cannot_be_nvrtc_compiled() {
        // Documents WHY new() fails fast: the kernel source depends on
        // header includes, which the NVRTC pipeline used by this crate
        // (compile_ptx_optimized_cached) cannot resolve. If this assertion
        // ever fails, the source became self-contained and the fail-fast
        // gate in new() should be revisited.
        const SRC: &str = include_str!("kernels/fp8_gemm_cutlass.cu");
        assert!(
            SRC.contains("#include"),
            "fp8_gemm_cutlass.cu no longer has includes; re-evaluate FP8GemmCutlass::new()"
        );
        assert!(SRC.contains("cutlass"), "expected CUTLASS dependency");
    }

    #[test]
    #[ignore] // Requires GPU (any CUDA device)
    fn test_fp8_gemm_cutlass_new_fails_fast() {
        let device = GpuDevice::new().expect("Failed to create GPU device");
        match FP8GemmCutlass::new(&device) {
            Err(GpuError::ComputationErrorStatic(msg)) => {
                assert!(msg.contains("CUTLASS"));
            }
            Err(GpuError::InsufficientComputeCapability { .. }) => {
                // Acceptable on pre-Ada hardware
            }
            other => panic!(
                "FP8GemmCutlass::new must fail fast, got {:?}",
                other.map(|_| "Ok(FP8GemmCutlass)")
            ),
        }
    }
}
