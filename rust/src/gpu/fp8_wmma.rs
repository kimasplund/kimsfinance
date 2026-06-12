//! FP8 WMMA Tensor Core Support for Ada Lovelace GPUs
//!
//! This module provides hardware FP8 tensor core acceleration using WMMA (Warp Matrix Multiply-Accumulate)
//! instructions on NVIDIA Ada Lovelace GPUs (Compute Capability 8.9+, e.g., RTX 3500 Ada).
//!
//! # FP8 Format (E4M3)
//!
//! - 1 sign bit
//! - 4 exponent bits (bias 7)
//! - 3 mantissa bits
//! - Range: ±448
//! - Precision: ~2 decimal digits (0.01 resolution)
//!
//! # Status
//!
//! Only the **FP16 matmul path is functional**. `matmul_fp8` and
//! `matmul_tf32` are gated off with explanatory errors because their previous
//! implementations produced numerically-wrong results (see the function docs).
//! No speedup figures are quoted here: none have been validated on real
//! hardware for this module.
//!
//! # Hardware Requirements
//!
//! - GPU: NVIDIA Ada Lovelace (RTX 3500 Ada, RTX 4000 series)
//! - Compute Capability: 8.9+
//! - CUDA Driver: 11.8+
//! - CUDA Toolkit: 12.0+ (for FP8 support at runtime)
//!
//! # Compilation Strategy
//!
//! This module uses **cached JIT compilation** instead of AOT:
//! - FP8 kernels are embedded as CUDA source via `include_str!`
//! - Compiled on first use with `compile_ptx_optimized_cached()`
//! - PTX cached in memory for subsequent uses (50-200x faster)
//! - No build.rs dependency (works even if nvcc not in PATH at build time)
//! - Runtime compilation only happens once per process
//!
//! # Example
//!
//! ```rust,ignore
//! use kimsfinance_core::gpu::{GpuDevice, FP8TensorCore};
//!
//! let device = GpuDevice::new()?;
//! let fp8_core = FP8TensorCore::new(&device)?;
//!
//! if fp8_core.is_fp16_supported() {
//!     // FP16 tensor core matmul (the validated path)
//!     let result = fp8_core.matmul_fp16(&a, &b, m, n, k)?;
//! }
//! ```

use crate::gpu::{GpuDevice, GpuError};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig};
use std::sync::Arc;

/// Error message returned by [`FP8TensorCore::matmul_fp8`] while the FP8 path
/// is disabled.
pub(crate) const FP8_MATMUL_UNSUPPORTED_MSG: &str = "matmul_fp8 is disabled: the previous \
implementation quantized FP32 values in place (still stored as f32) and fed them to a \
byte-oriented FP8 MMA kernel, producing garbage results. A correct implementation needs real \
E4M3 byte packing plus runtime numerical validation. Use matmul_fp16 instead.";

/// Error message returned by [`FP8TensorCore::matmul_tf32`] while the TF32
/// path is disabled.
pub(crate) const TF32_MATMUL_UNSUPPORTED_MSG: &str = "matmul_tf32 is disabled: no TF32 kernel \
exists; the previous implementation aliased the FP16 (u16 half) MMA kernel and fed it f32 \
buffers, producing garbage results. Use matmul_fp16 instead.";

/// Whether a compute capability supports FP8 E4M3 tensor cores
///
/// FP8 MMA instructions exist on Ada Lovelace (8.9) and on every newer major
/// architecture (Hopper 9.x, Blackwell 10.x/12.x). A previous version used
/// `major >= 8 && minor >= 9`, which wrongly excluded e.g. 9.0 and 10.0.
pub(crate) const fn supports_fp8_hardware(major: u32, minor: u32) -> bool {
    major > 8 || (major == 8 && minor >= 9)
}

/// Software FP8 quantization kernel (element-wise, NVRTC-compatible)
///
/// Mirrors [`quantize_fp8_cpu`]: clamp to ±448 and round to 0.01 steps.
/// Values stay in `float` storage — this simulates FP8 precision, it does
/// NOT produce packed E4M3 bytes.
const QUANTIZE_KERNEL: &str = r#"
extern "C" __global__ void quantize_fp8_kernel(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float value = input[idx];

    // Handle special values
    if (isnan(value) || isinf(value)) {
        output[idx] = value;
        return;
    }

    // FP8 E4M3 range: ±448
    const float MAX_FP8 = 448.0f;
    if (fabsf(value) > MAX_FP8) {
        output[idx] = copysignf(MAX_FP8, value);
        return;
    }

    // Quantize to ~2 decimal digits (100 steps)
    const float SCALE = 100.0f;
    output[idx] = roundf(value * SCALE) / SCALE;
}
"#;

/// FP8 E4M3 format tensor core wrapper
///
/// Provides tensor core matrix multiplication using embedded CUDA source with
/// cached NVRTC JIT compilation:
/// - Zero build-time dependencies (no nvcc required at build time)
/// - Fast initialization after first compilation (PTX cached per process)
/// - Automatic architecture optimization (via compile_ptx_optimized_cached)
///
/// # Precision Modes
///
/// - **FP16** (Volta sm_70+): ±65,504 range, 3-4 decimal digit precision —
///   the only functional matmul path ([`Self::matmul_fp16`])
/// - **FP8 E4M3** (Ada sm_89+): hardware detection only;
///   [`Self::matmul_fp8`] is disabled (see [`FP8_MATMUL_UNSUPPORTED_MSG`])
/// - **TF32** (Ampere sm_80+): hardware detection only;
///   [`Self::matmul_tf32`] is disabled (see [`TF32_MATMUL_UNSUPPORTED_MSG`])
pub struct FP8TensorCore {
    device: Arc<GpuDevice>,
    compute_capability: (u32, u32),
    fp8_supported: bool,
    fp16_supported: bool,
    tf32_supported: bool,

    // FP8 MMA kernel (compiled for capability detection; the matmul_fp8 entry
    // point is gated off until a numerically-validated host path exists)
    fp8_module: Option<Arc<CudaModule>>,
    fp8_matmul_kernel: Option<CudaFunction>,

    // FP16 kernels and conversions
    fp16_module: Option<Arc<CudaModule>>,
    fp16_matmul_kernel: Option<CudaFunction>,
    fp32_to_fp16_kernel: Option<CudaFunction>,
    fp16_to_fp32_kernel: Option<CudaFunction>,

    // FP32/TF32 tensor core kernels
    fp32_module: Option<Arc<CudaModule>>,
    tf32_matmul_kernel: Option<CudaFunction>,
}

impl FP8TensorCore {
    /// Create FP8 tensor core context
    ///
    /// Verifies GPU compute capability and JIT-compiles the tensor core
    /// kernels for the supported precision modes (cached after first call).
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    ///
    /// # Returns
    ///
    /// - `Ok(FP8TensorCore)` if GPU supports FP8 and kernels loaded successfully
    /// - `Err(FP8Error::UnsupportedHardware)` if GPU doesn't support FP8 (compute capability < 8.9)
    /// - `Err(FP8Error::ModuleLoadFailed)` if CUDA Toolkit not available at runtime
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = GpuDevice::new()?;
    /// let fp8_core = FP8TensorCore::new(&device)?;
    /// assert!(fp8_core.is_fp8_supported()); // RTX 3500 Ada = 8.9
    /// ```
    pub fn new(device: Arc<GpuDevice>) -> Result<Self, FP8Error> {
        // Get compute capability
        let compute_capability = device.compute_capability();

        // Detect supported precision modes
        let fp8_supported = supports_fp8_hardware(compute_capability.0, compute_capability.1); // Ada sm_89+
        let fp16_supported = compute_capability.0 >= 7; // Volta sm_70+
        let tf32_supported = compute_capability.0 >= 8; // Ampere sm_80+

        // Create instance with all precision modes
        let mut instance = FP8TensorCore {
            device,
            compute_capability,
            fp8_supported,
            fp16_supported,
            tf32_supported,

            // FP8 kernels
            fp8_module: None,
            fp8_matmul_kernel: None,

            // FP16 kernels
            fp16_module: None,
            fp16_matmul_kernel: None,
            fp32_to_fp16_kernel: None,
            fp16_to_fp32_kernel: None,

            // FP32/TF32 kernels
            fp32_module: None,
            tf32_matmul_kernel: None,
        };

        // Load kernels for supported precision modes (graceful degradation)
        // We don't fail if kernels don't load - just mark as unsupported
        if fp8_supported && let Err(e) = instance.load_fp8_kernels() {
            eprintln!("⚠️  FP8 kernels failed to load: {}", e);
            instance.fp8_supported = false;
        }

        if fp16_supported && let Err(e) = instance.load_fp16_kernels() {
            eprintln!("⚠️  FP16 kernels failed to load: {}", e);
            instance.fp16_supported = false;
        }

        if tf32_supported && let Err(e) = instance.load_fp32_kernels() {
            eprintln!("⚠️  FP32/TF32 kernels failed to load: {}", e);
            instance.tf32_supported = false;
        }

        // Verify at least one precision mode is available
        if !instance.fp8_supported && !instance.fp16_supported && !instance.tf32_supported {
            return Err(FP8Error::UnsupportedHardware(format!(
                "No tensor core kernels loaded for compute capability {}.{}. Minimum requirement: sm_70 (Volta)",
                compute_capability.0, compute_capability.1
            )));
        }

        Ok(instance)
    }

    /// Check if hardware supports FP8 tensor cores
    ///
    /// # Returns
    ///
    /// `true` if compute capability >= 8.9 (Ada Lovelace or newer) AND kernels loaded successfully
    pub fn is_fp8_supported(&self) -> bool {
        self.fp8_supported && self.fp8_matmul_kernel.is_some()
    }

    /// Check if hardware supports FP16 tensor cores
    ///
    /// # Returns
    ///
    /// `true` if compute capability >= 7.0 (Volta or newer) AND kernels loaded successfully
    pub fn is_fp16_supported(&self) -> bool {
        self.fp16_supported && self.fp16_matmul_kernel.is_some()
    }

    /// Check if hardware supports TF32 tensor cores
    ///
    /// # Returns
    ///
    /// `true` if compute capability >= 8.0 (Ampere or newer) AND kernels loaded successfully
    pub fn is_tf32_supported(&self) -> bool {
        self.tf32_supported && self.tf32_matmul_kernel.is_some()
    }

    /// Get compute capability
    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    /// Load FP8 kernels from embedded source using cached JIT compilation
    ///
    /// Only the MMA kernel is compiled (capability/compile detection). The
    /// FP8 conversion kernels were dropped along with the disabled
    /// `matmul_fp8` host path; they produced "FP8" values stored as f32,
    /// which the byte-oriented MMA kernel cannot consume.
    fn load_fp8_kernels(&mut self) -> Result<(), FP8Error> {
        // Load FP8 E4M3 tensor core kernel using RAW PTX inline assembly
        const FP8_MMA_KERNELS: &str = include_str!("kernels/fp8_mma_ptx.cu");

        // Compile FP8 MMA kernel
        let ptx_arc =
            crate::gpu::compile::compile_ptx_optimized_cached(FP8_MMA_KERNELS).map_err(|e| {
                FP8Error::ModuleLoadFailed(format!("Failed to compile FP8 MMA kernels: {:?}", e))
            })?;

        let module_mma = self
            .device
            .context()
            .load_module(std::sync::Arc::unwrap_or_clone(ptx_arc))
            .map_err(|e| {
                FP8Error::ModuleLoadFailed(format!("Failed to load FP8 MMA module: {:?}", e))
            })?;

        let fp8_matmul_kernel = module_mma
            .load_function("fp8_matmul_mma_ptx")
            .map_err(|e| {
                FP8Error::ModuleLoadFailed(format!("Failed to load fp8_matmul_mma_ptx: {:?}", e))
            })?;

        // Store module and kernel
        self.fp8_module = Some(module_mma);
        self.fp8_matmul_kernel = Some(fp8_matmul_kernel);

        Ok(())
    }

    /// Load FP16 kernels from embedded source using cached JIT compilation
    fn load_fp16_kernels(&mut self) -> Result<(), FP8Error> {
        // Load FP16 raw PTX tensor core kernel + conversions (NVRTC compatible)
        // Note: Using fp16_mma_ptx.cu (raw PTX) instead of fp16_wmma.cu (requires mma.h)
        const FP16_MMA_KERNELS: &str = include_str!("kernels/fp16_mma_ptx.cu");
        const FP16_CONVERSION_KERNELS: &str = include_str!("kernels/fp16_conversions.cu");

        // Compile FP16 MMA kernel (raw PTX)
        let ptx_mma_arc = crate::gpu::compile::compile_ptx_optimized_cached(FP16_MMA_KERNELS)
            .map_err(|e| {
                FP8Error::ModuleLoadFailed(format!("Failed to compile FP16 MMA kernels: {:?}", e))
            })?;

        let module_mma = self
            .device
            .context()
            .load_module(std::sync::Arc::unwrap_or_clone(ptx_mma_arc))
            .map_err(|e| {
                FP8Error::ModuleLoadFailed(format!("Failed to load FP16 MMA module: {:?}", e))
            })?;

        let fp16_matmul_kernel = module_mma
            .load_function("fp16_matmul_mma_ptx")
            .map_err(|e| {
                FP8Error::ModuleLoadFailed(format!("Failed to load fp16_matmul_mma_ptx: {:?}", e))
            })?;

        // Compile FP16 conversion kernels
        let ptx_conv_arc = crate::gpu::compile::compile_ptx_optimized_cached(
            FP16_CONVERSION_KERNELS,
        )
        .map_err(|e| {
            FP8Error::ModuleLoadFailed(format!(
                "Failed to compile FP16 conversion kernels: {:?}",
                e
            ))
        })?;

        let module_conv = self
            .device
            .context()
            .load_module(std::sync::Arc::unwrap_or_clone(ptx_conv_arc))
            .map_err(|e| {
                FP8Error::ModuleLoadFailed(format!(
                    "Failed to load FP16 conversion module: {:?}",
                    e
                ))
            })?;

        let fp32_to_fp16_kernel = module_conv.load_function("fp32_to_fp16").map_err(|e| {
            FP8Error::ModuleLoadFailed(format!("Failed to load fp32_to_fp16: {:?}", e))
        })?;

        let fp16_to_fp32_kernel = module_conv.load_function("fp16_to_fp32").map_err(|e| {
            FP8Error::ModuleLoadFailed(format!("Failed to load fp16_to_fp32: {:?}", e))
        })?;

        self.fp16_module = Some(module_mma);
        self.fp16_matmul_kernel = Some(fp16_matmul_kernel);
        self.fp32_to_fp16_kernel = Some(fp32_to_fp16_kernel);
        self.fp16_to_fp32_kernel = Some(fp16_to_fp32_kernel);

        Ok(())
    }

    /// Load FP32/TF32 tensor core kernels
    fn load_fp32_kernels(&mut self) -> Result<(), FP8Error> {
        // Load FP32 MMA PTX tensor core kernel (supports TF32 mode on Ampere+)
        const FP32_KERNELS: &str = include_str!("kernels/fp16_mma_ptx.cu");

        let ptx_arc =
            crate::gpu::compile::compile_ptx_optimized_cached(FP32_KERNELS).map_err(|e| {
                FP8Error::ModuleLoadFailed(format!("Failed to compile FP32 kernels: {:?}", e))
            })?;

        let module = self
            .device
            .context()
            .load_module(std::sync::Arc::unwrap_or_clone(ptx_arc))
            .map_err(|e| {
                FP8Error::ModuleLoadFailed(format!("Failed to load FP32 module: {:?}", e))
            })?;

        // FP16 kernel can be used for TF32 (just using FP32 inputs/outputs)
        let tf32_matmul_kernel = module.load_function("fp16_matmul_mma_ptx").map_err(|e| {
            FP8Error::ModuleLoadFailed(format!("Failed to load fp16_matmul_mma_ptx: {:?}", e))
        })?;

        self.fp32_module = Some(module);
        self.tf32_matmul_kernel = Some(tf32_matmul_kernel);

        Ok(())
    }

    /// FP8 matrix multiplication using tensor cores - DISABLED
    ///
    /// # Status: always returns an error
    ///
    /// The previous implementation was verifiably wrong: it "quantized" FP32
    /// inputs with an element-wise rounding kernel that still stored the
    /// values as `f32`, then fed those 4-byte values to `fp8_matmul_mma_ptx`,
    /// which reads packed 1-byte E4M3 operands. The kernel therefore consumed
    /// reinterpreted float bit patterns and produced garbage. A correct host
    /// path requires real E4M3 byte packing (u8 buffers end-to-end) plus
    /// runtime numerical validation against an FP32 reference, neither of
    /// which exists yet.
    ///
    /// Use [`Self::matmul_fp16`] for a working tensor core matmul.
    ///
    /// # Errors
    ///
    /// Always returns [`FP8_MATMUL_UNSUPPORTED_MSG`] wrapped in
    /// `FP8Error::GpuError(GpuError::ComputationErrorStatic)`.
    pub fn matmul_fp8(
        &self,
        _a: &CudaSlice<f32>,
        _b: &CudaSlice<f32>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaSlice<f32>, FP8Error> {
        Err(FP8Error::GpuError(GpuError::ComputationErrorStatic(
            FP8_MATMUL_UNSUPPORTED_MSG,
        )))
    }

    /// FP16 matrix multiplication using tensor cores
    ///
    /// Performs C = A * B using FP16 tensor cores with FP32 accumulation.
    /// Available on Volta GPUs and newer (compute capability >= 7.0).
    ///
    /// # Precision
    ///
    /// ~3-4 decimal digits (±65,504 range); inputs are converted
    /// FP32 → FP16 on device and the result back to FP32.
    pub fn matmul_fp16(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaSlice<f32>, FP8Error> {
        if !self.is_fp16_supported() {
            return Err(FP8Error::ExecutionFailed(
                "FP16 kernels not loaded. Use is_fp16_supported() to check before calling."
                    .to_string(),
            ));
        }

        // Validate dimensions
        if a.len() != m * k {
            return Err(FP8Error::ExecutionFailed(format!(
                "Matrix A size mismatch: expected {} ({}x{}), got {}",
                m * k,
                m,
                k,
                a.len()
            )));
        }
        if b.len() != k * n {
            return Err(FP8Error::ExecutionFailed(format!(
                "Matrix B size mismatch: expected {} ({}x{}), got {}",
                k * n,
                k,
                n,
                b.len()
            )));
        }

        // STEP 1: Convert FP32 → FP16
        let a_fp16 = self.convert_fp32_to_fp16(a)?;
        let b_fp16 = self.convert_fp32_to_fp16(b)?;

        // STEP 2: FP16 tensor core matmul (hardware accelerated)
        let c_fp16 = self.matmul_fp16_internal(&a_fp16, &b_fp16, m, n, k)?;

        // STEP 3: Convert FP16 → FP32
        let c_fp32 = self.convert_fp16_to_fp32(&c_fp16)?;

        Ok(c_fp32)
    }

    /// Internal FP16 matmul (assumes inputs in FP16, outputs FP16)
    fn matmul_fp16_internal(
        &self,
        a_fp16: &CudaSlice<u16>,
        b_fp16: &CudaSlice<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaSlice<u16>, FP8Error> {
        // Allocate output buffer (FP16 = u16)
        let mut c_fp16 = self.device.allocate_device_buffer(m * n).map_err(|e| {
            FP8Error::ExecutionFailed(format!("Failed to allocate output: {:?}", e))
        })?;

        // FP16 raw PTX tensor cores work on 16x8x16 tiles
        let tile_m = 16;
        let tile_n = 8;
        let blocks_m = m.div_ceil(tile_m);
        let blocks_n = n.div_ceil(tile_n);

        let config = LaunchConfig {
            grid_dim: (blocks_m as u32, blocks_n as u32, 1),
            block_dim: (32, 1, 1), // 1 warp per block
            shared_mem_bytes: 0,
        };

        use cudarc::driver::PushKernelArg;

        let kernel = self.fp16_matmul_kernel.as_ref().unwrap();
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;

        let mut builder = self.device.stream.launch_builder(kernel);
        builder.arg(a_fp16);
        builder.arg(b_fp16);
        builder.arg(&mut c_fp16);
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        builder.arg(&k_i32);

        unsafe {
            builder.launch(config).map_err(|e| {
                FP8Error::ExecutionFailed(format!("FP16 matmul kernel launch failed: {:?}", e))
            })?;
        }

        Ok(c_fp16)
    }

    /// Convert FP32 to FP16 format
    fn convert_fp32_to_fp16(&self, input: &CudaSlice<f32>) -> Result<CudaSlice<u16>, FP8Error> {
        let n = input.len();
        let mut output = self.device.allocate_device_buffer(n).map_err(|e| {
            FP8Error::ExecutionFailed(format!("Failed to allocate FP16 buffer: {:?}", e))
        })?;

        let block_size = 256;
        let n_blocks = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        use cudarc::driver::PushKernelArg;

        let kernel = self.fp32_to_fp16_kernel.as_ref().unwrap();
        let n_i32 = n as i32;

        let mut builder = self.device.stream.launch_builder(kernel);
        builder.arg(input);
        builder.arg(&mut output);
        builder.arg(&n_i32);

        unsafe {
            builder.launch(config).map_err(|e| {
                FP8Error::ExecutionFailed(format!("FP32→FP16 conversion failed: {:?}", e))
            })?;
        }

        Ok(output)
    }

    /// Convert FP16 to FP32 format
    fn convert_fp16_to_fp32(&self, input: &CudaSlice<u16>) -> Result<CudaSlice<f32>, FP8Error> {
        let n = input.len();
        let mut output = self.device.allocate_device_buffer(n).map_err(|e| {
            FP8Error::ExecutionFailed(format!("Failed to allocate FP32 buffer: {:?}", e))
        })?;

        let block_size = 256;
        let n_blocks = n.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        use cudarc::driver::PushKernelArg;

        let kernel = self.fp16_to_fp32_kernel.as_ref().unwrap();
        let n_i32 = n as i32;

        let mut builder = self.device.stream.launch_builder(kernel);
        builder.arg(input);
        builder.arg(&mut output);
        builder.arg(&n_i32);

        unsafe {
            builder.launch(config).map_err(|e| {
                FP8Error::ExecutionFailed(format!("FP16→FP32 conversion failed: {:?}", e))
            })?;
        }

        Ok(output)
    }

    /// TF32 matrix multiplication using tensor cores - DISABLED
    ///
    /// # Status: always returns an error
    ///
    /// No TF32 kernel exists in this crate. The previous implementation
    /// loaded `fp16_matmul_mma_ptx` (which reads packed u16 half-precision
    /// operands) under the name "TF32" and passed it raw `f32` buffers, so
    /// the kernel reinterpreted float bit halves as FP16 values and produced
    /// garbage. A real TF32 path needs a dedicated `mma.sync` tf32 kernel
    /// plus runtime numerical validation.
    ///
    /// Use [`Self::matmul_fp16`] for a working tensor core matmul.
    ///
    /// # Errors
    ///
    /// Always returns [`TF32_MATMUL_UNSUPPORTED_MSG`] wrapped in
    /// `FP8Error::GpuError(GpuError::ComputationErrorStatic)`.
    pub fn matmul_tf32(
        &self,
        _a: &CudaSlice<f32>,
        _b: &CudaSlice<f32>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaSlice<f32>, FP8Error> {
        Err(FP8Error::GpuError(GpuError::ComputationErrorStatic(
            TF32_MATMUL_UNSUPPORTED_MSG,
        )))
    }

    /// Batch convert FP32 values to FP8 E4M3 format (software simulation)
    ///
    /// This is a fallback for GPUs without FP8 hardware support.
    /// Hardware FP8 conversion is done automatically in the tensor core kernel.
    ///
    /// # Arguments
    ///
    /// * `values` - FP32 values on device
    ///
    /// # Returns
    ///
    /// Device buffer with quantized FP8 values (stored as FP32 for compatibility)
    pub fn quantize_fp8_batch(&self, values: &CudaSlice<f32>) -> Result<CudaSlice<f32>, FP8Error> {
        // Allocate output buffer (f32 for FP8 values stored in FP32 format)
        let mut quantized = self
            .device
            .allocate_device_buffer::<f32>(values.len())
            .map_err(|e| {
                FP8Error::ExecutionFailed(format!("Failed to allocate quantized buffer: {:?}", e))
            })?;

        // Launch quantization kernel
        let block_size = 256;
        let n_blocks = values.len().div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let ptx =
            crate::gpu::compile::compile_ptx_optimized_cached(QUANTIZE_KERNEL).map_err(|e| {
                FP8Error::CompilationFailed(format!("Quantize kernel compilation failed: {:?}", e))
            })?;

        let module = self
            .device
            .context()
            .load_module(Arc::unwrap_or_clone(ptx))
            .map_err(|e| {
                FP8Error::CompilationFailed(format!("Failed to load quantize module: {:?}", e))
            })?;

        let kernel = module.load_function("quantize_fp8_kernel").map_err(|e| {
            FP8Error::CompilationFailed(format!("Failed to load quantize_fp8_kernel: {:?}", e))
        })?;

        use cudarc::driver::PushKernelArg;

        let n_i32 = values.len() as i32;
        let mut builder = self.device.stream.launch_builder(&kernel);
        builder.arg(values);
        builder.arg(&mut quantized);
        builder.arg(&n_i32);

        unsafe {
            builder.launch(config).map_err(|e| {
                FP8Error::ExecutionFailed(format!("Quantize kernel launch failed: {:?}", e))
            })?;
        }

        Ok(quantized)
    }
}

/// FP8 error types
#[derive(Debug, thiserror::Error)]
pub enum FP8Error {
    #[error("Hardware does not support FP8: {0}")]
    UnsupportedHardware(String),

    #[error("FP8 module loading failed: {0}")]
    ModuleLoadFailed(String),

    #[error("FP8 kernel compilation failed: {0}")]
    CompilationFailed(String),

    #[error("FP8 kernel execution failed: {0}")]
    ExecutionFailed(String),

    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
}

/// Software FP8 quantization (CPU fallback)
///
/// Simulates FP8 E4M3 precision on CPU (clamp to ±448, round to 0.01 steps).
///
/// # Arguments
///
/// * `value` - FP32/FP64 value to quantize
///
/// # Returns
///
/// Quantized value with ~2 decimal digits precision
///
/// # Example
///
/// ```rust
/// use kimsfinance_core::gpu::fp8_wmma::quantize_fp8_cpu;
///
/// assert_eq!(quantize_fp8_cpu(1.234567), 1.23);
/// assert_eq!(quantize_fp8_cpu(100.456), 100.46);
/// assert_eq!(quantize_fp8_cpu(500.0), 448.0); // Clamped to FP8 range
/// ```
pub fn quantize_fp8_cpu(value: f64) -> f64 {
    if value.is_nan() || value.is_infinite() {
        return value;
    }

    // FP8 E4M3 has range ±448 (roughly)
    let max_fp8 = 448.0;
    if value.abs() > max_fp8 {
        return value.signum() * max_fp8;
    }

    // Quantize to ~2 decimal digits (100 steps)
    let scale = 100.0;
    (value * scale).round() / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_fp8_cpu() {
        assert_eq!(quantize_fp8_cpu(1.234567), 1.23);
        assert_eq!(quantize_fp8_cpu(100.456), 100.46);
        assert_eq!(quantize_fp8_cpu(-50.789), -50.79);
        assert_eq!(quantize_fp8_cpu(500.0), 448.0); // Clamped to max
        assert_eq!(quantize_fp8_cpu(-500.0), -448.0); // Clamped to min
        assert!(quantize_fp8_cpu(f64::NAN).is_nan());
        assert!(quantize_fp8_cpu(f64::INFINITY).is_infinite());
    }

    #[test]
    fn test_quantize_fp8_precision() {
        // Test ~2 decimal digits precision
        let values = vec![
            1.111, 2.222, 3.333, 10.105, 99.999, 100.001, 200.555, 447.999,
        ];
        for val in values {
            let quantized = quantize_fp8_cpu(val);
            let error = (val - quantized).abs();
            assert!(
                error < 0.01,
                "Value {} quantized to {} with error {}",
                val,
                quantized,
                error
            );
        }
    }

    #[test]
    fn test_quantize_fp8_range() {
        // Test FP8 E4M3 range limits
        assert_eq!(quantize_fp8_cpu(448.0), 448.0); // Exactly at max
        assert_eq!(quantize_fp8_cpu(-448.0), -448.0); // Exactly at min
        assert_eq!(quantize_fp8_cpu(1000.0), 448.0); // Beyond max
        assert_eq!(quantize_fp8_cpu(-1000.0), -448.0); // Beyond min
    }

    #[test]
    fn test_supports_fp8_hardware_capability_matrix() {
        // Ada Lovelace (sm_89): the minimum supported capability
        assert!(supports_fp8_hardware(8, 9));
        // Hopper (sm_90): minor == 0 — the old `major >= 8 && minor >= 9`
        // check wrongly rejected this
        assert!(supports_fp8_hardware(9, 0));
        // Blackwell (sm_100 / sm_120): also minor == 0
        assert!(supports_fp8_hardware(10, 0));
        assert!(supports_fp8_hardware(12, 0));
        // Ampere (sm_80, sm_86): no FP8 tensor cores
        assert!(!supports_fp8_hardware(8, 0));
        assert!(!supports_fp8_hardware(8, 6));
        // Turing/Volta: no FP8 tensor cores
        assert!(!supports_fp8_hardware(7, 5));
        assert!(!supports_fp8_hardware(7, 0));
    }

    #[test]
    fn test_disabled_matmul_messages_point_at_fp16() {
        // The honest-gating errors must tell users which path works.
        assert!(FP8_MATMUL_UNSUPPORTED_MSG.contains("matmul_fp16"));
        assert!(TF32_MATMUL_UNSUPPORTED_MSG.contains("matmul_fp16"));
        // And explain why the path is disabled.
        assert!(FP8_MATMUL_UNSUPPORTED_MSG.contains("disabled"));
        assert!(TF32_MATMUL_UNSUPPORTED_MSG.contains("disabled"));
    }

    #[test]
    fn test_quantize_kernel_is_nvrtc_compatible() {
        // NVRTC compilation (compile_ptx_optimized_cached) cannot resolve
        // header includes; the kernel must be self-contained with an
        // extern "C" entry point matching the load_function() name.
        assert!(
            !QUANTIZE_KERNEL.contains("#include"),
            "quantize kernel must not use #include (NVRTC-incompatible)"
        );
        assert!(QUANTIZE_KERNEL.contains("extern \"C\" __global__ void quantize_fp8_kernel"));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_fp8_support_detection() {
        // This test requires GPU
        if let Ok(device) = GpuDevice::new() {
            let device_arc = Arc::new(device);
            match FP8TensorCore::new(device_arc.clone()) {
                Ok(fp8_core) => {
                    let (major, minor) = fp8_core.compute_capability();
                    println!("GPU Compute Capability: {}.{}", major, minor);

                    if supports_fp8_hardware(major, minor) {
                        assert!(
                            fp8_core.is_fp8_supported(),
                            "FP8 should be supported on compute capability {}.{}",
                            major,
                            minor
                        );
                        println!("✅ FP8 tensor cores supported!");
                    } else {
                        assert!(
                            !fp8_core.is_fp8_supported(),
                            "FP8 should not be supported on compute capability {}.{}",
                            major,
                            minor
                        );
                        println!("❌ FP8 tensor cores not supported (need 8.9+)");
                    }
                }
                Err(FP8Error::UnsupportedHardware(msg)) => {
                    println!("⚠️ FP8 not supported (hardware): {}", msg);
                }
                Err(FP8Error::ModuleLoadFailed(msg)) => {
                    println!("⚠️ FP8 not available (kernels failed to load): {}", msg);
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        } else {
            println!("⚠️ GPU not available, skipping FP8 support test");
        }
    }
}
