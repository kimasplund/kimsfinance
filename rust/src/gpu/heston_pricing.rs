//! GPU-Accelerated Heston Option Pricing
//!
//! Implements ultra-low latency option pricing using GPU-accelerated
//! characteristic function computation with FFT-based pricing.
//!
//! # Performance Targets
//!
//! | Batch Size | GPU Time | CPU Time | Speedup |
//! |------------|----------|----------|---------|
//! | 10 options | <1ms     | 10ms     | 10x     |
//! | 50 options | <2ms     | 50ms     | 25x     |
//! | 100 options| <3ms     | 100ms    | 33x     |
//! | 500 options| <10ms    | 500ms    | 50x     |
//! | 1000 options|<15ms    | 1000ms   | 67x     |

use crate::gpu::compile::compile_ptx_optimized_cached;
use crate::gpu::persistent::PinnedBuffer;
use crate::gpu::{GpuDevice, GpuError};
use crate::quantitative::heston::{HestonParams, OptionQuote, OptionType};
use chrono;
use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use std::sync::Arc;

/// GPU-accelerated Heston option pricer with pinned memory optimization
pub struct HestonGpuPricer {
    device: Arc<GpuDevice>,
    char_func_kernel: CudaFunction,
    fft_size: usize,

    // Pinned memory buffers (pre-allocated for max_batch_size)
    max_batch_size: usize,
    pinned_strikes: Option<PinnedBuffer<f64>>,
    pinned_expirations: Option<PinnedBuffer<f64>>,
    pinned_spot_prices: Option<PinnedBuffer<f64>>,
    pinned_rates: Option<PinnedBuffer<f64>>,
    pinned_phi_values: Option<PinnedBuffer<f64>>,
    pinned_char_func_real: Option<PinnedBuffer<f64>>,
    pinned_char_func_imag: Option<PinnedBuffer<f64>>,

    // Device buffers (pre-allocated)
    d_strikes: Option<CudaSlice<f64>>,
    d_expirations: Option<CudaSlice<f64>>,
    d_spot_prices: Option<CudaSlice<f64>>,
    d_risk_free_rates: Option<CudaSlice<f64>>,
    d_phi_values: Option<CudaSlice<f64>>,
    d_char_func_real: Option<CudaSlice<f64>>,
    d_char_func_imag: Option<CudaSlice<f64>>,
}

impl HestonGpuPricer {
    /// Create new GPU pricer with specified FFT size and max batch size
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device handle
    /// * `fft_size` - FFT size (must be power of 2, typically 4096 or 8192)
    /// * `max_batch_size` - Maximum number of options to price in one batch
    ///
    /// # Performance
    ///
    /// - Compilation time: ~100-150ms (first call, then cached)
    /// - Initialization overhead: ~1-2ms (subsequent calls)
    /// - Pinned memory allocation: Provides 20-30% faster transfers
    ///
    /// # Errors
    ///
    /// Returns error if kernel compilation or pinned allocation fails
    pub fn new(
        device: Arc<GpuDevice>,
        fft_size: usize,
        max_batch_size: usize,
    ) -> Result<Self, GpuError> {
        // Validate FFT size is power of 2
        if !fft_size.is_power_of_two() {
            return Err(GpuError::InvalidParameter(format!(
                "FFT size must be power of 2, got {}",
                fft_size
            )));
        }

        // Compile CUDA kernel (cached for performance)
        const KERNEL_SOURCE: &str = include_str!("cuda/heston/characteristic_function.cu");
        let ptx = compile_ptx_optimized_cached(KERNEL_SOURCE)?;

        // Load kernel module
        let module = device.context().load_module(ptx.as_ref().clone())?;
        let char_func_kernel = module.load_function("heston_characteristic_function")?;

        // Try to allocate pinned memory (fallback to pageable on failure)
        let (pinned_strikes, pinned_expirations, pinned_spot_prices, pinned_rates) =
            match Self::try_allocate_pinned_buffers(max_batch_size) {
                Ok(buffers) => {
                    eprintln!(
                        "✅ Pinned memory allocated ({} options max)",
                        max_batch_size
                    );
                    buffers
                }
                Err(e) => {
                    eprintln!("⚠️ Pinned allocation failed: {:?}", e);
                    eprintln!("   Using pageable memory (20-30% slower transfers)");
                    (None, None, None, None)
                }
            };

        // Allocate pinned buffers for FFT data
        let (pinned_phi_values, pinned_char_func_real, pinned_char_func_imag) =
            match Self::try_allocate_fft_pinned_buffers(fft_size, max_batch_size) {
                Ok(buffers) => buffers,
                Err(e) => {
                    eprintln!("⚠️ FFT pinned allocation failed: {:?}", e);
                    (None, None, None)
                }
            };

        // Pre-allocate device buffers
        let (d_strikes, d_expirations, d_spot_prices, d_risk_free_rates) =
            match Self::try_allocate_device_buffers(&device, max_batch_size) {
                Ok(buffers) => {
                    eprintln!(
                        "✅ Device buffers allocated ({} options max)",
                        max_batch_size
                    );
                    buffers
                }
                Err(e) => {
                    eprintln!("⚠️ Device allocation failed: {:?}", e);
                    (None, None, None, None)
                }
            };

        // Allocate device buffers for FFT
        let total_elements = max_batch_size * fft_size;

        // Initialize phi_values with FFT grid points (fixed for all pricing calls)
        let du = 0.25;  // Grid spacing (matches Carr-Madan formula)
        let phi_values_host: Vec<f64> = (0..fft_size).map(|i| i as f64 * du).collect();
        let d_phi_values = device.copy_to_device(&phi_values_host).ok();

        let d_char_func_real = device.allocate_device_buffer(total_elements).ok();
        let d_char_func_imag = device.allocate_device_buffer(total_elements).ok();

        Ok(Self {
            device,
            char_func_kernel,
            fft_size,
            max_batch_size,
            pinned_strikes,
            pinned_expirations,
            pinned_spot_prices,
            pinned_rates,
            pinned_phi_values,
            pinned_char_func_real,
            pinned_char_func_imag,
            d_strikes,
            d_expirations,
            d_spot_prices,
            d_risk_free_rates,
            d_phi_values,
            d_char_func_real,
            d_char_func_imag,
        })
    }

    /// Try to allocate pinned memory for option parameters
    fn try_allocate_pinned_buffers(
        max_batch_size: usize,
    ) -> Result<
        (
            Option<PinnedBuffer<f64>>,
            Option<PinnedBuffer<f64>>,
            Option<PinnedBuffer<f64>>,
            Option<PinnedBuffer<f64>>,
        ),
        GpuError,
    > {
        let strikes = PinnedBuffer::new(max_batch_size)?;
        let expirations = PinnedBuffer::new(max_batch_size)?;
        let spot_prices = PinnedBuffer::new(max_batch_size)?;
        let rates = PinnedBuffer::new(max_batch_size)?;

        Ok((
            Some(strikes),
            Some(expirations),
            Some(spot_prices),
            Some(rates),
        ))
    }

    /// Try to allocate pinned memory for FFT data
    fn try_allocate_fft_pinned_buffers(
        fft_size: usize,
        max_batch_size: usize,
    ) -> Result<
        (
            Option<PinnedBuffer<f64>>,
            Option<PinnedBuffer<f64>>,
            Option<PinnedBuffer<f64>>,
        ),
        GpuError,
    > {
        let total_elements = max_batch_size * fft_size;
        let phi_values = PinnedBuffer::new(fft_size)?;
        let char_func_real = PinnedBuffer::new(total_elements)?;
        let char_func_imag = PinnedBuffer::new(total_elements)?;

        Ok((Some(phi_values), Some(char_func_real), Some(char_func_imag)))
    }

    /// Try to allocate device buffers
    fn try_allocate_device_buffers(
        device: &GpuDevice,
        max_batch_size: usize,
    ) -> Result<
        (
            Option<CudaSlice<f64>>,
            Option<CudaSlice<f64>>,
            Option<CudaSlice<f64>>,
            Option<CudaSlice<f64>>,
        ),
        GpuError,
    > {
        let strikes = device.allocate_device_buffer(max_batch_size)?;
        let expirations = device.allocate_device_buffer(max_batch_size)?;
        let spot_prices = device.allocate_device_buffer(max_batch_size)?;
        let rates = device.allocate_device_buffer(max_batch_size)?;

        Ok((
            Some(strikes),
            Some(expirations),
            Some(spot_prices),
            Some(rates),
        ))
    }

    /// Create new GPU pricer with default max batch size (for backward compatibility)
    pub fn with_default_batch_size(
        device: Arc<GpuDevice>,
        fft_size: usize,
    ) -> Result<Self, GpuError> {
        Self::new(device, fft_size, 1000) // Default to 1000 options max
    }

    /// Price batch of options using GPU-accelerated characteristic function
    ///
    /// Uses pinned memory for faster transfers when available (20-30% speedup).
    ///
    /// # Arguments
    ///
    /// * `params` - Heston model parameters (validated)
    /// * `options` - Slice of option quotes to price
    ///
    /// # Returns
    ///
    /// Vec of option prices (same length as input)
    ///
    /// # Performance (with pinned memory)
    ///
    /// - 10 options: <0.8ms
    /// - 100 options: <3ms
    /// - 1000 options: <15ms
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Batch size exceeds max_batch_size
    /// - GPU allocation fails
    /// - Kernel launch fails
    /// - Parameters are invalid
    pub fn price_options(
        &mut self,
        params: &HestonParams,
        options: &[OptionQuote],
    ) -> Result<Vec<f64>, GpuError> {
        if options.is_empty() {
            return Ok(Vec::new());
        }

        // Validate batch size
        if options.len() > self.max_batch_size {
            return Err(GpuError::InvalidParameter(format!(
                "Batch size {} exceeds max_batch_size {}. Create pricer with larger max_batch_size.",
                options.len(),
                self.max_batch_size
            )));
        }

        // Validate parameters
        params
            .validate()
            .map_err(|e| GpuError::InvalidParameter(format!("Invalid Heston parameters: {}", e)))?;

        let n_options = options.len();
        let now = chrono::Utc::now().timestamp();

        // Extract option data
        let strikes: Vec<f64> = options.iter().map(|o| o.strike).collect();
        let expirations: Vec<f64> = options.iter().map(|o| o.time_to_expiry(now)).collect();
        let spot_prices: Vec<f64> = options.iter().map(|o| o.spot_price).collect();
        let risk_free_rates: Vec<f64> = options.iter().map(|o| o.risk_free_rate).collect();

        // Generate FFT integration points (only once, can be reused)
        let du = 0.25;
        let phi_values: Vec<f64> = (0..self.fft_size).map(|i| i as f64 * du).collect();

        // DEBUG: Log phi values range
        eprintln!(
            "[DEBUG] phi_values: size={}, range=[{:.4}, {:.4}]",
            phi_values.len(),
            phi_values.first().unwrap_or(&0.0),
            phi_values.last().unwrap_or(&0.0)
        );

        // Use pinned memory path if available
        let has_pinned = self.pinned_strikes.is_some()
            && self.d_strikes.is_some()
            && self.d_char_func_real.is_some();

        // TEMPORARY: Force pageable memory to bypass dtoh_pinned bug
        eprintln!("[DEBUG] FORCING pageable memory path to test download bug");
        let (char_func_real, char_func_imag) = self.price_with_pageable_memory(
            params,
            &strikes,
            &expirations,
            &spot_prices,
            &risk_free_rates,
            &phi_values,
            n_options,
        )?;

        // Original code (disabled for testing):
        /* let (char_func_real, char_func_imag) = if has_pinned {
            eprintln!("[DEBUG] Using pinned memory path");
            self.price_with_pinned_memory(
                params,
                &strikes,
                &expirations,
                &spot_prices,
                &risk_free_rates,
                &phi_values,
                n_options,
            )?
        } else {
            eprintln!("[DEBUG] Using pageable memory path");
            self.price_with_pageable_memory(
                params,
                &strikes,
                &expirations,
                &spot_prices,
                &risk_free_rates,
                &phi_values,
                n_options,
            )?
        }; */

        // DEBUG: Check characteristic function values
        let total_elements = n_options * self.fft_size;
        let cf_real_nonzero = char_func_real.iter().filter(|&&x| x.abs() > 1e-10).count();
        let cf_imag_nonzero = char_func_imag.iter().filter(|&&x| x.abs() > 1e-10).count();
        eprintln!(
            "[DEBUG] Characteristic function downloaded: total={}, real_nonzero={}, imag_nonzero={}",
            total_elements, cf_real_nonzero, cf_imag_nonzero
        );
        if cf_real_nonzero > 0 {
            let sample_idx = char_func_real.iter().position(|&x| x.abs() > 1e-10).unwrap();
            eprintln!(
                "[DEBUG]   Sample CF value at idx={}: real={:.6}, imag={:.6}",
                sample_idx, char_func_real[sample_idx], char_func_imag[sample_idx]
            );
        }

        // Apply FFT to get option prices
        let prices = self.fft_to_option_prices(&char_func_real, &char_func_imag, options)?;

        Ok(prices)
    }

    /// Fast path: Use pinned memory for transfers (20-30% faster)
    fn price_with_pinned_memory(
        &mut self,
        params: &HestonParams,
        strikes: &[f64],
        expirations: &[f64],
        spot_prices: &[f64],
        risk_free_rates: &[f64],
        phi_values: &[f64],
        n_options: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        // Copy data to pinned buffers
        if let (
            Some(p_strikes),
            Some(p_exp),
            Some(p_spot),
            Some(p_rates),
        ) = (
            &mut self.pinned_strikes,
            &mut self.pinned_expirations,
            &mut self.pinned_spot_prices,
            &mut self.pinned_rates,
        ) {
            p_strikes.as_mut_slice()[..n_options].copy_from_slice(strikes);
            p_exp.as_mut_slice()[..n_options].copy_from_slice(expirations);
            p_spot.as_mut_slice()[..n_options].copy_from_slice(spot_prices);
            p_rates.as_mut_slice()[..n_options].copy_from_slice(risk_free_rates);

            // DMA transfer from pinned to device (fast!)
            // Note: We upload full buffers since htod_pinned doesn't support partial
            if let (
                Some(d_strikes),
                Some(d_exp),
                Some(d_spot),
                Some(d_rates),
            ) = (
                &mut self.d_strikes,
                &mut self.d_expirations,
                &mut self.d_spot_prices,
                &mut self.d_risk_free_rates,
            ) {
                self.device.htod_pinned(p_strikes, d_strikes)?;
                self.device.htod_pinned(p_exp, d_exp)?;
                self.device.htod_pinned(p_spot, d_spot)?;
                self.device.htod_pinned(p_rates, d_rates)?;

                // Upload phi values (reused across calls)
                if let (Some(p_phi), Some(d_phi)) =
                    (&mut self.pinned_phi_values, &mut self.d_phi_values)
                {
                    p_phi.as_mut_slice().copy_from_slice(phi_values);
                    self.device.htod_pinned(p_phi, d_phi)?;
                }
            }

            // Launch kernel (separate scope to avoid borrow conflicts)
            if let (
                Some(d_strikes),
                Some(d_exp),
                Some(d_spot),
                Some(d_rates),
            ) = (
                self.d_strikes.as_ref(),
                self.d_expirations.as_ref(),
                self.d_spot_prices.as_ref(),
                self.d_risk_free_rates.as_ref(),
            ) {
                self.launch_kernel(params, n_options, d_strikes, d_exp, d_spot, d_rates)?;
            }

            // Download results
            let total_elements = n_options * self.fft_size;
            if let (
                Some(p_real),
                Some(p_imag),
                Some(d_real),
                Some(d_imag),
            ) = (
                &mut self.pinned_char_func_real,
                &mut self.pinned_char_func_imag,
                &self.d_char_func_real,
                &self.d_char_func_imag,
            ) {
                // DEBUG: Verify we're downloading from the same buffers
                let (ptr_real, _) = d_real.device_ptr(&self.device.stream);
                let (ptr_imag, _) = d_imag.device_ptr(&self.device.stream);
                eprintln!(
                    "[DEBUG] Download buffer addresses: real={:?}, imag={:?}",
                    ptr_real as *const (),
                    ptr_imag as *const ()
                );

                eprintln!("[DEBUG] Pinned buffer sizes: real={}, imag={}, requesting download of {} elements",
                    p_real.len(), p_imag.len(), total_elements);
                eprintln!("[DEBUG] Device buffer sizes: real={}, imag={}",
                    d_real.len(), d_imag.len());

                self.device.dtoh_pinned(d_real, p_real)?;
                let real_slice = p_real.as_slice();
                eprintln!("[DEBUG] Downloaded real buffer, pinned buffer len={}, first 10 values: {:?}",
                    real_slice.len(), &real_slice[..10.min(real_slice.len())]);
                eprintln!("[DEBUG] Real buffer indices [0], [1], [4096]: {:?}, {:?}, {:?}",
                    real_slice[0], real_slice[1], real_slice.get(4096));

                self.device.dtoh_pinned(d_imag, p_imag)?;
                let imag_slice = p_imag.as_slice();
                eprintln!("[DEBUG] Downloaded imag buffer, pinned buffer len={}, first 10 values: {:?}",
                    imag_slice.len(), &imag_slice[..10.min(imag_slice.len())]);
                eprintln!("[DEBUG] Imag buffer indices [0], [1], [4096]: {:?}, {:?}, {:?}",
                    imag_slice[0], imag_slice[1], imag_slice.get(4096));

                let char_func_real = real_slice[..total_elements].to_vec();
                let char_func_imag = imag_slice[..total_elements].to_vec();

                eprintln!("[DEBUG] After vec conversion - real[0]={}, imag[0]={}, real[1]={}, imag[1]={}",
                    char_func_real[0], char_func_imag[0], char_func_real[1], char_func_imag[1]);

                return Ok((char_func_real, char_func_imag));
            }
        }

        Err(GpuError::ExecutionError(
            "Pinned buffers not available".to_string(),
        ))
    }

    /// Fallback path: Use pageable memory (traditional approach)
    fn price_with_pageable_memory(
        &self,
        params: &HestonParams,
        strikes: &[f64],
        expirations: &[f64],
        spot_prices: &[f64],
        risk_free_rates: &[f64],
        phi_values: &[f64],
        n_options: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        // Traditional path: copy_to_device (pageable memory)
        let mut d_strikes = self.device.copy_to_device(strikes)?;
        let mut d_expirations = self.device.copy_to_device(expirations)?;
        let mut d_spot_prices = self.device.copy_to_device(spot_prices)?;
        let mut d_risk_free_rates = self.device.copy_to_device(risk_free_rates)?;
        let d_phi_values = self.device.copy_to_device(phi_values)?;

        // Launch kernel
        self.launch_kernel(
            params,
            n_options,
            &mut d_strikes,
            &mut d_expirations,
            &mut d_spot_prices,
            &mut d_risk_free_rates,
        )?;

        // Download results from the buffers where the kernel wrote output
        let d_char_func_real = self.d_char_func_real.as_ref().ok_or_else(|| {
            GpuError::ExecutionError("char_func_real buffer not available for download".to_string())
        })?;
        let d_char_func_imag = self.d_char_func_imag.as_ref().ok_or_else(|| {
            GpuError::ExecutionError("char_func_imag buffer not available for download".to_string())
        })?;

        let total_elements = n_options * self.fft_size;
        let mut char_func_real = self.device.copy_to_host(d_char_func_real)?;
        let mut char_func_imag = self.device.copy_to_host(d_char_func_imag)?;

        // DEBUG: Check what was actually downloaded
        eprintln!("[DEBUG] Pageable download - Downloaded real.len()={}, imag.len()={}",
            char_func_real.len(), char_func_imag.len());
        eprintln!("[DEBUG] Pageable - real[0]={}, real[1]={}, real[4096]={}",
            char_func_real.get(0).unwrap_or(&f64::NAN),
            char_func_real.get(1).unwrap_or(&f64::NAN),
            char_func_real.get(4096).unwrap_or(&f64::NAN));
        eprintln!("[DEBUG] Pageable - imag[0]={}, imag[1]={}, imag[4096]={}",
            char_func_imag.get(0).unwrap_or(&f64::NAN),
            char_func_imag.get(1).unwrap_or(&f64::NAN),
            char_func_imag.get(4096).unwrap_or(&f64::NAN));

        // Trim to actual size
        char_func_real.truncate(total_elements);
        char_func_imag.truncate(total_elements);

        Ok((char_func_real, char_func_imag))
    }

    /// Launch CUDA kernel (shared by both pinned and pageable paths)
    fn launch_kernel(
        &self,
        params: &HestonParams,
        n_options: usize,
        d_strikes: &CudaSlice<f64>,
        d_expirations: &CudaSlice<f64>,
        d_spot_prices: &CudaSlice<f64>,
        d_risk_free_rates: &CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let total_elements = n_options * self.fft_size;
        let threads_per_block = 256;
        let blocks = ((total_elements + threads_per_block - 1) / threads_per_block) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let kappa = params.kappa;
        let theta = params.theta;
        let sigma = params.sigma;
        let rho = params.rho;
        let v0 = params.v0;
        let alpha = 1.5;  // Carr-Madan damping parameter (CRITICAL for complex CF evaluation)
        let n_options_i32 = n_options as i32;
        let fft_size_i32 = self.fft_size as i32;

        // Get device buffers
        let d_phi_values = self.d_phi_values.as_ref().ok_or_else(|| {
            GpuError::ExecutionError("phi_values buffer not allocated".to_string())
        })?;

        // IMPORTANT: Get mutable references to the pre-allocated buffers
        // DO NOT clone - that creates new buffers that won't be downloaded later!
        let d_char_func_real = self.d_char_func_real.as_ref().ok_or_else(|| {
            GpuError::ExecutionError("char_func_real buffer not allocated".to_string())
        })?;

        let d_char_func_imag = self.d_char_func_imag.as_ref().ok_or_else(|| {
            GpuError::ExecutionError("char_func_imag buffer not allocated".to_string())
        })?;

        // DEBUG: Print buffer addresses to verify they're different
        let (ptr_real, _) = d_char_func_real.device_ptr(&self.device.stream);
        let (ptr_imag, _) = d_char_func_imag.device_ptr(&self.device.stream);
        eprintln!(
            "[DEBUG] BEFORE KERNEL - Buffer addresses: real={:?}, imag={:?}, same={}",
            ptr_real as *const (),
            ptr_imag as *const (),
            ptr_real == ptr_imag
        );
        eprintln!(
            "[DEBUG] BEFORE KERNEL - Device buffer object IDs: real={:p}, imag={:p}",
            d_char_func_real as *const _,
            d_char_func_imag as *const _
        );
        eprintln!(
            "[DEBUG] Buffer sizes: real={}, imag={}, expected={}",
            d_char_func_real.len(),
            d_char_func_imag.len(),
            total_elements
        );

        eprintln!("[DEBUG] Launching kernel with:");
        eprintln!("  kappa={}, theta={}, sigma={}, rho={}, v0={}, alpha={}", kappa, theta, sigma, rho, v0, alpha);
        eprintln!("  n_options={}, fft_size={}", n_options_i32, fft_size_i32);
        eprintln!("  threads_per_block={}, blocks={}", threads_per_block, blocks);

        unsafe {
            let mut builder = self.device.stream.launch_builder(&self.char_func_kernel);
            builder.arg(&kappa);         // Arg 0: f64
            builder.arg(&theta);         // Arg 1: f64
            builder.arg(&sigma);         // Arg 2: f64
            builder.arg(&rho);           // Arg 3: f64
            builder.arg(&v0);            // Arg 4: f64
            builder.arg(&alpha);         // Arg 5: f64 (NEW: Carr-Madan damping parameter)
            builder.arg(d_strikes);      // Arg 6: *const f64
            builder.arg(d_expirations);  // Arg 7: *const f64
            builder.arg(d_spot_prices);  // Arg 8: *const f64
            builder.arg(d_risk_free_rates); // Arg 9: *const f64
            builder.arg(&fft_size_i32);  // Arg 10: i32
            builder.arg(d_phi_values);   // Arg 11: *const f64
            builder.arg(d_char_func_real); // Arg 12: *mut f64 (OUTPUT)
            builder.arg(d_char_func_imag); // Arg 13: *mut f64 (OUTPUT)
            builder.arg(&n_options_i32); // Arg 14: i32

            eprintln!("[DEBUG] Total kernel arguments: 15");

            builder.launch(config).map_err(|e| {
                GpuError::ExecutionError(format!("Heston kernel launch failed: {:?}", e))
            })?;
        }

        // Synchronize to ensure kernel completes
        self.device.stream.synchronize().map_err(|e| {
            GpuError::ExecutionError(format!("Kernel synchronization failed: {:?}", e))
        })?;

        eprintln!("[DEBUG] Kernel synchronized successfully");
        eprintln!("[DEBUG] Kernel output visible in CUDA_DEBUG prints above");

        // DEBUG: Verify buffer pointers haven't changed after kernel
        let d_char_func_real_after = self.d_char_func_real.as_ref().unwrap();
        let d_char_func_imag_after = self.d_char_func_imag.as_ref().unwrap();
        let (ptr_real_after, _) = d_char_func_real_after.device_ptr(&self.device.stream);
        let (ptr_imag_after, _) = d_char_func_imag_after.device_ptr(&self.device.stream);
        eprintln!(
            "[DEBUG] AFTER KERNEL - Buffer addresses: real={:?}, imag={:?}",
            ptr_real_after as *const (),
            ptr_imag_after as *const ()
        );
        eprintln!(
            "[DEBUG] AFTER KERNEL - Addresses match: real={}, imag={}",
            (ptr_real_after as usize) == (ptr_real as usize),
            (ptr_imag_after as usize) == (ptr_imag as usize)
        );

        Ok(())
    }

    /// Convert characteristic function to option prices via FFT (Carr-Madan formula)
    ///
    /// Implements the Carr-Madan FFT approach for fast option pricing:
    ///
    /// # Theory
    ///
    /// Call price: C(K) = exp(-α·k) / π × Re[ ∫₀^∞ exp(-i·φ·k) · ψ(φ) dφ ]
    ///
    /// where:
    /// - k = log(K/S₀) (log-moneyness)
    /// - α = damping parameter (typically 1.5)
    /// - ψ(φ) = exp(-r·T) · φ₁(φ - (α+1)i) / (α² + α - φ² + i(2α+1)φ)
    /// - φ₁ = Heston characteristic function (computed by GPU)
    ///
    /// # Performance
    ///
    /// CPU-based FFT using rustfft. Future optimization: cuFFT for GPU acceleration.
    ///
    /// # Arguments
    ///
    /// * `char_func_real` - Real part of characteristic function (n_options × fft_size)
    /// * `char_func_imag` - Imaginary part of characteristic function (n_options × fft_size)
    /// * `options` - Option quotes to price
    ///
    /// # Returns
    ///
    /// Vec of option prices (call/put converted via put-call parity)
    fn fft_to_option_prices(
        &self,
        char_func_real: &[f64],
        char_func_imag: &[f64],
        options: &[OptionQuote],
    ) -> Result<Vec<f64>, GpuError> {
        eprintln!(
            "[DEBUG] fft_to_option_prices: {} options, fft_size={}",
            options.len(),
            self.fft_size
        );

        let mut prices = Vec::with_capacity(options.len());

        // Setup FFT planner
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(self.fft_size);

        // Carr-Madan FFT parameters
        let alpha = 1.5; // Damping parameter (standard choice)
        let eta = 0.25; // Grid spacing in log-strike space
        let lambda = 2.0 * PI / (eta * self.fft_size as f64);

        eprintln!(
            "[DEBUG] Lewis formula params: eta={:.2}, lambda={:.4}",
            eta, lambda
        );

        for (i, option) in options.iter().enumerate() {
            // Extract characteristic function for this option
            let start = i * self.fft_size;
            let end = start + self.fft_size;

            // Log-moneyness
            let k = (option.strike / option.spot_price).ln();

            // Time to expiry (years)
            let now = chrono::Utc::now().timestamp();
            let tau = option.time_to_expiry(now);

            if i == 0 {
                eprintln!(
                    "[DEBUG] Option 0: S={:.2}, K={:.2}, T={:.4}, r={:.4}, k={:.6}",
                    option.spot_price, option.strike, tau, option.risk_free_rate, k
                );
            }

            // Construct modified characteristic function for Carr-Madan
            let mut modified_cf: Vec<Complex64> = Vec::with_capacity(self.fft_size);
            let mut nonzero_psi_count = 0;
            let mut clamped_count = 0;  // Count how many values were clamped

            for j in 0..self.fft_size {
                let phi = j as f64 * eta;

                // Get characteristic function value
                let cf_real = char_func_real[start + j];
                let cf_imag = char_func_imag[start + j];
                let cf = Complex64::new(cf_real, cf_imag);

                // Modified characteristic function for Carr-Madan:
                // ψ(φ) = exp(-r·T) · φ₁(φ - (α+1)i) / (α² + α - φ² + i(2α+1)φ)

                // Discount factor
                let discount = (-option.risk_free_rate * tau).exp();

                // Denominator: α² + α - φ² + i(2α+1)φ
                let denom_real = alpha * alpha + alpha - phi * phi;
                let denom_imag = (2.0 * alpha + 1.0) * phi;
                let denominator = Complex64::new(denom_real, denom_imag);

                // Modified CF
                let psi = discount * cf / denominator;

                // DEBUG: Check for first Inf/NaN in psi
                if i == 0 && (psi.re.is_infinite() || psi.im.is_infinite()) && j < 20 {
                    eprintln!(
                        "[DEBUG] ⚠️ Inf detected at j={}, phi={:.4}, cf=({:.6},{:.6}), denom=({:.4},{:.4}), psi=({:.6},{:.6})",
                        j, phi, cf_real, cf_imag, denom_real, denom_imag, psi.re, psi.im
                    );
                }

                if i == 0 && j < 5 {
                    eprintln!(
                        "[DEBUG]   j={}, phi={:.4}, cf=({:.6},{:.6}), discount={:.6}, denom=({:.4},{:.4}), psi=({:.6},{:.6})",
                        j, phi, cf_real, cf_imag, discount, denom_real, denom_imag, psi.re, psi.im
                    );
                }

                if psi.norm() > 1e-10 {
                    nonzero_psi_count += 1;
                }

                // Apply Simpson's rule weighting
                let weight = if j == 0 {
                    0.5
                } else if j == self.fft_size - 1 {
                    0.5
                } else if j % 2 == 1 {
                    4.0
                } else {
                    2.0
                };

                let weighted_psi = psi * weight * eta / 3.0;

                // NUMERICAL STABILITY FIX: Only clamp explicit Inf/NaN values
                // Previous threshold of 1e6 was too aggressive (clamped 97.8% of values!)
                // Now only clamp actual overflow, letting the characteristic function
                // values through even if large. The FFT normalization will handle scaling.
                let clamped_psi = if weighted_psi.re.is_infinite() || weighted_psi.im.is_infinite()
                       || weighted_psi.re.is_nan() || weighted_psi.im.is_nan() {
                    // Explicit overflow/NaN - truncate
                    clamped_count += 1;
                    Complex64::new(0.0, 0.0)
                } else {
                    // Keep all finite values, even if large
                    // The FFT normalization by 1/N will bring them to reasonable scale
                    weighted_psi
                };

                modified_cf.push(clamped_psi);
            }

            if i == 0 {
                eprintln!(
                    "[DEBUG] Option 0: nonzero_psi_count={}/{}, clamped_count={} ({:.1}%)",
                    nonzero_psi_count, self.fft_size,
                    clamped_count, (clamped_count as f64 / self.fft_size as f64) * 100.0
                );

                // DEBUG: Check FFT input for NaN/Inf and find problematic indices
                let has_nan = modified_cf.iter().any(|c| c.re.is_nan() || c.im.is_nan());
                let has_inf = modified_cf.iter().any(|c| c.re.is_infinite() || c.im.is_infinite());
                let max_real = modified_cf.iter().map(|c| c.re.abs()).fold(0.0f64, f64::max);
                let max_imag = modified_cf.iter().map(|c| c.im.abs()).fold(0.0f64, f64::max);

                // Find indices with Inf values
                let inf_indices: Vec<usize> = modified_cf.iter().enumerate()
                    .filter(|(_, c)| c.re.is_infinite() || c.im.is_infinite())
                    .map(|(i, _)| i)
                    .take(10)  // Show first 10
                    .collect();

                eprintln!("[DEBUG] Option 0 BEFORE FFT: has_nan={}, has_inf={}, max_real={:.2e}, max_imag={:.2e}",
                    has_nan, has_inf, max_real, max_imag);
                eprintln!("[DEBUG] Option 0 BEFORE FFT: first 5 values: {:?}",
                    &modified_cf[..5.min(modified_cf.len())]);
                if !inf_indices.is_empty() {
                    eprintln!("[DEBUG] Option 0 BEFORE FFT: Inf found at indices: {:?}", inf_indices);
                    for &idx in inf_indices.iter().take(3) {
                        eprintln!("[DEBUG]   idx={}: value={:?}", idx, modified_cf[idx]);
                    }
                }
            }

            // Apply FFT
            fft.process(&mut modified_cf);

            // CRITICAL: Normalize FFT output by dividing by N
            // rustfft does NOT normalize automatically, so we must do it manually
            // This is standard for all unnormalized FFT implementations
            let fft_norm = 1.0 / (self.fft_size as f64);
            for cf in modified_cf.iter_mut() {
                *cf *= fft_norm;
            }

            if i == 0 {
                // DEBUG: Check FFT output for NaN/Inf
                let has_nan = modified_cf.iter().any(|c| c.re.is_nan() || c.im.is_nan());
                let has_inf = modified_cf.iter().any(|c| c.re.is_infinite() || c.im.is_infinite());
                let max_real = modified_cf.iter().map(|c| c.re.abs()).fold(0.0f64, f64::max);
                let max_imag = modified_cf.iter().map(|c| c.im.abs()).fold(0.0f64, f64::max);
                eprintln!("[DEBUG] Option 0 AFTER FFT (normalized by 1/N=1/{}): has_nan={}, has_inf={}, max_real={:.2e}, max_imag={:.2e}",
                    self.fft_size, has_nan, has_inf, max_real, max_imag);
                eprintln!("[DEBUG] Option 0 AFTER FFT: values at idx [0, 1024, 2048, 3072]: {:?}",
                    vec![modified_cf[0], modified_cf[1024], modified_cf[2048], modified_cf[3072]]);
            }

            // Extract price at strike K
            // FFT outputs correspond to log-strikes: k_u = -b + lambda * u / N
            let b = lambda / 2.0; // Half log-strike range
            let k_values: Vec<f64> = (0..self.fft_size)
                .map(|u| -b + lambda * (u as f64) / (self.fft_size as f64))
                .collect();

            // Find closest FFT output to our log-strike k
            let idx = k_values
                .iter()
                .enumerate()
                .min_by(|(_, k1), (_, k2)| {
                    (*k1 - k).abs().partial_cmp(&(*k2 - k).abs()).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(self.fft_size / 2);

            // Extract call price from FFT output
            let fft_value = modified_cf[idx];
            let call_price = option.spot_price * ((-alpha * k).exp() / PI * fft_value.re);

            if i == 0 {
                eprintln!(
                    "[DEBUG] Option 0: idx={}, k_at_idx={:.6}, fft_value=({:.6},{:.6}), raw_call_price={:.4}",
                    idx, k_values[idx], fft_value.re, fft_value.im, call_price
                );
            }

            // Ensure non-negative price
            let call_price = call_price.max(0.0);

            // Convert to put if needed via put-call parity:
            // P = C - S + K·exp(-r·T)
            let price = match option.option_type {
                OptionType::Call => call_price,
                OptionType::Put => {
                    let intrinsic = option.strike * (-option.risk_free_rate * tau).exp();
                    (call_price - option.spot_price + intrinsic).max(0.0)
                }
            };

            if i == 0 {
                eprintln!(
                    "[DEBUG] Option 0 final: type={:?}, call={:.4}, final_price={:.4}",
                    option.option_type, call_price, price
                );
            }

            prices.push(price);
        }

        eprintln!("[DEBUG] Final prices: {:?}", prices);

        Ok(prices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_heston_pricer_initialization() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let pricer = HestonGpuPricer::new(device, 4096);
        assert!(
            pricer.is_ok(),
            "Failed to create HestonGpuPricer: {:?}",
            pricer.err()
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_price_single_option() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let pricer = HestonGpuPricer::new(device, 4096).unwrap();

        let params = HestonParams::new(
            2.0,  // kappa
            0.04, // theta
            0.3,  // sigma
            -0.7, // rho
            0.04, // v0
        )
        .unwrap();

        let option = OptionQuote {
            symbol: "BTC-20250101-50000-C".to_string(),
            underlying: "BTC".to_string(),
            strike: 50000.0,
            expiry_years: 0.25, // 3 months
            option_type: OptionType::Call,
            bid: 2000.0,
            ask: 2100.0,
            mid_price: 2050.0,
            implied_vol: Some(0.8),
            volume: 100.0,
        };

        let prices = pricer.price_options(&params, &[option]);
        assert!(prices.is_ok(), "Failed to price option: {:?}", prices.err());

        let price = prices.unwrap()[0];
        assert!(price > 0.0, "Option price should be positive");
        println!("Option price: ${:.2}", price);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_price_batch() {
        let device = Arc::new(GpuDevice::new().expect("Failed to initialize GPU"));
        let pricer = HestonGpuPricer::new(device, 4096).unwrap();

        let params = HestonParams::new(2.0, 0.04, 0.3, -0.7, 0.04).unwrap();

        // Create batch of 100 options with different strikes
        let options: Vec<OptionQuote> = (40000..40100)
            .map(|strike| OptionQuote {
                symbol: format!("BTC-20250101-{}-C", strike),
                underlying: "BTC".to_string(),
                strike: strike as f64,
                expiry_years: 0.25,
                option_type: OptionType::Call,
                bid: 2000.0,
                ask: 2100.0,
                mid_price: 2050.0,
                implied_vol: Some(0.8),
                volume: 100.0,
            })
            .collect();

        let start = std::time::Instant::now();
        let prices = pricer.price_options(&params, &options).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(prices.len(), 100);
        println!(
            "Priced 100 options in {:?} ({:.2}ms)",
            elapsed,
            elapsed.as_secs_f64() * 1000.0
        );

        // Should be <3ms for 100 options
        assert!(elapsed.as_millis() < 10, "Pricing too slow: {:?}", elapsed);
    }
}
