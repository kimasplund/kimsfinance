//! Build script for kimsfinance_core
//!
//! Compiles the RSI fused kernel (CUB-based) at build time when the `gpu`
//! feature is enabled. CUB templates require nvcc; they cannot be compiled
//! with the runtime NVRTC pipeline.
//!
//! NOTE: Earlier versions also nvcc-compiled `src/gpu/kernels_fp8_wmma.cu`
//! and `src/gpu/kernels/fp8_cutlass.cu` to CUBINs and exported their paths
//! via `FP8_WMMA_CUBIN_PATH` / `FP8_CUTLASS_CUBIN_PATH`. No Rust code ever
//! loaded those CUBINs or read those env vars (the FP8 modules JIT-compile
//! NVRTC-compatible sources via `include_str!` instead), so those build
//! steps were dead work and have been removed.
//!
//! # Build Process
//!
//! 1. Detect if `gpu` feature is enabled (skip if not)
//! 2. Find nvcc compiler (warn if not found, skip kernel compilation)
//! 3. Detect CUDA toolkit path (CUDA_HOME, /usr/local/cuda-13.0, /usr/local/cuda)
//! 4. Compile RSI fused kernel to a shared library
//! 5. Emit cargo directives for rebuild detection
//!
//! # Environment Variables
//!
//! - `CUDA_HOME`: Override CUDA toolkit path
//! - `CUDA_ARCH`: Override target architecture (default: sm_89)
//!
//! # Example Usage
//!
//! ```bash
//! # Standard build (uses defaults)
//! cargo build --features gpu
//!
//! # Custom CUDA path
//! CUDA_HOME=/opt/cuda-13.0 cargo build --features gpu
//!
//! # Custom architecture (for different GPU)
//! CUDA_ARCH=sm_86 cargo build --features gpu
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Only compile CUDA kernels if `gpu` feature is enabled
    let gpu_enabled = env::var("CARGO_FEATURE_GPU").is_ok();

    if !gpu_enabled {
        println!("cargo:warning=GPU feature not enabled, skipping CUDA kernel compilation");
        return;
    }

    println!("cargo:warning=GPU feature enabled, attempting CUDA kernel compilation");

    // Detect nvcc compiler
    let nvcc = match find_nvcc() {
        Some(path) => {
            println!("cargo:warning=Found nvcc at: {}", path.display());
            path
        }
        None => {
            println!(
                "cargo:warning=nvcc not found in PATH. Skipping build-time CUDA kernel compilation."
            );
            println!(
                "cargo:warning=Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) to enable the RSI fused kernel."
            );
            return;
        }
    };

    // Detect CUDA toolkit path
    let cuda_home = match find_cuda_home() {
        Some(path) => {
            println!("cargo:warning=CUDA toolkit found at: {}", path.display());
            path
        }
        None => {
            println!("cargo:warning=CUDA toolkit not found. Set CUDA_HOME or install CUDA.");
            return;
        }
    };

    // Get target architecture (default: sm_89 for RTX 3500 Ada)
    // Can also auto-detect GPU if available
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| {
        // Try to auto-detect GPU architecture
        detect_cuda_architecture(&nvcc).unwrap_or_else(|| "sm_89".to_string())
    });
    println!(
        "cargo:warning=Compiling for CUDA architecture: {}",
        cuda_arch
    );

    // Compile RSI fused kernel with CUB (requires nvcc, not NVRTC)
    compile_rsi_fused_kernel(&nvcc, &cuda_home, &cuda_arch);

    // Emit rebuild directives
    emit_rebuild_directives();
}

/// Detect GPU architecture at build time
///
/// Uses nvidia-smi to query GPU compute capability.
/// Returns architecture string like "sm_89" or None if detection fails.
fn detect_cuda_architecture(_nvcc: &PathBuf) -> Option<String> {
    // Try nvidia-smi first (most reliable)
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if output.status.success() {
        let compute_cap = String::from_utf8(output.stdout).ok()?;
        let compute_cap = compute_cap.trim();

        // Convert "8.9" to "sm_89"
        let arch = compute_cap.replace('.', "");
        if !arch.is_empty() {
            println!(
                "cargo:warning=Auto-detected GPU architecture: sm_{} (compute cap {})",
                arch, compute_cap
            );
            return Some(format!("sm_{}", arch));
        }
    }

    // Fallback: try deviceQuery from CUDA samples (if available)
    // This is less reliable and might not be installed
    None
}

/// Find nvcc compiler in PATH
fn find_nvcc() -> Option<PathBuf> {
    // Try `which nvcc` on Unix
    #[cfg(unix)]
    {
        let output = Command::new("which").arg("nvcc").output().ok()?;
        if output.status.success() {
            let path_str = String::from_utf8(output.stdout).ok()?;
            return Some(PathBuf::from(path_str.trim()));
        }
    }

    // Try `where nvcc` on Windows
    #[cfg(windows)]
    {
        let output = Command::new("where").arg("nvcc.exe").output().ok()?;
        if output.status.success() {
            let path_str = String::from_utf8(output.stdout).ok()?;
            return Some(PathBuf::from(path_str.trim()));
        }
    }

    None
}

/// Find CUDA toolkit installation path
///
/// Tries in order:
/// 1. CUDA_HOME environment variable
/// 2. /usr/local/cuda-13.0 (CUDA 13.0 specific)
/// 3. /usr/local/cuda (default symlink)
/// 4. Detect from nvcc path
fn find_cuda_home() -> Option<PathBuf> {
    // Try CUDA_HOME environment variable
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let path = PathBuf::from(cuda_home);
        if path.exists() {
            return Some(path);
        }
    }

    // Try standard CUDA 13.0 installation
    let cuda_13 = PathBuf::from("/usr/local/cuda-13.0");
    if cuda_13.exists() {
        return Some(cuda_13);
    }

    // Try default CUDA symlink
    let cuda_default = PathBuf::from("/usr/local/cuda");
    if cuda_default.exists() {
        return Some(cuda_default);
    }

    // Try to detect from nvcc path (remove /bin/nvcc)
    if let Some(nvcc) = find_nvcc() {
        let parent = nvcc.parent()?;
        let cuda_root = parent.parent()?;
        if cuda_root.join("include").exists() {
            return Some(cuda_root.to_path_buf());
        }
    }

    None
}

/// Compile RSI fused kernel with CUB to shared library
///
/// This kernel uses CUB DeviceScan for parallel Wilder's smoothing, which requires
/// template instantiation at compile time (cannot be done with NVRTC).
///
/// Compilation command:
/// ```bash
/// nvcc -shared \
///      -arch=sm_89 \
///      -std=c++17 \
///      -I{cuda}/include \
///      -O3 \
///      -use_fast_math \
///      --expt-relaxed-constexpr \
///      -Xcompiler -fPIC \
///      -o {out_dir}/librsi_fused.so \
///      src/gpu/kernels/rsi_fused.cu
/// ```
fn compile_rsi_fused_kernel(nvcc: &Path, cuda_home: &Path, cuda_arch: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let kernel_source = PathBuf::from("src/gpu/kernels/rsi_fused.cu");

    // Use .so for Linux, .dylib for macOS, .dll for Windows
    #[cfg(target_os = "linux")]
    let output_lib = out_dir.join("librsi_fused.so");
    #[cfg(target_os = "macos")]
    let output_lib = out_dir.join("librsi_fused.dylib");
    #[cfg(target_os = "windows")]
    let output_lib = out_dir.join("rsi_fused.dll");

    if !kernel_source.exists() {
        println!(
            "cargo:warning=RSI fused kernel source not found: {}",
            kernel_source.display()
        );
        return;
    }

    println!(
        "cargo:warning=Compiling RSI fused kernel with CUB: {}",
        kernel_source.display()
    );

    // Build include paths
    let cuda_include = cuda_home.join("include");
    let cuda_targets_include = cuda_home.join("targets/x86_64-linux/include");

    // Build nvcc command
    // Note: Use --diag-suppress to ignore rsqrt exception spec warning
    let mut cmd = Command::new(nvcc);
    cmd.arg("-shared") // Compile to shared library
        .arg(format!("-arch={}", cuda_arch)) // Target architecture
        .arg("-std=c++17") // C++17 for CUB
        .arg(format!("-I{}", cuda_include.display())) // CUDA + CUB headers
        .arg(format!("-I{}", cuda_targets_include.display())) // CUDA target headers
        // Avoid glibc C23 rsqrt()/rsqrtf() host prototypes colliding with CUDA math declarations.
        // Keep X/Open interfaces enabled so libstdc++ headers (e.g., <cwchar>) still see fwide.
        .arg("-U_GNU_SOURCE")
        .arg("-D_XOPEN_SOURCE=700")
        .arg("-O3") // Maximum optimization
        // Fast math: omitted under the `strict-fp` feature so the (signal) RSI
        // fused kernel is bit-reproducible; default keeps it for throughput.
        .args(if std::env::var_os("CARGO_FEATURE_STRICT_FP").is_some() {
            &[] as &[&str]
        } else {
            &["-use_fast_math"] as &[&str]
        })
        .arg("--expt-relaxed-constexpr") // Relaxed constexpr for CUB
        .arg("--expt-extended-lambda") // Extended lambda for CUB
        .arg("-D_FORCE_INLINES") // Force inline to avoid header conflicts
        .arg("--diag-suppress=20092") // Suppress exception spec mismatch warning
        .arg("--diag-suppress=20041") // Suppress additional compatibility warnings
        .arg("-Xcompiler=-fPIC") // Position-independent code for shared library
        .arg("-Xcompiler=-w") // Suppress host compiler warnings
        .arg("-o")
        .arg(&output_lib)
        .arg(&kernel_source);

    println!("cargo:warning=Running: {:?}", cmd);

    // Execute nvcc
    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) => {
            println!(
                "cargo:warning=Failed to execute nvcc for RSI fused kernel: {}",
                e
            );
            return;
        }
    };

    // Check compilation result
    if output.status.success() {
        println!(
            "cargo:warning=Successfully compiled RSI fused kernel to: {}",
            output_lib.display()
        );

        // Emit linker directives
        println!(
            "cargo:rustc-env=RSI_FUSED_LIB_PATH={}",
            output_lib.display()
        );
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=dylib=rsi_fused");

        // Print stdout/stderr for debugging
        if !output.stdout.is_empty() {
            println!(
                "cargo:warning=nvcc stdout: {}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
    } else {
        println!(
            "cargo:warning=Failed to compile RSI fused kernel (non-critical, will use hybrid)"
        );
        println!("cargo:warning=Exit code: {:?}", output.status.code());

        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let first_lines: Vec<&str> = stderr.lines().take(15).collect();
            println!("cargo:warning=nvcc stderr: {}", first_lines.join("\n"));
        }
    }
}

/// Emit cargo directives for rebuild detection
fn emit_rebuild_directives() {
    // Rebuild if CUDA kernels change
    println!("cargo:rerun-if-changed=src/gpu/kernels/rsi_fused.cu");

    // Rebuild if build script changes
    println!("cargo:rerun-if-changed=build.rs");

    // Rebuild if environment variables change
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
}
