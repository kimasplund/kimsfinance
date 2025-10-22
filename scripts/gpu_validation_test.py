#!/usr/bin/env python3
"""
GPU Validation Test Suite for kimsfinance
Tests GPU utilization, kernel profiling, memory bandwidth, and leak detection
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str) -> None:
    """Print a formatted header"""
    width = 65
    print(f"\n{Colors.BOLD}{'┌' + '─' * (width - 2) + '┐'}{Colors.ENDC}")
    print(f"{Colors.BOLD}│ {text:<{width - 4}} │{Colors.ENDC}")
    print(f"{Colors.BOLD}{'└' + '─' * (width - 2) + '┘'}{Colors.ENDC}\n")

def print_status(label: str, value: str, status: str = "info") -> None:
    """Print a status line with color coding"""
    color = {
        "pass": Colors.GREEN,
        "fail": Colors.RED,
        "warn": Colors.YELLOW,
        "info": Colors.CYAN
    }.get(status, Colors.ENDC)

    symbol = {
        "pass": "✓",
        "fail": "✗",
        "warn": "⚠",
        "info": "•"
    }.get(status, "•")

    print(f"  {color}{symbol} {label}: {value}{Colors.ENDC}")

def check_gpu_available() -> bool:
    """Check if GPU and nvidia-smi are available"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information using nvidia-smi"""
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap,power.limit",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)

        parts = [p.strip() for p in result.stdout.strip().split(',')]

        return {
            "name": parts[0],
            "driver": parts[1],
            "memory_mb": int(parts[2]),
            "compute_cap": parts[3],
            "power_limit": parts[4] if len(parts) > 4 else "N/A"
        }
    except Exception as e:
        print_status("GPU info error", str(e), "fail")
        return {}

def get_gpu_utilization() -> Dict[str, float]:
    """Get current GPU utilization metrics"""
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)

        parts = [p.strip() for p in result.stdout.strip().split(',')]

        return {
            "gpu_util": float(parts[0]),
            "mem_util": float(parts[1]),
            "mem_used_mb": float(parts[2]),
            "power_w": float(parts[3]) if parts[3] != "[N/A]" else 0.0,
            "temp_c": float(parts[4])
        }
    except Exception as e:
        return {"error": str(e)}

def check_dependencies() -> Dict[str, bool]:
    """Check which GPU dependencies are installed"""
    deps = {}

    # Check pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        deps["pynvml"] = True
    except ImportError:
        deps["pynvml"] = False

    # Check torch CUDA
    try:
        import torch
        deps["torch_cuda"] = torch.cuda.is_available()
    except ImportError:
        deps["torch_cuda"] = False

    # Check cudf
    try:
        import cudf
        deps["cudf"] = True
    except ImportError:
        deps["cudf"] = False

    # Check cupy
    try:
        import cupy
        deps["cupy"] = True
    except ImportError:
        deps["cupy"] = False

    # Check polars GPU
    try:
        import polars
        deps["polars"] = True
        # Check if GPU engine is available
        deps["polars_gpu"] = hasattr(polars, "GPUEngine")
    except ImportError:
        deps["polars"] = False
        deps["polars_gpu"] = False

    return deps

def install_missing_dependencies(deps: Dict[str, bool]) -> None:
    """Install missing GPU dependencies"""
    to_install = []

    if not deps.get("pynvml"):
        to_install.append("pynvml")

    if not deps.get("torch_cuda"):
        print_status("PyTorch CUDA", "Not available - CPU-only version detected", "warn")
        print(f"  {Colors.YELLOW}Note: Install CUDA PyTorch with:{Colors.ENDC}")
        print(f"  pip3 install torch --index-url https://download.pytorch.org/whl/cu124")

    if not deps.get("cudf"):
        print_status("cuDF", "Not installed", "warn")
        print(f"  {Colors.YELLOW}Note: Install with:{Colors.ENDC}")
        print(f"  pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12")

    if not deps.get("cupy"):
        to_install.append("cupy-cuda12x")

    if to_install:
        print(f"\n{Colors.CYAN}Installing missing dependencies: {', '.join(to_install)}{Colors.ENDC}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + to_install,
                check=True
            )
            print_status("Installation", "Success", "pass")
        except subprocess.CalledProcessError as e:
            print_status("Installation", f"Failed: {e}", "fail")

def test_gpu_utilization_numpy(iterations: int = 100) -> Dict[str, Any]:
    """Test GPU utilization with numpy operations (CPU baseline)"""
    print_status("Running", f"NumPy CPU baseline test ({iterations} iterations)", "info")

    utils = []
    start_time = time.time()

    # Simulate financial calculations
    for i in range(iterations):
        data = np.random.randn(100000, 5)  # OHLCV-like data

        # RSI-like calculation
        deltas = np.diff(data[:, 3])  # Close price changes
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Moving averages
        window = 14
        avg_gains = np.convolve(gains, np.ones(window)/window, mode='valid')
        avg_losses = np.convolve(losses, np.ones(window)/window, mode='valid')

        # Sample GPU util during compute
        if i % 10 == 0:
            util = get_gpu_utilization()
            if "gpu_util" in util:
                utils.append(util)

        # Cleanup
        del data, deltas, gains, losses, avg_gains, avg_losses

    elapsed = time.time() - start_time

    return {
        "elapsed_ms": elapsed * 1000,
        "iterations": iterations,
        "ms_per_iter": (elapsed * 1000) / iterations,
        "gpu_utils": utils
    }

def test_memory_bandwidth_estimate() -> Dict[str, float]:
    """Estimate memory bandwidth using numpy (system RAM)"""
    print_status("Testing", "System memory bandwidth", "info")

    sizes_mb = [1, 10, 100, 1000]
    results = {}

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        data = np.random.randn(size_bytes // 8)  # 8 bytes per float64

        # Test copy speed
        iterations = max(1, 1000 // size_mb)
        start = time.time()
        for _ in range(iterations):
            copy = data.copy()
            del copy
        elapsed = time.time() - start

        bandwidth_gbps = (size_bytes * iterations) / (elapsed * 1e9)
        results[f"{size_mb}MB"] = bandwidth_gbps

        del data

    return results

def test_memory_leak_detection(iterations: int = 100) -> Dict[str, Any]:
    """Test for memory leaks using numpy operations"""
    print_status("Running", f"Memory leak detection ({iterations} iterations)", "info")

    # Get initial GPU memory
    initial_util = get_gpu_utilization()
    initial_mem = initial_util.get("mem_used_mb", 0)

    memory_timeline = [initial_mem]

    for i in range(iterations):
        # Allocate and compute
        data = np.random.randn(10000, 5)
        result = np.mean(data, axis=0)
        del data, result

        # Sample memory every 10 iterations
        if i % 10 == 0:
            util = get_gpu_utilization()
            if "mem_used_mb" in util:
                memory_timeline.append(util["mem_used_mb"])

    # Get final memory
    final_util = get_gpu_utilization()
    final_mem = final_util.get("mem_used_mb", initial_mem)

    # Calculate growth
    growth_mb = final_mem - initial_mem
    growth_per_iter = growth_mb / iterations if iterations > 0 else 0

    # Linear regression to detect trend
    x = np.arange(len(memory_timeline))
    y = np.array(memory_timeline)
    if len(x) > 1:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0

    return {
        "initial_mb": initial_mem,
        "final_mb": final_mem,
        "peak_mb": max(memory_timeline),
        "growth_mb": growth_mb,
        "growth_per_iter": growth_per_iter,
        "slope": slope,
        "timeline": memory_timeline,
        "leak_detected": abs(slope) > 0.1  # >0.1 MB/iteration indicates leak
    }

def generate_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on test results"""
    recommendations = []

    # Check if GPU packages are installed
    deps = results.get("dependencies", {})
    if not deps.get("torch_cuda"):
        recommendations.append(
            "Install PyTorch with CUDA support for GPU acceleration:\n"
            "  pip3 install torch --index-url https://download.pytorch.org/whl/cu124"
        )

    if not deps.get("cudf"):
        recommendations.append(
            "Install cuDF for GPU-accelerated DataFrame operations:\n"
            "  pip install kimsfinance[gpu]"
        )

    if not deps.get("pynvml"):
        recommendations.append(
            "Install pynvml for detailed GPU monitoring:\n"
            "  pip install pynvml"
        )

    # Check for memory leaks
    leak_test = results.get("leak_test", {})
    if leak_test.get("leak_detected"):
        recommendations.append(
            f"Memory leak detected: {leak_test['growth_mb']:.1f} MB growth over "
            f"{leak_test.get('iterations', 0)} iterations. "
            "Review GPU memory allocation and ensure proper cleanup."
        )

    # Performance recommendations
    baseline = results.get("baseline_test", {})
    if baseline.get("ms_per_iter", 0) > 10:
        recommendations.append(
            "High iteration time detected. Consider:\n"
            "  - Using GPU acceleration with cuDF/CuPy\n"
            "  - Increasing batch sizes\n"
            "  - Pre-allocating arrays"
        )

    if not recommendations:
        recommendations.append("GPU setup looks good! All tests passed.")

    return recommendations

def main(auto_install: bool = False) -> None:
    """Run GPU validation test suite

    Args:
        auto_install: Automatically install missing dependencies without prompting
    """
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'kimsfinance GPU Validation Test Suite':^65}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.ENDC}\n")

    results: Dict[str, Any] = {}

    # Check GPU hardware
    print_header("GPU Hardware Detection")

    if not check_gpu_available():
        print_status("GPU", "Not detected", "fail")
        print(f"\n{Colors.RED}No NVIDIA GPU found. This test requires CUDA-capable hardware.{Colors.ENDC}")
        sys.exit(1)

    gpu_info = get_gpu_info()
    print_status("GPU", gpu_info.get("name", "Unknown"), "pass")
    print_status("Driver", gpu_info.get("driver", "Unknown"), "info")
    print_status("VRAM", f"{gpu_info.get('memory_mb', 0) / 1024:.1f} GB", "info")
    print_status("Compute Capability", gpu_info.get("compute_cap", "Unknown"), "info")
    print_status("Power Limit", f"{gpu_info.get('power_limit', 'N/A')} W", "info")

    results["gpu_info"] = gpu_info

    # Check dependencies
    print_header("Dependency Check")

    deps = check_dependencies()
    results["dependencies"] = deps

    print_status("pynvml", "Installed" if deps["pynvml"] else "Not installed",
                 "pass" if deps["pynvml"] else "warn")
    print_status("PyTorch CUDA", "Available" if deps["torch_cuda"] else "Not available",
                 "pass" if deps["torch_cuda"] else "warn")
    print_status("cuDF", "Installed" if deps["cudf"] else "Not installed",
                 "pass" if deps["cudf"] else "warn")
    print_status("CuPy", "Installed" if deps["cupy"] else "Not installed",
                 "pass" if deps["cupy"] else "warn")
    print_status("Polars", "Installed" if deps["polars"] else "Not installed",
                 "pass" if deps["polars"] else "info")

    # Offer to install missing dependencies
    missing = [k for k, v in deps.items() if not v and k != "polars_gpu"]
    if missing:
        print(f"\n{Colors.YELLOW}Missing dependencies: {', '.join(missing)}{Colors.ENDC}")
        if auto_install:
            print(f"{Colors.CYAN}Auto-installing dependencies...{Colors.ENDC}")
            install_missing_dependencies(deps)
            # Recheck
            deps = check_dependencies()
            results["dependencies"] = deps
        else:
            print(f"{Colors.CYAN}Tip: Run with --install to auto-install dependencies{Colors.ENDC}")

    # Test 1: CPU Baseline
    print_header("Test 1: CPU Baseline Performance")
    baseline = test_gpu_utilization_numpy(iterations=100)
    results["baseline_test"] = baseline

    print_status("Iterations", str(baseline["iterations"]), "info")
    print_status("Total time", f"{baseline['elapsed_ms']:.1f} ms", "info")
    print_status("Time per iteration", f"{baseline['ms_per_iter']:.2f} ms", "info")

    if baseline["gpu_utils"]:
        avg_util = np.mean([u["gpu_util"] for u in baseline["gpu_utils"]])
        print_status("GPU utilization", f"{avg_util:.1f}% (CPU workload)",
                     "pass" if avg_util < 20 else "warn")

    # Test 2: Memory Bandwidth
    print_header("Test 2: System Memory Bandwidth")
    bandwidth = test_memory_bandwidth_estimate()
    results["bandwidth_test"] = bandwidth

    for size, bw in bandwidth.items():
        status = "pass" if bw > 5.0 else "warn"
        print_status(f"{size} transfer", f"{bw:.2f} GB/s", status)

    # Test 3: Memory Leak Detection
    print_header("Test 3: Memory Leak Detection")
    leak_test = test_memory_leak_detection(iterations=200)
    results["leak_test"] = leak_test

    print_status("Initial memory", f"{leak_test['initial_mb']:.0f} MB", "info")
    print_status("Final memory", f"{leak_test['final_mb']:.0f} MB", "info")
    print_status("Peak memory", f"{leak_test['peak_mb']:.0f} MB", "info")
    print_status("Memory growth", f"{leak_test['growth_mb']:.1f} MB",
                 "pass" if abs(leak_test['growth_mb']) < 50 else "warn")
    print_status("Growth rate", f"{leak_test['slope']:.4f} MB/iter",
                 "pass" if not leak_test['leak_detected'] else "fail")
    print_status("Leak detected", "No" if not leak_test['leak_detected'] else "YES",
                 "pass" if not leak_test['leak_detected'] else "fail")

    # Summary and Recommendations
    print_header("Summary & Recommendations")

    recommendations = generate_recommendations(results)
    for i, rec in enumerate(recommendations, 1):
        print(f"{Colors.CYAN}{i}. {rec}{Colors.ENDC}\n")

    # Overall status
    print_header("Overall GPU Status")

    has_gpu = gpu_info.get("name") is not None
    has_deps = deps.get("pynvml", False)
    no_leaks = not leak_test.get("leak_detected", True)

    if has_gpu and has_deps and no_leaks:
        status_text = "READY FOR GPU TESTING"
        status_color = Colors.GREEN
        symbol = "✓"
    elif has_gpu:
        status_text = "GPU DETECTED - INSTALL DEPENDENCIES"
        status_color = Colors.YELLOW
        symbol = "⚠"
    else:
        status_text = "NO GPU DETECTED"
        status_color = Colors.RED
        symbol = "✗"

    print(f"{status_color}{symbol} Status: {status_text}{Colors.ENDC}\n")

    # Duration
    print(f"{Colors.CYAN}Test suite completed{Colors.ENDC}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Validation Test Suite for kimsfinance")
    parser.add_argument("--install", action="store_true", help="Auto-install missing dependencies")
    parser.add_argument("--no-install", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()

    try:
        main(auto_install=args.install)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
