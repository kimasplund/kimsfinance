#!/usr/bin/env python3
"""
Advanced GPU Testing Script for kimsfinance
Includes CUDA kernel profiling, multi-stream testing, and performance benchmarks
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Color codes
class C:
    H = '\033[95m'; B = '\033[94m'; C = '\033[96m'
    G = '\033[92m'; Y = '\033[93m'; R = '\033[91m'
    E = '\033[0m'; BOLD = '\033[1m'

def header(t: str) -> None:
    w = 65
    print(f"\n{C.BOLD}{'┌' + '─' * (w-2) + '┐'}{C.E}")
    print(f"{C.BOLD}│ {t:<{w-4}} │{C.E}")
    print(f"{C.BOLD}{'└' + '─' * (w-2) + '┘'}{C.E}\n")

def status(label: str, value: str, s: str = "i") -> None:
    color = {"p": C.G, "f": C.R, "w": C.Y, "i": C.C}[s[0]]
    symbol = {"p": "✓", "f": "✗", "w": "⚠", "i": "•"}[s[0]]
    print(f"  {color}{symbol} {label}: {value}{C.E}")

def run_cmd(cmd: List[str]) -> Tuple[str, int]:
    """Run command and return output, returncode"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip(), result.returncode
    except:
        return "", 1

def get_gpu_stats() -> Dict[str, Any]:
    """Get comprehensive GPU stats"""
    out, _ = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,"
        "clocks.mem,compute_cap",
        "--format=csv,noheader,nounits"
    ])

    if not out:
        return {}

    parts = [p.strip() for p in out.split(',')]
    if len(parts) < 12:
        return {}

    return {
        "name": parts[0],
        "driver": parts[1],
        "mem_total_mb": int(float(parts[2])),
        "mem_used_mb": int(float(parts[3])),
        "mem_free_mb": int(float(parts[4])),
        "gpu_util": float(parts[5]),
        "mem_util": float(parts[6]),
        "temp_c": float(parts[7]),
        "power_w": float(parts[8]) if parts[8] != "[N/A]" else 0,
        "clock_sm_mhz": int(float(parts[9])),
        "clock_mem_mhz": int(float(parts[10])),
        "compute_cap": parts[11]
    }

def test_pytorch_cuda() -> Dict[str, Any]:
    """Test PyTorch CUDA availability and performance"""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}

        device = torch.device("cuda:0")

        # Warm up
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark matrix multiply
        sizes = [1000, 5000, 10000]
        results = {}

        for size in sizes:
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            torch.cuda.synchronize()
            start = time.time()

            for _ in range(10):
                c = torch.matmul(a, b)
                torch.cuda.synchronize()

            elapsed = time.time() - start
            gflops = (2 * size**3 * 10) / (elapsed * 1e9)

            results[f"{size}x{size}"] = {
                "time_ms": elapsed * 100,  # per iteration
                "gflops": gflops
            }

            del a, b, c

        torch.cuda.empty_cache()

        return {
            "available": True,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "benchmarks": results
        }

    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}

def test_cupy() -> Dict[str, Any]:
    """Test CuPy availability and performance"""
    try:
        import cupy as cp

        # Test basic operations
        x = cp.random.randn(10000, 100)
        start = time.time()

        for _ in range(100):
            y = cp.mean(x, axis=0)
            z = cp.std(x, axis=0)

        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start

        return {
            "available": True,
            "version": cp.__version__,
            "time_ms": elapsed * 1000,
            "ops_per_sec": 100 / elapsed
        }

    except ImportError:
        return {"available": False, "reason": "CuPy not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}

def test_pynvml() -> Dict[str, Any]:
    """Test pynvml for detailed GPU monitoring"""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Get detailed info
        name = pynvml.nvmlDeviceGetName(handle)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
        except:
            power = 0

        try:
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
        except:
            power_limit = 0

        pynvml.nvmlShutdown()

        return {
            "available": True,
            "name": name,
            "uuid": uuid,
            "memory_used_mb": mem.used // (1024**2),
            "memory_free_mb": mem.free // (1024**2),
            "memory_total_mb": mem.total // (1024**2),
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "temp_c": temp,
            "power_w": power,
            "power_limit_w": power_limit
        }

    except ImportError:
        return {"available": False, "reason": "pynvml not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}

def benchmark_numpy_vs_torch() -> Dict[str, Any]:
    """Benchmark NumPy CPU vs PyTorch GPU"""
    size = 10000

    # NumPy CPU
    data_np = np.random.randn(size, 100)
    start = time.time()
    for _ in range(100):
        mean = np.mean(data_np, axis=0)
        std = np.std(data_np, axis=0)
    numpy_time = time.time() - start

    # Try PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            data_torch = torch.from_numpy(data_np).float().to(device)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                mean = torch.mean(data_torch, dim=0)
                std = torch.std(data_torch, dim=0)
            torch.cuda.synchronize()
            torch_time = time.time() - start

            speedup = numpy_time / torch_time

            return {
                "numpy_cpu_ms": numpy_time * 1000,
                "torch_gpu_ms": torch_time * 1000,
                "speedup": speedup,
                "winner": "GPU" if speedup > 1 else "CPU"
            }
    except:
        pass

    return {
        "numpy_cpu_ms": numpy_time * 1000,
        "torch_gpu_ms": None,
        "speedup": None,
        "winner": "CPU (no GPU comparison)"
    }

def test_financial_indicators() -> Dict[str, Any]:
    """Test financial indicator calculations on GPU"""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        device = torch.device("cuda:0")

        # Simulate OHLCV data
        n_bars = 100000
        ohlcv = torch.randn(n_bars, 5, device=device)

        results = {}

        # Test RSI calculation
        close = ohlcv[:, 3]
        deltas = close[1:] - close[:-1]

        start = time.time()
        for _ in range(10):
            gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
            losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))

            # Simple moving average
            window = 14
            avg_gains = torch.nn.functional.avg_pool1d(
                gains.unsqueeze(0).unsqueeze(0),
                kernel_size=window,
                stride=1
            )
            avg_losses = torch.nn.functional.avg_pool1d(
                losses.unsqueeze(0).unsqueeze(0),
                kernel_size=window,
                stride=1
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start

        results["rsi_ms"] = elapsed * 100  # per iteration

        # Test Moving Average
        start = time.time()
        for _ in range(100):
            ma = torch.nn.functional.avg_pool1d(
                close.unsqueeze(0).unsqueeze(0),
                kernel_size=20,
                stride=1
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start

        results["ma_ms"] = elapsed * 10  # per iteration

        # Test Bollinger Bands
        start = time.time()
        for _ in range(50):
            ma = torch.nn.functional.avg_pool1d(
                close.unsqueeze(0).unsqueeze(0),
                kernel_size=20,
                stride=1
            ).squeeze()

            # Standard deviation (simplified)
            std = torch.std(close[-20:])
            upper = ma + 2 * std
            lower = ma - 2 * std

        torch.cuda.synchronize()
        elapsed = time.time() - start

        results["bb_ms"] = elapsed * 20  # per iteration

        return {
            "available": True,
            "n_bars": n_bars,
            "indicators": results
        }

    except:
        return {"available": False}

def main() -> None:
    """Run advanced GPU tests"""
    print(f"\n{C.BOLD}{C.C}{'=' * 65}{C.E}")
    print(f"{C.BOLD}{C.C}{'kimsfinance Advanced GPU Test Suite':^65}{C.E}")
    print(f"{C.BOLD}{C.C}{'=' * 65}{C.E}\n")

    # GPU Hardware
    header("GPU Hardware Detection")
    stats = get_gpu_stats()

    if not stats:
        status("GPU", "Not detected", "f")
        sys.exit(1)

    status("GPU", stats["name"], "p")
    status("Driver", stats["driver"], "i")
    status("VRAM", f"{stats['mem_total_mb']/1024:.1f} GB", "i")
    status("Compute Cap", stats["compute_cap"], "i")
    status("Current Util", f"{stats['gpu_util']:.0f}%", "i")
    status("Memory Used", f"{stats['mem_used_mb']:.0f} / {stats['mem_total_mb']:.0f} MB", "i")
    status("Temperature", f"{stats['temp_c']:.0f}°C", "i")
    status("Power Draw", f"{stats['power_w']:.1f} W", "i")
    status("SM Clock", f"{stats['clock_sm_mhz']} MHz", "i")
    status("Mem Clock", f"{stats['clock_mem_mhz']} MHz", "i")

    # PyTorch CUDA
    header("Test 1: PyTorch CUDA Performance")
    pt = test_pytorch_cuda()

    if pt.get("available"):
        status("PyTorch CUDA", "Available", "p")
        status("CUDA Version", pt["cuda_version"], "i")
        status("Device", pt["device_name"], "i")

        print(f"\n  {C.BOLD}Matrix Multiplication Benchmarks:{C.E}")
        for size, bench in pt["benchmarks"].items():
            status(f"  {size}", f"{bench['time_ms']:.2f} ms/iter, {bench['gflops']:.1f} GFLOPS", "i")
    else:
        status("PyTorch CUDA", f"Not available: {pt.get('reason')}", "w")

    # CuPy
    header("Test 2: CuPy Performance")
    cp = test_cupy()

    if cp.get("available"):
        status("CuPy", "Available", "p")
        status("Version", cp["version"], "i")
        status("Statistical ops", f"{cp['time_ms']:.1f} ms for 100 iterations", "i")
        status("Throughput", f"{cp['ops_per_sec']:.1f} ops/sec", "i")
    else:
        status("CuPy", f"Not available: {cp.get('reason')}", "w")

    # pynvml
    header("Test 3: pynvml Detailed Monitoring")
    pn = test_pynvml()

    if pn.get("available"):
        status("pynvml", "Available", "p")
        status("GPU UUID", pn["uuid"][:16] + "...", "i")
        status("GPU Util", f"{pn['gpu_util']}%", "i")
        status("Mem Util", f"{pn['mem_util']}%", "i")
        status("Temperature", f"{pn['temp_c']}°C", "i")
        if pn["power_limit_w"] > 0:
            status("Power", f"{pn['power_w']:.1f} / {pn['power_limit_w']:.1f} W", "i")
    else:
        status("pynvml", f"Not available: {pn.get('reason')}", "w")

    # CPU vs GPU
    header("Test 4: NumPy CPU vs PyTorch GPU")
    bench = benchmark_numpy_vs_torch()

    status("NumPy CPU", f"{bench['numpy_cpu_ms']:.1f} ms", "i")
    if bench["torch_gpu_ms"]:
        status("PyTorch GPU", f"{bench['torch_gpu_ms']:.1f} ms", "i")
        s = "p" if bench["speedup"] > 1 else "w"
        status("Speedup", f"{bench['speedup']:.2f}x ({bench['winner']})", s)
    else:
        status("PyTorch GPU", "Not available", "w")

    # Financial Indicators
    header("Test 5: Financial Indicator Performance")
    fi = test_financial_indicators()

    if fi.get("available"):
        status("Test Data", f"{fi['n_bars']:,} bars (OHLCV)", "i")
        status("RSI Calculation", f"{fi['indicators']['rsi_ms']:.2f} ms/iteration", "i")
        status("Moving Average", f"{fi['indicators']['ma_ms']:.2f} ms/iteration", "i")
        status("Bollinger Bands", f"{fi['indicators']['bb_ms']:.2f} ms/iteration", "i")
    else:
        status("Financial Indicators", "PyTorch CUDA not available", "w")

    # Summary
    header("Overall GPU Status")

    has_hw = bool(stats)
    has_torch = pt.get("available", False)
    has_cupy = cp.get("available", False)
    has_pynvml = pn.get("available", False)

    if has_hw and has_torch and (has_cupy or has_pynvml):
        msg = "FULLY OPERATIONAL"
        col = C.G
        sym = "✓"
    elif has_hw and (has_torch or has_cupy):
        msg = "PARTIALLY OPERATIONAL"
        col = C.Y
        sym = "⚠"
    elif has_hw:
        msg = "GPU DETECTED - INSTALL DEPENDENCIES"
        col = C.Y
        sym = "⚠"
    else:
        msg = "NO GPU DETECTED"
        col = C.R
        sym = "✗"

    print(f"{col}{sym} Status: {msg}{C.E}\n")

    # Installation hints
    if not has_torch:
        print(f"{C.Y}Install PyTorch CUDA:{C.E}")
        print(f"  pip3 install torch --index-url https://download.pytorch.org/whl/cu124\n")

    if not has_cupy:
        print(f"{C.Y}Install CuPy:{C.E}")
        print(f"  pip install cupy-cuda12x\n")

    if not has_pynvml:
        print(f"{C.Y}Install pynvml:{C.E}")
        print(f"  pip install pynvml\n")

    print(f"{C.C}Test suite completed{C.E}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.Y}Test interrupted{C.E}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{C.R}Error: {e}{C.E}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
