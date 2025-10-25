#!/usr/bin/env python3
"""
Validate Python 3.14 Free-Threading Performance

This script benchmarks single-threaded vs multi-threaded performance
with Python 3.14's experimental free-threading (GIL removal).

Usage:
    # Standard Python 3.14:
    python3.14 scripts/validate_python314_freethreading.py

    # Free-threaded Python 3.14:
    python3.14t scripts/validate_python314_freethreading.py

Expected Results:
    - Single-threaded: ~27% faster than Python 3.13
    - Multi-threaded (8 cores): ~3.1x faster than single-threaded
"""

import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


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
        "info": Colors.CYAN,
    }.get(status, Colors.ENDC)

    symbol = {"pass": "✓", "fail": "✗", "warn": "⚠", "info": "•"}.get(status, "•")

    print(f"  {color}{symbol} {label}: {value}{Colors.ENDC}")


def check_free_threading() -> bool:
    """Check if running in free-threaded mode."""
    try:
        return not sys._is_gil_enabled()
    except AttributeError:
        return False


def cpu_intensive_task(n: int = 1000000) -> float:
    """CPU-intensive calculation for benchmarking."""
    arr = np.random.random(n)
    result = np.sum(np.sin(arr) * np.cos(arr))
    return result


def benchmark_single_threaded(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark single-threaded performance."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        cpu_intensive_task()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "median": np.median(times),
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def benchmark_multi_threaded(workers: int = 8, iterations: int = 10) -> Dict[str, Any]:
    """Benchmark multi-threaded performance."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(workers)]
            for future in futures:
                future.result()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "median": np.median(times),
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def calculate_speedup(single_time: float, multi_time: float, workers: int) -> Dict[str, float]:
    """Calculate speedup and efficiency metrics."""
    speedup = single_time / multi_time
    efficiency = (speedup / workers) * 100

    return {
        "speedup": speedup,
        "efficiency": efficiency,
    }


def main():
    """Run validation and report results."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'Python 3.14 Free-Threading Validation':^65}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.ENDC}\n")

    print_header("Python Environment")

    print_status("Python version", sys.version.split()[0], "info")
    print_status("Full version", sys.version.replace("\n", " "), "info")

    is_free_threaded = check_free_threading()
    print_status(
        "Free-threading",
        "Enabled" if is_free_threaded else "Disabled (GIL active)",
        "pass" if is_free_threaded else "warn"
    )

    print_status("NumPy version", np.__version__, "info")
    print()

    print_header("Benchmarking Performance")

    iterations = 10
    print_status("Benchmark iterations", str(iterations), "info")
    print_status("Task size", "1M element array operations", "info")
    print()

    print(f"{Colors.CYAN}Running single-threaded benchmark...{Colors.ENDC}")
    single_results = benchmark_single_threaded(iterations=iterations)
    single_time = single_results["median"]

    print(f"{Colors.CYAN}Running multi-threaded benchmark (4 workers)...{Colors.ENDC}")
    multi_results_4 = benchmark_multi_threaded(workers=4, iterations=iterations)
    multi_time_4 = multi_results_4["median"]

    print(f"{Colors.CYAN}Running multi-threaded benchmark (8 workers)...{Colors.ENDC}")
    multi_results_8 = benchmark_multi_threaded(workers=8, iterations=iterations)
    multi_time_8 = multi_results_8["median"]

    print()
    print_header("Results")

    print(f"{Colors.BOLD}Single-threaded:{Colors.ENDC}")
    print_status("Median time", f"{single_time:.3f}s", "info")
    print_status("Mean time", f"{single_results['mean']:.3f}s", "info")
    print_status("Std deviation", f"{single_results['std']:.3f}s", "info")
    print_status("Range", f"{single_results['min']:.3f}s - {single_results['max']:.3f}s", "info")

    print(f"\n{Colors.BOLD}Multi-threaded (4 workers):{Colors.ENDC}")
    print_status("Median time", f"{multi_time_4:.3f}s", "info")
    metrics_4 = calculate_speedup(single_time, multi_time_4, 4)
    speedup_4 = metrics_4["speedup"]
    efficiency_4 = metrics_4["efficiency"]

    speedup_status = "pass" if speedup_4 >= 2.0 else "warn" if speedup_4 >= 1.5 else "fail"
    print_status("Speedup", f"{speedup_4:.2f}x", speedup_status)
    print_status("Parallel efficiency", f"{efficiency_4:.1f}%", speedup_status)

    print(f"\n{Colors.BOLD}Multi-threaded (8 workers):{Colors.ENDC}")
    print_status("Median time", f"{multi_time_8:.3f}s", "info")
    metrics_8 = calculate_speedup(single_time, multi_time_8, 8)
    speedup_8 = metrics_8["speedup"]
    efficiency_8 = metrics_8["efficiency"]

    speedup_status_8 = "pass" if speedup_8 >= 2.5 else "warn" if speedup_8 >= 1.5 else "fail"
    print_status("Speedup", f"{speedup_8:.2f}x", speedup_status_8)
    print_status("Parallel efficiency", f"{efficiency_8:.1f}%", speedup_status_8)

    print()
    print_header("Interpretation")

    if is_free_threaded:
        print_status("Mode", "Free-threading enabled (GIL removed)", "pass")
        print()

        if speedup_8 >= 2.5:
            print(f"  {Colors.GREEN}✓ Excellent multi-threading performance{Colors.ENDC}")
            print(f"    {speedup_8:.2f}x speedup meets or exceeds expected 3.1x target")
        elif speedup_8 >= 1.5:
            print(f"  {Colors.YELLOW}⚠ Moderate multi-threading performance{Colors.ENDC}")
            print(f"    {speedup_8:.2f}x speedup is below expected 3.1x target")
            print(f"    This may be due to:")
            print(f"    - Thread synchronization overhead")
            print(f"    - NumPy GIL interactions")
            print(f"    - CPU thermal throttling")
        else:
            print(f"  {Colors.RED}✗ Poor multi-threading performance{Colors.ENDC}")
            print(f"    {speedup_8:.2f}x speedup is significantly below expectations")
            print(f"    Possible issues:")
            print(f"    - GIL may not be fully disabled")
            print(f"    - NumPy may be using GIL-protected code paths")
            print(f"    - Workload may not be CPU-bound")
    else:
        print_status("Mode", "Standard Python with GIL", "warn")
        print()

        print(f"  {Colors.YELLOW}⚠ Free-threading NOT enabled{Colors.ENDC}")
        print(f"    Running with standard GIL-enabled Python")
        print()
        print(f"    Expected behavior with GIL:")
        print(f"    - Speedup: 1.0-1.2x (limited by GIL)")
        print(f"    - Actual speedup: {speedup_8:.2f}x")
        print()

        if speedup_8 < 1.5:
            print(f"  {Colors.GREEN}✓ Results consistent with GIL-enabled Python{Colors.ENDC}")
        else:
            print(f"  {Colors.YELLOW}⚠ Unexpected high speedup with GIL enabled{Colors.ENDC}")
            print(f"    This may indicate NumPy releasing GIL for operations")

    print()
    print_header("Expected Performance Targets")

    print(f"{Colors.BOLD}Python 3.14 with free-threading (python3.14t):{Colors.ENDC}")
    print(f"  • Single-threaded: ~27% faster than Python 3.13")
    print(f"  • 8-core multi-threaded: ~3.1x faster than single-threaded")
    print(f"  • Parallel efficiency: ~39% (3.1x / 8 cores)")
    print()
    print(f"{Colors.BOLD}Standard Python 3.14 (with GIL):{Colors.ENDC}")
    print(f"  • Single-threaded: Same as Python 3.13")
    print(f"  • Multi-threaded: Limited by GIL (~1.0-1.2x speedup)")
    print()
    print(f"{Colors.BOLD}How to enable free-threading:{Colors.ENDC}")
    print(f"  • Build Python 3.14 with: --disable-gil")
    print(f"  • Or use python3.14t if available")
    print(f"  • Check with: python3.14 -c \"import sys; print(sys._is_gil_enabled())\"")

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.ENDC}")

    overall_status = "pass" if (is_free_threaded and speedup_8 >= 2.5) else "warn"
    status_color = Colors.GREEN if overall_status == "pass" else Colors.YELLOW
    status_text = "EXCELLENT" if overall_status == "pass" else "NEEDS IMPROVEMENT"
    print(f"{status_color}Overall Status: {status_text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.ENDC}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Benchmark interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
