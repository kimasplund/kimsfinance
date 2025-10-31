#!/bin/bash
# Generate comprehensive CUDA-ext performance report
# Runs all benchmarks and generates markdown report

set -e

echo "===== CUDA-ext Performance Benchmark Suite ====="
echo ""
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU benchmarks require NVIDIA GPU."
    exit 1
fi

# Print hardware info
echo "===== Hardware Configuration ====="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "Compute Capability: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo ""

CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:\s*//')
CPU_CORES=$(nproc)
echo "CPU: ${CPU_MODEL}"
echo "CPU Cores: ${CPU_CORES}"
echo ""

# Check Rust version
echo "===== Software Versions ====="
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

echo "===== Running Benchmarks ====="
echo ""

# Create output directory
mkdir -p target/benchmark_reports
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_DIR="target/benchmark_reports/${TIMESTAMP}"
mkdir -p "${REPORT_DIR}"

# 1. Genetic optimizer comparison (mutex removal, scaling, convergence)
echo "[1/5] Running genetic optimizer comparison benchmarks..."
cargo bench --features gpu --bench genetic_optimizer_comparison 2>&1 | tee "${REPORT_DIR}/genetic_comparison.txt"

# 2. FP8 precision validation
echo ""
echo "[2/5] Running FP8 precision validation..."
cargo bench --features gpu --bench genetic_optimizer_precision 2>&1 | tee "${REPORT_DIR}/genetic_precision.txt"

# 3. GPU vs CPU backtest comparison
echo ""
echo "[3/5] Running GPU vs CPU backtest comparison..."
cargo bench --features gpu --bench backtest_gpu_cpu_comparison 2>&1 | tee "${REPORT_DIR}/gpu_cpu_comparison.txt" || echo "Note: GPU batch kernel not yet implemented (Agent 2 WIP)"

# 4. Combined optimizations (pinned memory, occupancy)
echo ""
echo "[4/5] Running combined optimizations benchmark..."
cargo bench --features gpu --bench combined_optimizations_benchmark 2>&1 | tee "${REPORT_DIR}/combined_optimizations.txt" || echo "Note: Some GPU optimizations pending"

# 5. Optimization validation
echo ""
echo "[5/5] Running optimization validation..."
cargo bench --features gpu --bench optimization_validation 2>&1 | tee "${REPORT_DIR}/optimization_validation.txt" || echo "Note: Validation suite incomplete"

echo ""
echo "===== Generating Performance Report ====="
echo ""

# Generate markdown report (if parser exists)
if [ -f "scripts/parse_benchmark_results.py" ]; then
    python3 scripts/parse_benchmark_results.py \
        "${REPORT_DIR}/genetic_comparison.txt" \
        "${REPORT_DIR}/genetic_precision.txt" \
        "${REPORT_DIR}/gpu_cpu_comparison.txt" \
        > "${REPORT_DIR}/PERFORMANCE_REPORT.md"

    echo "Markdown report: ${REPORT_DIR}/PERFORMANCE_REPORT.md"
else
    echo "Note: Benchmark parser not found. Skipping markdown generation."
    echo "Raw benchmark outputs saved to: ${REPORT_DIR}/"
fi

# Copy Criterion HTML reports
if [ -d "target/criterion" ]; then
    cp -r target/criterion "${REPORT_DIR}/criterion_html"
    echo "Criterion HTML reports: ${REPORT_DIR}/criterion_html/report/index.html"
fi

echo ""
echo "===== Benchmark Suite Complete ====="
echo ""
echo "Results saved to: ${REPORT_DIR}/"
echo ""
echo "To view Criterion HTML reports:"
echo "  open ${REPORT_DIR}/criterion_html/report/index.html"
echo ""
echo "To view detailed benchmark logs:"
echo "  cat ${REPORT_DIR}/genetic_comparison.txt"
echo "  cat ${REPORT_DIR}/genetic_precision.txt"
echo ""
