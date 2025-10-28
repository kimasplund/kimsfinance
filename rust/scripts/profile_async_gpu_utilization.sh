#!/bin/bash
# Profile GPU utilization for async vs fused modes

set -e

echo "=== GPU Utilization Profiling ==="
echo

# Start GPU monitoring in background
nvidia-smi dmon -s pucvmet -c 60 -d 1 > /tmp/gpu_utilization_async.txt &
GPU_MON_PID=$!

echo "Running async mode benchmark..."
cargo bench --bench async_execution_benchmark --features gpu -- async_1000 --sample-size 10

sleep 2

echo "Running fused mode benchmark..."
cargo bench --bench async_execution_benchmark --features gpu -- fused_1000 --sample-size 10

# Kill GPU monitor
kill $GPU_MON_PID 2>/dev/null || true

echo
echo "GPU utilization saved to /tmp/gpu_utilization_async.txt"
echo

# Analyze utilization
python3 <<'EOF'
import re

with open('/tmp/gpu_utilization_async.txt', 'r') as f:
    lines = f.readlines()

# Parse GPU utilization (simplified)
utils = []
for line in lines:
    if line.strip() and not line.startswith('#'):
        parts = line.split()
        if len(parts) >= 3 and parts[2].isdigit():
            utils.append(int(parts[2]))

if utils:
    avg_util = sum(utils) / len(utils)
    max_util = max(utils)
    print(f"Average GPU utilization: {avg_util:.1f}%")
    print(f"Peak GPU utilization: {max_util}%")

    if avg_util >= 80:
        print("✅ GPU utilization is excellent (>= 80%)")
    elif avg_util >= 60:
        print("⚠️  GPU utilization is moderate (60-80%)")
    else:
        print("❌ GPU utilization is low (< 60%)")
else:
    print("⚠️  Could not parse GPU utilization data")
EOF
