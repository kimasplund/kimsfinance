# CLAUDE.md

## Project Configuration for Claude Code

This file configures Claude Code for the kimsfinance project.

**Project**: kimsfinance v0.1.0
**Domain**: Finance / Quantitative Trading / Data Visualization
**Language**: Python 3.13+
**GPU**: Optional (NVIDIA RTX 3500 Ada)

---

## Agents

This project uses **11 specialized agents** for performance optimization, testing, and advanced reasoning:

### Performance & Benchmarking

1. **kimsfinance-benchmark-specialist**
   - Purpose: Specialized benchmarking for 178x performance validation
   - Location: `.claude/agents/kimsfinance-benchmark-specialist.md`
   - Use for: Running comprehensive benchmarks, comparing with baseline

2. **kimsfinance-performance-tester**
   - Purpose: Performance regression testing across releases
   - Location: `.claude/agents/kimsfinance-performance-tester.md`
   - Use for: Validating performance claims, detecting regressions

### GPU Optimization

3. **cudf-gpu-optimizer**
   - Purpose: Optimize GPU-accelerated OHLCV processing (6.4x speedup)
   - Location: `.claude/agents/cudf-gpu-optimizer.md`
   - Use for: cuDF integration, GPU dataframe operations

4. **cuda-python-expert**
   - Purpose: CUDA optimization for Python (CuPy, Numba CUDA)
   - Location: `.claude/agents/cuda-python-expert.md`
   - Use for: Technical indicators on GPU, kernel optimization

5. **python-jit-optimizer**
   - Purpose: Numba JIT optimization (50-100% faster coordinate computation)
   - Location: `.claude/agents/python-jit-optimizer.md`
   - Use for: JIT compilation, performance hotspots

### Documentation & Git

6. **docs-git-committer**
   - Purpose: Sync documentation with code changes
   - Location: `.claude/agents/docs-git-committer.md`
   - Use for: Updating README, API docs, committing changes

7. **git-expert**
   - Purpose: Advanced git operations, branch management, conflict resolution
   - Location: `.claude/agents/git-expert.md`
   - Use for: Complex merges, rebases, cherry-picks, git workflows

### Cognitive Reasoning

8. **integrated-reasoning**
   - Purpose: Master orchestrator for cognitive reasoning patterns (95-99% confidence)
   - Location: `.claude/agents/integrated-reasoning.md`
   - Use for: Complex architectural decisions, multi-dimensional problems, high-stakes choices
   - Performance: 45-60 minutes for maximum rigor

9. **integrated-reasoning-fast**
   - Purpose: Fast version of integrated-reasoning (75-85% confidence)
   - Location: `.claude/agents/integrated-reasoning-fast.md`
   - Use for: Important but not critical decisions when time is limited
   - Performance: 15 minutes for quick but thorough analysis

10. **breadth-of-thought**
   - Purpose: Exhaustive exploration of all possible solutions
   - Location: `.claude/agents/breadth-of-thought.md`
   - Use for: Unknown solution spaces, multiple valid approaches, comprehensive analysis
   - Method: Spawns 8-10 parallel analyses, explores wide before deep

11. **tree-of-thoughts**
    - Purpose: Advanced recursive reasoning with parallel task exploration
    - Location: `.claude/agents/tree-of-thoughts.md`
    - Use for: Optimization problems, trade-off analysis, finding best path
    - Method: Minimum 5 parallel branches per level, 4+ levels deep

---

## Skills

No skills plugins are currently configured.

Skills can be added via:
```bash
/plugin add marketplace anthropics/skills
```

---

## MCPs

This project uses **2 MCP servers**:

1. **filesystem** - Standard file operations
2. **ide** - VS Code integration for diagnostics (if available)

Configuration: `.claude/mcp-servers.json`

---

## Commands

### Global Commands (from `~/.claude/commands/`)

**kimsfinance Benchmark & Testing Suite** (`/kf/*` - 9 commands):

**Benchmarking**:
- `/kf/bench/all` - Run all benchmarks comprehensively
- `/kf/bench/compare` - Compare performance with mplfinance
- `/kf/bench/scaling` - Test scaling with dataset size

**Testing**:
- `/kf/test/gpu` - GPU validation tests (CUDA, cuDF, memory)
- `/kf/test/memory` - Memory leak detection
- `/kf/test/performance` - Performance regression tests

**Profiling**:
- `/kf/profile/gpu-kernel` - Profile GPU kernels with Nsight
- `/kf/profile/full` - Full profiling with cProfile

**Main Entry**:
- `/kf` - Interactive menu for all kimsfinance commands

### Project Commands (in `.claude/commands/`)

**Development**:
- `/test [pattern]` - Run pytest test suite (329+ tests)
- `/build [mode]` - Build Python package with setuptools
- `/benchmark-quick [candles]` - Quick performance sanity check

---

## Project-Specific Instructions

### Performance Standards

kimsfinance claims **178x speedup** over mplfinance baseline. Maintain these targets:

| Metric | Target | Excellent |
|--------|--------|-----------|
| Chart Rendering | <10ms | <5ms |
| Throughput | >1000 img/sec | >6000 img/sec |
| Speedup | >50x | >150x |
| File Size (WebP) | <1 KB | <0.5 KB |

### Development Workflow

1. **Before Changes**: Run `/benchmark-quick` to establish baseline
2. **Make Changes**: Implement optimization or feature
3. **Test**: Run `/test` to ensure no regressions
4. **Benchmark**: Run `/kf/bench/compare` to validate improvement
5. **Document**: Use `docs-git-committer` agent to update docs
6. **Commit**: Commit with descriptive message

### GPU Development

- GPU is **optional** - code must work on CPU-only systems
- Use smart routing: GPU for OHLCV processing, CPU for rendering
- Test both CPU and GPU paths: `/kf/test/gpu`
- Monitor memory: `/kf/test/memory`

### Code Quality

- **Python 3.13+** required
- **Type hints**: Use mypy strict mode
- **Testing**: Maintain 329+ tests, add tests for new features
- **Performance**: Never accept regressions without justification

### Benchmarking Best Practices

- Always use `/kf/bench/compare` for official comparisons
- Run benchmarks on stable system (no background loads)
- Check CPU governor: `performance` mode preferred
- Monitor thermals: prevent throttling during long benchmarks

---

## Performance Targets by Operation

### Chart Rendering (Sequential)
- **50 candles**: <5ms (Target: >200 charts/sec)
- **100 candles**: <8ms (Target: >125 charts/sec)
- **500 candles**: <30ms (Target: >33 charts/sec)

### Batch Rendering (1000 charts)
- **WebP Fast Mode**: <2 seconds total
- **Throughput**: >500 charts/sec

### OHLCV Processing (GPU)
- **cuDF vs pandas**: 6.4x speedup minimum
- **1M candles**: <200ms processing time

### Technical Indicators (GPU)
- **ATR**: 1.2-1.5x speedup over CPU
- **RSI**: 1.5-2.0x speedup over CPU
- **Stochastic**: 2.0-2.9x speedup over CPU

---

## Common Tasks

### Run All Tests
```bash
/test
```

### Quick Performance Check
```bash
/benchmark-quick
```

### Comprehensive Benchmark
```bash
/kf/bench/all
```

### Build Package
```bash
/build all
```

### GPU Validation
```bash
/kf/test/gpu
```

### Profile Performance
```bash
/kf/profile/full
```

---

## Notes

- **Dual Licensing**: AGPL-3.0 (open source) + Commercial License
- **Production Ready**: Beta (v0.1.0), extensively tested
- **Hardware**: Tested on ThinkPad P16 Gen2 (i9-13980HX, RTX 3500 Ada)
- **Dependencies**: Pillow 12.0+, NumPy 2.0+, Polars 1.0+

---

**Last Updated**: 2025-10-20
**Initialized by**: init-workspace-v2
**Agent Configuration**: 11 agents (performance, GPU, docs, git, cognitive reasoning)
**Command Suite**: 12 commands (9 global + 3 project-specific)
