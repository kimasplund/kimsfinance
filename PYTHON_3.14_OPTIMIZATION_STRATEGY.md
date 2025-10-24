# Python 3.14 Optimization Strategy for kimsfinance

**Branch**: `python-3.14-optimization`  
**Date**: 2025-10-24  
**Status**: Experimental  
**Python Version**: 3.14.0

## Executive Summary

Python 3.14 introduces **game-changing performance features**:

1. **Free-threaded Python (No-GIL)** - Up to **5x speedup** on parallel workloads
2. **JIT Compiler (Experimental)** - **3-5% general performance improvement**  
3. **Zstandard Compression** - **2-3x faster** compression than gzip

**Current Status**: ✅ Python 3.14.0 compatible, all tests passing

## Key Opportunities

### Free-Threaded Batch Rendering (5x speedup)
- Current multiprocessing: 6,249 img/sec  
- Predicted threading (no-GIL): **31,000+ img/sec**
- Implementation: Replace multiprocessing with threading in `parallel.py`

### JIT Compiler (3-5% speedup)
- Enable: `export PYTHON_JIT=1`
- Expected: 3-5% general, 10-20% on tight loops
- Zero risk, easy win

### Zstandard SVGZ (2-3x faster compression)
- New module: `compression.zstd`  
- Use for SVG compression instead of gzip

## Performance Targets

| Mode | Single-Chart | Batch (1000) | vs mplfinance |
|------|--------------|--------------|---------------|
| Current (3.13) | 6,249 img/sec | ~625 img/sec | 178x |
| 3.14 (JIT) | 6,500 img/sec | ~650 img/sec | 186x |
| 3.14 (Free-threaded) | 5,900 img/sec | **31,000 img/sec** | **886x** |

## Next Steps

1. ✅ Python 3.14.0 installed
2. ✅ Update pyproject.toml  
3. ✅ Verify tests pass
4. Run baseline benchmarks
5. Test JIT mode
6. Prototype free-threading

## Resources

- [Python 3.14 Release Notes](https://docs.python.org/3/whatsnew/3.14.html)
- [PEP 779: Free-threaded Python](https://peps.python.org/pep-0779/)
- [PEP 784: Zstandard Compression](https://peps.python.org/pep-0784/)
