# Custom Candles Implementation - Master Plan

## Agent Fleet Assignment (8 Parallel Agents)

### Agent 1: Foundation & Core Types
**Files to create:**
- `src/gpu/candles/mod.rs` - Module structure and exports
- `src/gpu/candles/types.rs` - TradeData, CandleData, common types
- `src/gpu/candles/traits.rs` - CandleAggregator trait system

**Deliverables:**
- Core data structures (TradeData, OHLCVCandle)
- Trait system for candle aggregators
- Integration with existing persistent kernel infrastructure

---

### Agent 2: Time Bar Aggregation
**Files to create:**
- `src/gpu/candles/time_bars.rs` - Time-based candle aggregation kernel

**Deliverables:**
- `TimeBarAggregator` implementation
- `TimeBarParams` (interval_seconds: 60, 300, 3600, etc.)
- CUDA kernel: `persistent_time_bars_kernel`
- Batch type: `TimeBarBatch`

---

### Agent 3: Heikin-Ashi Transformation
**Files to create:**
- `src/gpu/candles/heikin_ashi.rs` - Heikin-Ashi smoothed candles

**Deliverables:**
- `HeikinAshiAggregator` implementation
- CUDA kernel: `persistent_heikin_ashi_kernel`
- Transform existing OHLC → smoothed HA candles
- Batch type: `HeikinAshiBatch`

---

### Agent 4: Volume & Tick Bars
**Files to create:**
- `src/gpu/candles/volume_bars.rs` - Volume-based aggregation
- `src/gpu/candles/tick_bars.rs` - Tick count-based aggregation

**Deliverables:**
- `VolumeBarAggregator` (fixed volume per bar)
- `TickBarAggregator` (fixed trades per bar)
- Two CUDA kernels
- Batch types for both

---

### Agent 5: Range & Renko Bars
**Files to create:**
- `src/gpu/candles/range_bars.rs` - Fixed price range bars
- `src/gpu/candles/renko.rs` - Renko bricks

**Deliverables:**
- `RangeBarAggregator` (fixed price movement)
- `RenkoAggregator` (brick-based charting)
- Two CUDA kernels
- Batch types for both

---

### Agent 6: CSV Ingestion Pipeline
**Files to create:**
- `src/gpu/candles/csv_loader.rs` - Trade data CSV parsing
- `src/gpu/candles/batch_builder.rs` - Batch construction helpers

**Deliverables:**
- `TradeData::from_csv()` implementation
- Multi-symbol CSV loading
- Efficient CSV parsing (polars or csv crate)
- Memory-efficient streaming for large files

---

### Agent 7: Comprehensive Tests
**Files to create:**
- `tests/candles/test_time_bars.rs`
- `tests/candles/test_heikin_ashi.rs`
- `tests/candles/test_volume_tick_bars.rs`
- `tests/candles/test_range_renko.rs`
- `tests/candles/test_csv_loader.rs`
- `examples/candles_full_demo.rs`

**Deliverables:**
- Unit tests for all 6 candle types
- Integration tests with real trade data
- Validation against known-good implementations
- Example with full pipeline

---

### Agent 8: Documentation & Examples
**Files to create:**
- `examples/time_bars_from_csv.rs` - Simple time bar example
- `examples/multi_symbol_batch.rs` - Batch processing example
- `examples/heikin_ashi_strategy.rs` - Strategy using HA candles
- `docs/CANDLES_API.md` - API documentation
- `docs/CANDLES_BENCHMARKS.md` - Performance results

**Deliverables:**
- 3+ runnable examples
- API documentation
- Performance benchmarks
- Usage guides

---

## Shared Context for All Agents

### Project Structure
```
src/gpu/candles/
├── mod.rs              # Agent 1
├── types.rs            # Agent 1
├── traits.rs           # Agent 1
├── time_bars.rs        # Agent 2
├── heikin_ashi.rs      # Agent 3
├── volume_bars.rs      # Agent 4
├── tick_bars.rs        # Agent 4
├── range_bars.rs       # Agent 5
├── renko.rs            # Agent 5
├── csv_loader.rs       # Agent 6
└── batch_builder.rs    # Agent 6
```

### Integration Requirements
- Follow existing persistent kernel pattern (5-parameter signature)
- Reuse `execute_batch()` from `src/gpu/persistent/mod.rs`
- Support pinned memory
- Dynamic occupancy optimization
- Cooperative grid synchronization

### Code Style
- Rust Edition 2024
- Feature flag: `#[cfg(feature = "gpu")]`
- Documentation with examples
- Type-safe generic batches
- Zero-cost abstractions

---

## Success Criteria

✅ All 6 candle types implemented and tested
✅ CSV ingestion pipeline working
✅ Batch processing for multiple symbols
✅ 3+ runnable examples
✅ Performance benchmarks showing 20-100x speedups
✅ Integration tests passing
✅ Documentation complete

---

## Execution Strategy

1. **Launch all 8 agents in parallel** (maximize throughput)
2. **Agent 1 creates foundation** - others depend on this
3. **Agents 2-5 implement kernels** - can work independently
4. **Agent 6 creates ingestion** - can work independently
5. **Agent 7 creates tests** - starts after agents 2-5 complete
6. **Agent 8 creates docs** - starts after all others complete

**Estimated Time:** 45-60 minutes with 8 parallel agents
