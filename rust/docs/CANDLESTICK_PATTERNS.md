# Candlestick Pattern Recognition

Comprehensive implementation of 35+ candlestick patterns for technical analysis.

## Overview

kimsfinance_core provides high-performance Rust-accelerated candlestick pattern recognition with Python bindings. The implementation achieves **21-23 million candles/second** throughput, significantly exceeding the 1M/sec target.

## Features

- **35+ Patterns**: Complete coverage of classical candlestick patterns
- **Confidence Scores**: Each detection includes a 0.0-1.0 confidence rating
- **Configurable Sensitivity**: Adjust detection strictness (strict/default/relaxed)
- **Volume Confirmation**: Optional volume-based confidence boosting
- **Batch Processing**: Efficient processing of multiple securities
- **Zero-Allocation Hot Paths**: Optimized for maximum performance

## Pattern Categories

### Bullish Patterns (15)
1. **Hammer** - Small body at top, long lower shadow
2. **Inverted Hammer** - Small body at bottom, long upper shadow
3. **Bullish Engulfing** - Large bullish candle engulfs previous bearish
4. **Piercing Line** - Bullish candle closes above midpoint of previous bearish
5. **Morning Star** - Three-candle reversal (bearish → doji → bullish)
6. **Three White Soldiers** - Three consecutive strong bullish candles
7. **White Marubozu** - Strong bullish candle with no shadows
8. **Three Inside Up** - Harami followed by bullish breakout
9. **Three Outside Up** - Engulfing followed by bullish continuation
10. **Bullish Harami** - Large bearish followed by small bullish inside
11. **Tweezer Bottom** - Two candles with identical lows (reversal)
12. **Rising Three Methods** - Bullish continuation pattern
13. **Dragonfly Doji** - Doji with long lower shadow, no upper
14. **Bullish Kicking** - Gap up with strong bullish candle
15. **Concealing Baby Swallow** - Rare bearish reversal pattern

### Bearish Patterns (15)
1. **Hanging Man** - Small body at top, long lower shadow (in uptrend)
2. **Shooting Star** - Small body at bottom, long upper shadow (in uptrend)
3. **Bearish Engulfing** - Large bearish candle engulfs previous bullish
4. **Dark Cloud Cover** - Bearish candle closes below midpoint of previous bullish
5. **Evening Star** - Three-candle reversal (bullish → doji → bearish)
6. **Three Black Crows** - Three consecutive strong bearish candles
7. **Black Marubozu** - Strong bearish candle with no shadows
8. **Three Inside Down** - Harami followed by bearish breakdown
9. **Three Outside Down** - Engulfing followed by bearish continuation
10. **Bearish Harami** - Large bullish followed by small bearish inside
11. **Tweezer Top** - Two candles with identical highs (reversal)
12. **Falling Three Methods** - Bearish continuation pattern
13. **Gravestone Doji** - Doji with long upper shadow, no lower
14. **Bearish Kicking** - Gap down with strong bearish candle
15. **Identical Three Crows** - Three bearish with same opens

### Neutral Patterns (5)
1. **Doji** - Open equals close (indecision)
2. **Spinning Top** - Small body with upper/lower shadows
3. **High Wave** - Long shadows both sides
4. **Long-Legged Doji** - Doji with long shadows
5. **Rickshaw Man** - Similar to long-legged doji

## Quick Start

### Python API

```python
import kimsfinance_core
import numpy as np

# Historical OHLCV data
open_prices = np.array([100.0, 102.0, 105.0, 103.0])
high = np.array([103.0, 106.0, 108.0, 106.0])
low = np.array([99.0, 101.0, 104.0, 101.0])
close = np.array([102.0, 105.0, 107.0, 102.0])
volume = np.array([1000.0, 1500.0, 2000.0, 1200.0])

# Detect patterns (default config)
patterns = kimsfinance_core.recognize_candlestick_patterns(
    open_prices, high, low, close, volume
)

# Print results
for p in patterns:
    print(f"{p['pattern']} at index {p['index']} "
          f"(type: {p['type']}, confidence: {p['confidence']:.2f})")
```

### Rust API

```rust
use kimsfinance_core::indicators::candlestick::{recognize_patterns, PatternConfig};

let open = vec![100.0, 102.0, 105.0];
let high = vec![103.0, 106.0, 108.0];
let low = vec![99.0, 101.0, 104.0];
let close = vec![102.0, 105.0, 107.0];
let volume = vec![1000.0, 1500.0, 2000.0];

let config = PatternConfig::default();
let patterns = recognize_patterns(&open, &high, &low, &close, &volume, &config);

for detection in patterns {
    println!("{} at index {} (confidence: {:.2})",
             detection.pattern.name(), detection.index, detection.confidence);
}
```

## Configuration

### PatternConfig Presets

```python
# Default (balanced)
patterns = kimsfinance_core.recognize_candlestick_patterns(
    open, high, low, close, volume
)

# Strict (fewer false positives)
patterns_strict = kimsfinance_core.recognize_candlestick_patterns(
    open, high, low, close, volume,
    doji_threshold=0.03,  # Stricter doji detection (default: 0.05)
    shadow_ratio=2.5,     # Stricter hammer/star (default: 2.0)
    body_threshold=0.7,   # Stricter body requirements (default: 0.6)
    min_confidence=0.7    # Higher confidence threshold (default: 0.5)
)

# Relaxed (more detections, more false positives)
patterns_relaxed = kimsfinance_core.recognize_candlestick_patterns(
    open, high, low, close, volume,
    doji_threshold=0.1,   # More lenient doji detection
    shadow_ratio=1.5,     # More lenient hammer/star
    body_threshold=0.5,   # More lenient body requirements
    use_volume=False,     # Don't require volume confirmation
    min_confidence=0.3    # Lower confidence threshold
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `doji_threshold` | float | 0.05 | Body-to-range ratio for doji detection (0.0-1.0) |
| `shadow_ratio` | float | 2.0 | Shadow-to-body ratio for hammer/star patterns |
| `body_threshold` | float | 0.6 | Minimum body percentage for strong candles |
| `use_volume` | bool | True | Use volume confirmation (boosts confidence) |
| `min_confidence` | float | 0.5 | Minimum confidence to report (0.0-1.0) |

## Advanced Usage

### Filter by Pattern Type

```python
# Get all patterns
all_patterns = kimsfinance_core.recognize_candlestick_patterns(
    open, high, low, close, volume
)

# Filter by type
bullish = kimsfinance_core.filter_patterns_by_type(all_patterns, 'bullish')
bearish = kimsfinance_core.filter_patterns_by_type(all_patterns, 'bearish')
neutral = kimsfinance_core.filter_patterns_by_type(all_patterns, 'neutral')

print(f"Bullish: {len(bullish)}, Bearish: {len(bearish)}, Neutral: {len(neutral)}")
```

### Pattern Statistics

```python
patterns = kimsfinance_core.recognize_candlestick_patterns(
    open, high, low, close, volume
)

stats = kimsfinance_core.get_pattern_statistics(patterns)

print(f"Total patterns: {stats['total']}")
print(f"Bullish: {stats['bullish']} ({stats['bullish']/stats['total']*100:.1f}%)")
print(f"Bearish: {stats['bearish']} ({stats['bearish']/stats['total']*100:.1f}%)")
print(f"Neutral: {stats['neutral']} ({stats['neutral']/stats['total']*100:.1f}%)")
print(f"Average confidence: {stats['avg_confidence']:.3f}")

# Most common patterns
for name, count in sorted(stats['pattern_counts'].items(),
                         key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name}: {count} occurrences")
```

### Batch Processing

```python
# Process multiple securities at once
opens = [np.array([...]), np.array([...]), ...]
highs = [np.array([...]), np.array([...]), ...]
lows = [np.array([...]), np.array([...]), ...]
closes = [np.array([...]), np.array([...]), ...]
volumes = [np.array([...]), np.array([...]), ...]

# Batch recognize
results = kimsfinance_core.recognize_candlestick_patterns_batch(
    opens, highs, lows, closes, volumes
)

# results is a list of pattern lists (one per security)
for i, patterns in enumerate(results):
    print(f"Security {i}: {len(patterns)} patterns detected")
```

### List All Available Patterns

```python
all_patterns = kimsfinance_core.get_candlestick_patterns()

# Returns: {'Hammer': 'bullish', 'Doji': 'neutral', ...}
for name, type_ in all_patterns.items():
    print(f"{name}: {type_}")
```

## Performance

### Benchmarks

Measured on Intel i9-13980HX (24 cores):

| Dataset Size | Time | Throughput |
|-------------|------|------------|
| 100 candles | 1.8 µs | 55.5 M/sec |
| 1,000 candles | 18.3 µs | 54.6 M/sec |
| 10,000 candles | 427 µs | 23.4 M/sec |
| 100,000 candles | 4.57 ms | 21.9 M/sec |

**Target**: >1M candles/sec ✅
**Achieved**: 21-23M candles/sec (**21-23x over target!**)

### Configuration Impact

| Configuration | Time (10K candles) | Throughput |
|--------------|-------------------|------------|
| Default | 430 µs | 23.2 M/sec |
| Strict | 414 µs | 24.2 M/sec |
| Relaxed | 451 µs | 22.2 M/sec |

**Note**: Strict config is slightly faster due to fewer false positives.

## Pattern Detection Examples

### Hammer (Bullish Reversal)

```
     │
     │
     │
   ┌─┴─┐
   │ ▓ │  <- Small body at top
   └───┘
     │
     │
     │
     │    <- Long lower shadow (≥2x body)
```

**Characteristics**:
- Small body at top (body_pos > 0.65)
- Long lower shadow (≥2x body length)
- Short upper shadow (<0.5x body)

**Confidence**:
- Base: 0.6
- +0.0-0.3 based on shadow/body ratio

### Bullish Engulfing (Bullish Reversal)

```
  Day 1    Day 2
  ┌───┐
  │ ░ │   ┌─────┐
  │ ░ │   │     │
  │ ░ │   │  ▓  │  <- Large bullish engulfs
  └───┘   │  ▓  │     previous bearish
          │  ▓  │
          └─────┘
```

**Characteristics**:
- Previous candle bearish
- Current candle bullish
- Current body engulfs previous body
- Optional: full engulfment (including shadows)

**Confidence**:
- Body engulfment: 0.7
- Full engulfment: 0.85
- +0.1 if volume >1.5x previous

### Doji (Indecision)

```
     │
     │
   ──┼──  <- Open ≈ Close (body < 5% of range)
     │
     │
```

**Types**:
- **Dragonfly Doji**: Long lower shadow, no upper
- **Gravestone Doji**: Long upper shadow, no lower
- **Long-Legged Doji**: Both shadows long
- **Standard Doji**: Small shadows

## Implementation Details

### Pattern Recognition Algorithm

1. **Pre-processing**: Convert OHLCV arrays to internal `Candle` structures
2. **Single-Candle Scan**: Detect Doji, Hammer, Marubozu, etc.
3. **Two-Candle Scan**: Detect Engulfing, Harami, Piercing Line, etc.
4. **Three-Candle Scan**: Detect Morning/Evening Star, Three Soldiers/Crows, etc.
5. **Five-Candle Scan**: Detect Rising/Falling Three Methods
6. **Confidence Scoring**: Calculate 0.0-1.0 confidence per detection
7. **Filtering**: Apply min_confidence threshold
8. **Volume Confirmation**: Boost confidence if volume supports pattern

### Confidence Scoring

Confidence scores (0.0-1.0) are calculated based on:

- **Pattern Quality**: How well candle geometry matches ideal pattern
- **Shadow/Body Ratios**: Better ratios → higher confidence
- **Engulfment Completeness**: Full engulfment > body-only engulfment
- **Volume Confirmation**: Higher volume on key candles boosts confidence
- **Multi-Candle Coherence**: Consistent behavior across pattern candles

### Performance Optimizations

- **Zero-Allocation Hot Paths**: No heap allocations during pattern detection
- **SIMD-Friendly Layout**: Candle data laid out for vectorization
- **Early Returns**: Skip patterns that can't match based on initial checks
- **Inline Functions**: All hot path functions marked `#[inline]`
- **Cache-Friendly**: Sequential access patterns, minimal branching

## Comparison with LEAN/QuantConnect

| Feature | kimsfinance_core | LEAN/QuantConnect |
|---------|------------------|-------------------|
| **Patterns** | 35 | 30+ |
| **Language** | Rust + Python | C# |
| **Performance** | 21-23 M/sec | Unknown |
| **Confidence Scores** | ✅ 0.0-1.0 | ❌ Binary |
| **Configurable Sensitivity** | ✅ 3 presets + custom | ❌ Fixed |
| **Volume Confirmation** | ✅ Optional | ❌ |
| **Batch Processing** | ✅ Native | ❌ |
| **Python Bindings** | ✅ Native PyO3 | ❌ Requires IronPython |

## Best Practices

### Choosing Configuration

```python
# For live trading (high precision required)
patterns = recognize_candlestick_patterns(
    open, high, low, close, volume,
    doji_threshold=0.03,
    shadow_ratio=2.5,
    body_threshold=0.7,
    min_confidence=0.75  # High threshold
)

# For backtesting/research (explore more signals)
patterns = recognize_candlestick_patterns(
    open, high, low, close, volume,
    # Use default or relaxed settings
)

# For screening (quick overview)
patterns = recognize_candlestick_patterns(
    open, high, low, close, volume,
    min_confidence=0.6  # Medium threshold
)
```

### Combining with Technical Indicators

```python
# Detect patterns
patterns = recognize_candlestick_patterns(open, high, low, close, volume)

# Filter for high-confidence bullish
bullish = [p for p in patterns
           if p['type'] == 'bullish' and p['confidence'] > 0.75]

# Confirm with RSI
rsi = kimsfinance_core.calculate_rsi(close, period=14)

for p in bullish:
    idx = p['index']
    if rsi[idx] < 30:  # Oversold
        print(f"Strong bullish signal: {p['pattern']} at {idx} (RSI: {rsi[idx]:.1f})")
```

## Troubleshooting

### No Patterns Detected

**Possible causes**:
- `min_confidence` threshold too high
- Data quality issues (missing candles, incorrect OHLCV)
- Insufficient data (need at least 3-5 candles for most patterns)

**Solution**:
```python
# Try relaxed config
patterns = recognize_candlestick_patterns(
    open, high, low, close, volume,
    min_confidence=0.3,  # Lower threshold
    doji_threshold=0.1   # More lenient
)

if not patterns:
    print("Still no patterns - check data quality")
```

### Too Many False Positives

**Solution**: Use strict configuration
```python
patterns = recognize_candlestick_patterns(
    open, high, low, close, volume,
    doji_threshold=0.03,
    shadow_ratio=2.5,
    body_threshold=0.7,
    min_confidence=0.7
)
```

### Performance Issues

**For large datasets** (>1M candles):
- Use batch processing for multiple securities
- Process in chunks if memory-constrained
- Expected: ~43-48ms per 1M candles

## Further Reading

- **Candlestick Charting**: Steve Nison - "Japanese Candlestick Charting Techniques"
- **Pattern Analysis**: Thomas Bulkowski - "Encyclopedia of Candlestick Charts"
- **Implementation**: `rust/src/indicators/candlestick.rs` (source code)
- **Benchmarks**: `rust/benches/candlestick_patterns.rs`

## Contributing

Found a bug or want to add a pattern? See CONTRIBUTING.md

## License

See LICENSE file in repository root.
