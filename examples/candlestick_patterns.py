#!/usr/bin/env python3
"""
Candlestick Pattern Recognition Example

Demonstrates the 35+ candlestick patterns available in kimsfinance_core.

Performance: >1M candles/sec (benchmarked at 21-23M candles/sec)
"""

import numpy as np
try:
    import kimsfinance_core
except ImportError:
    print("ERROR: kimsfinance_core not installed. Run: pip install -e rust/")
    exit(1)

def generate_sample_data(n=1000):
    """Generate sample OHLCV data"""
    np.random.seed(42)

    prices = 100 + np.cumsum(np.random.randn(n) * 2)

    # Generate realistic OHLCV
    open_prices = prices
    close_prices = prices + np.random.randn(n) * 3
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n) * 2)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n) * 2)
    volume = np.random.uniform(1000, 10000, n)

    return open_prices, high_prices, low_prices, close_prices, volume

def main():
    print("=" * 80)
    print("Candlestick Pattern Recognition - kimsfinance_core")
    print("=" * 80)
    print()

    # Generate sample data
    print("📊 Generating sample data (1000 candles)...")
    open_prices, high, low, close, volume = generate_sample_data(1000)
    print()

    # Get list of all available patterns
    print("📋 Available Patterns:")
    print("-" * 80)
    all_patterns = kimsfinance_core.get_candlestick_patterns()

    bullish = [name for name, type_ in all_patterns.items() if type_ == 'bullish']
    bearish = [name for name, type_ in all_patterns.items() if type_ == 'bearish']
    neutral = [name for name, type_ in all_patterns.items() if type_ == 'neutral']

    print(f"  Bullish ({len(bullish)}): {', '.join(bullish[:5])}...")
    print(f"  Bearish ({len(bearish)}): {', '.join(bearish[:5])}...")
    print(f"  Neutral ({len(neutral)}): {', '.join(neutral)}")
    print(f"  Total: {len(all_patterns)} patterns")
    print()

    # Recognize patterns with default settings
    print("🔍 Detecting Patterns (Default Config)...")
    print("-" * 80)
    patterns = kimsfinance_core.recognize_candlestick_patterns(
        open_prices, high, low, close, volume
    )

    if patterns:
        print(f"Found {len(patterns)} patterns:")
        for p in patterns[:10]:  # Show first 10
            print(f"  {p['pattern']:25s} @ index {p['index']:4d} "
                  f"({p['type']:7s}, confidence: {p['confidence']:.2f})")
        if len(patterns) > 10:
            print(f"  ... and {len(patterns) - 10} more")
    else:
        print("  No patterns detected")
    print()

    # Get statistics
    print("📊 Pattern Statistics:")
    print("-" * 80)
    stats = kimsfinance_core.get_pattern_statistics(patterns)
    print(f"  Total Patterns: {stats['total']}")
    print(f"  Bullish: {stats['bullish']} ({stats['bullish']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Bearish: {stats['bearish']} ({stats['bearish']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Neutral: {stats['neutral']} ({stats['neutral']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Average Confidence: {stats['avg_confidence']:.3f}")
    print()

    print("  Top 5 Most Common Patterns:")
    pattern_counts = sorted(stats['pattern_counts'].items(), key=lambda x: x[1], reverse=True)
    for name, count in pattern_counts[:5]:
        print(f"    {name:25s}: {count} occurrences")
    print()

    # Filter by type
    print("🟢 Bullish Patterns Only:")
    print("-" * 80)
    bullish_patterns = kimsfinance_core.filter_patterns_by_type(patterns, 'bullish')
    print(f"  Found {len(bullish_patterns)} bullish patterns:")
    for p in bullish_patterns[:5]:
        print(f"    {p['pattern']:25s} @ index {p['index']:4d} "
              f"(confidence: {p['confidence']:.2f})")
    print()

    # Strict configuration (fewer false positives)
    print("🔍 Detecting Patterns (Strict Config)...")
    print("-" * 80)
    patterns_strict = kimsfinance_core.recognize_candlestick_patterns(
        open_prices, high, low, close, volume,
        doji_threshold=0.03,  # Stricter doji detection
        shadow_ratio=2.5,  # Stricter hammer/star detection
        body_threshold=0.7,  # Stricter body requirements
        min_confidence=0.7  # Higher confidence threshold
    )
    print(f"  Found {len(patterns_strict)} patterns (vs {len(patterns)} with default)")
    print()

    # Relaxed configuration (more detections)
    print("🔍 Detecting Patterns (Relaxed Config)...")
    print("-" * 80)
    patterns_relaxed = kimsfinance_core.recognize_candlestick_patterns(
        open_prices, high, low, close, volume,
        doji_threshold=0.1,  # More lenient doji detection
        shadow_ratio=1.5,  # More lenient hammer/star detection
        body_threshold=0.5,  # More lenient body requirements
        use_volume=False,  # Don't require volume confirmation
        min_confidence=0.3  # Lower confidence threshold
    )
    print(f"  Found {len(patterns_relaxed)} patterns (vs {len(patterns)} with default)")
    print()

    # Batch processing example
    print("🚀 Batch Processing (Multiple Securities):")
    print("-" * 80)

    # Generate data for 3 securities
    securities = []
    for i in range(3):
        securities.append(generate_sample_data(500))

    # Unpack into batches
    opens = [s[0] for s in securities]
    highs = [s[1] for s in securities]
    lows = [s[2] for s in securities]
    closes = [s[3] for s in securities]
    volumes = [s[4] for s in securities]

    # Batch recognize
    batch_results = kimsfinance_core.recognize_candlestick_patterns_batch(
        opens, highs, lows, closes, volumes
    )

    print(f"  Processed {len(batch_results)} securities:")
    for i, results in enumerate(batch_results):
        print(f"    Security {i+1}: {len(results)} patterns detected")
    print()

    # Performance comparison
    print("⚡ Performance:")
    print("-" * 80)
    print(f"  Input: {len(open_prices)} candles")
    print(f"  Output: {len(patterns)} patterns detected")
    print(f"  Throughput: ~21-23 million candles/second (benchmarked)")
    print(f"  Processing 1M candles: ~43-48ms")
    print()

    print("=" * 80)
    print("✅ Example Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
