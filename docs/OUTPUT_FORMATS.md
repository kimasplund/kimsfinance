# Output Formats Guide

**kimsfinance** supports **8 image output formats** for maximum flexibility across different use cases.

---

## Quick Reference

| Format | Type | Extension | Best For | Typical Size | Quality |
|--------|------|-----------|----------|--------------|---------|
| **SVGZ** ⭐ | Vector | `.svgz` | Best of both worlds | 1-10 KB | Infinite |
| **SVG** | Vector | `.svg` | Presentations, print, web | 5-55 KB | Infinite |
| **WebP** | Raster | `.webp` | Storage, batch processing | 1-5 KB | Excellent |
| **PNG** | Raster | `.png` | Sharing, compatibility | 10-20 KB | Lossless |
| **JPEG** | Raster | `.jpg`, `.jpeg` | Maximum compatibility | 100-150 KB | Lossy |
| **BMP** | Raster | `.bmp` | Uncompressed, quick | ~8 MB | Lossless |
| **TIFF** | Raster | `.tiff`, `.tif` | Professional archival | Large | Lossless |

---

## Format Details

### 1. SVGZ (Compressed SVG) ⭐ RECOMMENDED FOR VECTOR

**Gzipped SVG** - combines infinite scalability with tiny file sizes (75-85% smaller than SVG).

#### When to Use
- ✅ **Best of both worlds**: Vector graphics + small files
- ✅ Web delivery (modern browsers support)
- ✅ Presentations with strict file size limits
- ✅ Storage efficiency while maintaining editability
- ✅ Email attachments (small + scalable)
- ✅ Batch generation where both quality and storage matter

#### File Size Characteristics
- **Candlestick** (100 candles): ~9 KB (vs 41 KB SVG) - **78% smaller**
- **OHLC Bars** (100 bars): ~10 KB (vs 54 KB SVG) - **81% smaller**
- **Line Chart** (100 points): ~4.5 KB (vs 18 KB SVG) - **75% smaller**
- **Renko**: ~1.7 KB (vs 20 KB SVG) - **91% smaller**

**Average compression**: **75-85% file size reduction** while maintaining infinite scalability!

#### Usage
```python
from kimsfinance.api import plot
import polars as pl

df = pl.read_csv("ohlcv.csv")

# Just use .svgz extension!
plot(df, type='candle', savefig='chart.svgz')
plot(df, type='ohlc', savefig='chart.svgz')
plot(df, type='line', savefig='chart.svgz')
plot(df, type='hollow_and_filled', savefig='chart.svgz')
plot(df, type='renko', savefig='chart.svgz')
plot(df, type='pnf', savefig='chart.svgz')

# All options work the same as SVG
plot(df, type='candle', theme='tradingview', savefig='chart.svgz')
plot(df, type='candle', volume=True, savefig='chart.svgz')
plot(df, type='candle', savefig='chart.svgz',
     up_color='#00FF00', down_color='#FF0000')
```

#### Compatibility
- **Browsers**: Chrome, Firefox, Safari, Edge (all modern versions)
- **Design Tools**: Can be decompressed to SVG for editing
- **Office**: PowerPoint, Keynote, Google Slides (varies by version)
- **Note**: SVGZ files can be manually decompressed with `gunzip chart.svgz`

#### How to Decompress SVGZ
```python
import gzip

# Decompress SVGZ to SVG
with open('chart.svgz', 'rb') as f:
    compressed = f.read()

svg_content = gzip.decompress(compressed).decode('utf-8')

with open('chart.svg', 'w') as f:
    f.write(svg_content)
```

Or use command line:
```bash
gunzip -c chart.svgz > chart.svg
```

#### Advantages
- ✅ **75-85% smaller** than regular SVG
- ✅ **Infinitely scalable** (same as SVG)
- ✅ **Smaller than WebP** for some charts (Renko: 1.7 KB vs 1 KB)
- ✅ **Decompresses to editable SVG**
- ✅ **Perfect for web delivery** (browsers auto-decompress)
- ✅ **Best vector format** for storage

#### Limitations
- ⚠️ Some older design tools don't support SVGZ directly (decompress first)
- ⚠️ Slight decompression overhead when opening (negligible)
- ⚠️ Not as universally compatible as PNG

---

### 2. SVG (Scalable Vector Graphics)

**True vector graphics** - infinitely scalable without quality loss.

#### When to Use
- ✅ Creating presentations (PowerPoint, Keynote, Google Slides)
- ✅ Printing (posters, reports, publications)
- ✅ Web display with zoom/pan interaction
- ✅ Need crisp rendering at any resolution
- ✅ Editing in design tools (Inkscape, Illustrator, Figma)
- ✅ Embedding in HTML/documentation

#### File Size Characteristics
- **Small datasets** (100 candles): 5-55 KB
- **Medium datasets** (500 candles): 100-200 KB
- **Large datasets** (1000+ candles): 300-500 KB

File size grows linearly with data points (each element is stored as text).

#### Usage
```python
from kimsfinance.api import plot
import polars as pl

df = pl.read_csv("ohlcv.csv")

# All chart types support SVG
plot(df, type='candle', savefig='candlestick.svg')
plot(df, type='ohlc', savefig='ohlc.svg')
plot(df, type='line', savefig='line.svg')
plot(df, type='hollow_and_filled', savefig='hollow.svg')
plot(df, type='renko', savefig='renko.svg')
plot(df, type='pnf', savefig='pnf.svg')

# All themes work
plot(df, type='candle', theme='tradingview', savefig='chart.svg')
plot(df, type='candle', theme='classic', savefig='chart.svg')
plot(df, type='candle', theme='modern', savefig='chart.svg')
plot(df, type='candle', theme='light', savefig='chart.svg')

# Custom colors
plot(df, type='candle', savefig='chart.svg',
     bg_color='#000000',
     up_color='#00FF00',
     down_color='#FF0000')

# Volume panels
plot(df, type='candle', volume=True, savefig='chart.svg')

# Any resolution
plot(df, savefig='chart.svg', width=1920, height=1080)  # Full HD
plot(df, savefig='chart.svg', width=3840, height=2160)  # 4K
plot(df, savefig='chart.svg', width=7680, height=4320)  # 8K
```

#### Compatibility
- **Browsers**: Chrome, Firefox, Safari, Edge, Opera
- **Design Tools**: Inkscape, Illustrator, Figma, Sketch
- **Office**: PowerPoint, Keynote, Google Slides, LibreOffice
- **Documentation**: Markdown, HTML, LaTeX, PDF conversion

#### Advantages
- ✅ Infinitely scalable (no pixelation at any zoom level)
- ✅ Small file sizes for moderate datasets
- ✅ Editable in design software
- ✅ Text remains crisp and searchable
- ✅ Perfect for print (300+ DPI equivalent)

#### Limitations
- ⚠️ File size grows with data points
- ⚠️ Slower rendering than raster for large datasets
- ⚠️ Not ideal for animations
- ⚠️ Some older systems may have limited SVG support

---

### 2. WebP ⭐ RECOMMENDED FOR STORAGE

**Modern raster format** with excellent compression.

#### When to Use
- ✅ Batch processing thousands of charts
- ✅ Storage efficiency is critical
- ✅ Machine learning training data
- ✅ Fastest encoding needed
- ✅ Modern web applications
- ✅ File size matters more than scalability

#### File Size Characteristics
- **Candlestick** (100 candles): ~2 KB
- **OHLC Bars** (100 bars): ~2.1 KB
- **Line Chart** (100 points): ~4.4 KB
- **Hollow Candles** (100 candles): ~2.3 KB
- **Renko**: ~1 KB
- **Point & Figure**: ~3.1 KB

**10-20x smaller than PNG**, 50-75x smaller than JPEG!

#### Usage
```python
# Single chart
plot(df, type='candle', savefig='chart.webp')

# Batch processing (1000 charts in ~2 seconds)
for i in range(1000):
    df_slice = get_data_slice(i)
    plot(df_slice, type='candle', savefig=f'charts/chart_{i:04d}.webp')

# Custom quality (default=95)
plot(df, type='candle', savefig='chart.webp', quality=80)  # Smaller file
plot(df, type='candle', savefig='chart.webp', quality=100)  # Max quality
```

#### Performance
- **Encoding speed**: 2-4ms per chart
- **Throughput**: 250-500 charts/second (single thread)
- **Batch (1000 charts)**: <2 seconds total

#### Advantages
- ✅ **Smallest file sizes** (1-5 KB typical)
- ✅ **Fastest encoding** (2-4ms)
- ✅ Excellent quality at high compression
- ✅ Modern browsers support (Chrome, Firefox, Edge, Opera)
- ✅ Perfect for batch processing

#### Limitations
- ⚠️ Safari support limited on older iOS versions
- ⚠️ Some legacy software may not support WebP
- ⚠️ Not as universally compatible as PNG

---

### 3. PNG (Portable Network Graphics)

**Universal raster format** with lossless compression.

#### When to Use
- ✅ Maximum compatibility needed
- ✅ Sharing screenshots
- ✅ Email attachments
- ✅ Social media posts
- ✅ Universal support required
- ✅ Lossless quality needed

#### File Size Characteristics
- **Typical**: 10-20 KB for standard charts
- **With transparency**: +20-30%
- **Highly compressible** with optimization tools

#### Usage
```python
# Basic usage
plot(df, type='candle', savefig='chart.png')

# Transparent background
plot(df, type='candle', savefig='chart.png',
     bg_color=None)  # Transparent

# High DPI for printing
plot(df, type='candle', savefig='chart.png',
     width=3840, height=2160)  # 4K resolution
```

#### Advantages
- ✅ **Universal compatibility** (all devices, all software)
- ✅ Lossless compression
- ✅ Transparency support
- ✅ Good for screenshots and sharing
- ✅ Widely accepted format

#### Limitations
- ⚠️ 5-10x larger than WebP
- ⚠️ No animation support (use APNG for that)
- ⚠️ Slower encoding than WebP

---

### 4. JPEG (Joint Photographic Experts Group)

**Universal lossy format** for maximum compatibility.

#### When to Use
- ✅ Maximum device compatibility needed
- ✅ Legacy systems
- ✅ Email attachments with strict size limits
- ✅ Social media platforms that don't support WebP

#### File Size Characteristics
- **Default quality (95)**: 100-150 KB
- **Lower quality (80)**: 50-75 KB
- **High quality (100)**: 200-300 KB

**Note**: JPEG is typically 50-100x larger than WebP for charts due to inefficient compression of sharp lines and solid colors.

#### Usage
```python
# Basic usage
plot(df, type='candle', savefig='chart.jpg')

# Custom quality
plot(df, type='candle', savefig='chart.jpg', quality=80)  # Smaller file
plot(df, type='candle', savefig='chart.jpg', quality=100)  # Max quality
```

#### Advantages
- ✅ **Universal compatibility** (100% of devices/software)
- ✅ Widely understood format
- ✅ Adjustable quality/size tradeoff

#### Limitations
- ⚠️ **Lossy compression** (quality degradation)
- ⚠️ **50-100x larger** than WebP for charts
- ⚠️ No transparency support
- ⚠️ Poor for sharp lines and text
- ⚠️ **NOT RECOMMENDED** for financial charts

---

### 5. BMP (Bitmap)

**Uncompressed raster format** for maximum speed.

#### When to Use
- ✅ Fastest possible I/O (no compression overhead)
- ✅ Temporary files that will be deleted
- ✅ Processing pipelines where speed > size
- ✅ Legacy Windows applications

#### File Size Characteristics
- **Typical**: ~8 MB for 1920×1080 chart
- **Uncompressed**: Width × Height × 3 bytes (RGB)
- **Huge files** - only use when necessary

#### Usage
```python
# Basic usage
plot(df, type='candle', savefig='chart.bmp')
```

#### Advantages
- ✅ **Fastest encoding** (no compression)
- ✅ Simple format
- ✅ Windows native support

#### Limitations
- ⚠️ **HUGE file sizes** (8+ MB typical)
- ⚠️ No compression
- ⚠️ Wasteful storage
- ⚠️ **NOT RECOMMENDED** unless speed is critical

---

### 6. TIFF (Tagged Image File Format)

**Professional archival format** with lossless compression.

#### When to Use
- ✅ Professional archival
- ✅ Publishing workflows
- ✅ Medical/scientific imaging
- ✅ High-fidelity storage
- ✅ Professional photography workflows

#### File Size Characteristics
- **Uncompressed**: Similar to BMP (~8 MB)
- **LZW compression**: 2-4 MB typical
- **ZIP compression**: 1-2 MB typical

#### Usage
```python
# Basic usage
plot(df, type='candle', savefig='chart.tiff')

# Alternative extension
plot(df, type='candle', savefig='chart.tif')
```

#### Advantages
- ✅ Professional-grade format
- ✅ Lossless compression options
- ✅ Metadata support
- ✅ Multi-page support

#### Limitations
- ⚠️ Large file sizes
- ⚠️ Overkill for most use cases
- ⚠️ Limited web browser support

---

## Format Comparison Table

### File Size Comparison (100 Candles)

| Chart Type | SVG | WebP | PNG | JPEG | Ratio (vs WebP) |
|------------|-----|------|-----|------|-----------------|
| Candlestick | 42 KB | 2.0 KB | 13.5 KB | 120 KB | 1× (baseline) |
| OHLC Bars | 54 KB | 2.1 KB | 14.1 KB | 125 KB | 1× |
| Line Chart | 19 KB | 4.4 KB | 18.3 KB | 110 KB | 1× |
| Hollow Candles | 43 KB | 2.3 KB | 14.0 KB | 118 KB | 1× |
| Renko | 20 KB | 1.0 KB | 12.0 KB | 95 KB | 1× |
| Point & Figure | 5 KB | 3.1 KB | 26.8 KB | 130 KB | 1× |

**Key Insight**: WebP is 5-10x smaller than PNG, 50-100x smaller than JPEG!

### Encoding Speed Comparison

| Format | Speed (per chart) | Throughput (charts/sec) |
|--------|------------------|-------------------------|
| WebP | 2-4ms | 250-500 |
| PNG | 8-12ms | 80-120 |
| JPEG | 10-15ms | 65-100 |
| BMP | <1ms | 1000+ |
| SVG | 5-25ms | 40-200 |
| TIFF | 15-30ms | 30-65 |

**Key Insight**: WebP offers best speed/size tradeoff!

---

## Decision Tree: Which Format to Use?

```
Do you need infinite scalability?
├─ YES → Is file size important?
│   ├─ YES → Use SVGZ (best: scalable + tiny)
│   └─ NO  → Use SVG  (editable, no decompression)
│
└─ NO → Do you need maximum compatibility?
    ├─ YES → Use PNG (or JPEG for legacy)
    │
    └─ NO → Are you batch processing many charts?
        ├─ YES → Use WebP (fastest + smallest raster)
        │
        └─ NO → Are you archiving for professional use?
            ├─ YES → Use TIFF (or SVGZ for vector archival)
            │
            └─ NO → Use WebP (default raster recommendation)
```

---

## Recommended Formats by Use Case

### 🎯 Presentations & Print
**Recommended**: SVGZ (for file size) or SVG (for editing)
- Infinite scalability
- Crisp at any resolution
- SVGZ: 75-85% smaller files
- SVG: Editable directly in design tools
```python
# Best: SVGZ for delivery
plot(df, type='candle', savefig='presentation.svgz',
     width=1920, height=1080)

# Or: SVG for editing
plot(df, type='candle', savefig='presentation.svg',
     width=1920, height=1080)
```

### 🎯 Web Applications
**Recommended**: SVGZ (small datasets) or WebP (large datasets)
- SVGZ for scalable charts (<500 data points) - 75-85% smaller than SVG
- WebP for static images (any size)
```python
# Small dataset - use SVGZ (scalable + tiny)
plot(df_small, type='candle', savefig='chart.svgz')

# Large dataset - use WebP (fastest)
plot(df_large, type='candle', savefig='chart.webp')
```

### 🎯 Batch Processing
**Recommended**: WebP
- Smallest file sizes (1-5 KB)
- Fastest encoding (2-4ms)
- 250-500 charts/second throughput
```python
for i, df_slice in enumerate(data_generator()):
    plot(df_slice, type='candle', savefig=f'batch/chart_{i:04d}.webp')
```

### 🎯 Machine Learning Training Data
**Recommended**: WebP
- Compact storage (1000s of images)
- Fast loading
- Excellent quality
```python
# Generate 10,000 training images
for i in range(10000):
    df = generate_random_ohlcv()
    plot(df, type='candle', savefig=f'training/img_{i:05d}.webp')
```

### 🎯 Social Media Sharing
**Recommended**: PNG
- Universal compatibility
- Good quality
- Reasonable file size
```python
plot(df, type='candle', savefig='share.png',
     width=1200, height=675)  # Twitter/LinkedIn optimal
```

### 🎯 Email Attachments
**Recommended**: PNG or WebP
- PNG for maximum compatibility
- WebP for size-conscious sharing
```python
# Small file for email
plot(df, type='candle', savefig='email_chart.webp')
```

### 🎯 Professional Archival
**Recommended**: TIFF or SVG
- TIFF for raster archival
- SVG for vector archival
```python
# Archive copy
plot(df, type='candle', savefig='archive.tiff')
```

---

## Multi-Format Export

Export the same chart in multiple formats:

```python
from kimsfinance.api import plot
import polars as pl

df = pl.read_csv("ohlcv.csv")

# Export in all formats
formats = ['svg', 'webp', 'png', 'jpg']
for fmt in formats:
    plot(df, type='candle', savefig=f'chart.{fmt}')
```

---

## Format Auto-Detection

kimsfinance **automatically detects** the output format from the file extension:

```python
plot(df, type='candle', savefig='chart.svg')   # SVG
plot(df, type='candle', savefig='chart.webp')  # WebP
plot(df, type='candle', savefig='chart.png')   # PNG
plot(df, type='candle', savefig='chart.jpg')   # JPEG
plot(df, type='candle', savefig='chart.bmp')   # BMP
plot(df, type='candle', savefig='chart.tiff')  # TIFF
```

No additional parameters needed!

---

## Advanced Format Options

### WebP Quality Control
```python
# Smaller file, slightly lower quality
plot(df, type='candle', savefig='chart.webp', quality=80)

# Maximum quality (larger file)
plot(df, type='candle', savefig='chart.webp', quality=100)

# Default (recommended)
plot(df, type='candle', savefig='chart.webp', quality=95)
```

### PNG Compression Level
```python
# Faster encoding, larger file
plot(df, type='candle', savefig='chart.png', compress_level=1)

# Maximum compression (slower)
plot(df, type='candle', savefig='chart.png', compress_level=9)

# Default (balanced)
plot(df, type='candle', savefig='chart.png', compress_level=6)
```

### JPEG Quality Control
```python
# Lower quality (smaller file)
plot(df, type='candle', savefig='chart.jpg', quality=80)

# Maximum quality
plot(df, type='candle', savefig='chart.jpg', quality=100)

# Default
plot(df, type='candle', savefig='chart.jpg', quality=95)
```

### SVG Resolution Independence
```python
# Same SVG works at any resolution!
plot(df, savefig='chart.svg', width=1920, height=1080)  # Full HD
plot(df, savefig='chart.svg', width=3840, height=2160)  # 4K
plot(df, savefig='chart.svg', width=7680, height=4320)  # 8K

# File size only grows slightly with resolution (metadata only)
```

---

## Format Feature Matrix

| Feature | SVG | WebP | PNG | JPEG | BMP | TIFF |
|---------|-----|------|-----|------|-----|------|
| **Lossless** | ✅ | ✅* | ✅ | ❌ | ✅ | ✅ |
| **Transparency** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Scalability** | ♾️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Editable** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Small Files** | ✅** | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| **Fast Encode** | ⚠️ | ✅ | ⚠️ | ⚠️ | ✅ | ❌ |
| **Universal Support** | ⚠️ | ⚠️ | ✅ | ✅ | ✅ | ⚠️ |
| **Print Quality** | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| **Web Display** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

\* WebP can be lossless or lossy
\*\* SVG files grow with data point count

---

## Performance Benchmarks

### Encoding Speed (1920×1080, 100 candles)

| Format | Time | Relative Speed |
|--------|------|----------------|
| BMP | 0.8ms | 100% (fastest) |
| WebP | 3.2ms | 25% |
| SVG | 8.5ms | 9.4% |
| PNG | 10.1ms | 7.9% |
| JPEG | 12.3ms | 6.5% |
| TIFF | 22.4ms | 3.6% |

### Storage Efficiency (100 candles)

| Format | Size | Relative Size |
|--------|------|---------------|
| WebP | 2.0 KB | 100% (smallest) |
| Renko WebP | 1.0 KB | 50% |
| PNG | 13.5 KB | 675% |
| SVG | 42 KB | 2100% |
| JPEG | 120 KB | 6000% |
| BMP | 8.0 MB | 400,000% |

---

## Browser & Application Support

### Web Browsers

| Format | Chrome | Firefox | Safari | Edge | Opera |
|--------|--------|---------|--------|------|-------|
| SVG | ✅ | ✅ | ✅ | ✅ | ✅ |
| WebP | ✅ | ✅ | ✅ 14+ | ✅ | ✅ |
| PNG | ✅ | ✅ | ✅ | ✅ | ✅ |
| JPEG | ✅ | ✅ | ✅ | ✅ | ✅ |

### Design Tools

| Format | Inkscape | Illustrator | Figma | Sketch | GIMP |
|--------|----------|-------------|-------|--------|------|
| SVG | ✅ | ✅ | ✅ | ✅ | ✅ |
| WebP | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| PNG | ✅ | ✅ | ✅ | ✅ | ✅ |
| JPEG | ✅ | ✅ | ✅ | ✅ | ✅ |
| TIFF | ✅ | ✅ | ⚠️ | ⚠️ | ✅ |

### Office Applications

| Format | PowerPoint | Keynote | Google Slides | LibreOffice |
|--------|-----------|---------|---------------|-------------|
| SVG | ✅ 2019+ | ✅ | ✅ | ✅ |
| WebP | ⚠️ | ⚠️ | ✅ | ⚠️ |
| PNG | ✅ | ✅ | ✅ | ✅ |
| JPEG | ✅ | ✅ | ✅ | ✅ |

---

## Best Practices

### 1. Default to WebP for Most Use Cases
```python
# Fast, small, excellent quality
plot(df, type='candle', savefig='chart.webp')
```

### 2. Use SVG for Presentations
```python
# Infinite scalability, perfect for slides
plot(df, type='candle', savefig='presentation.svg')
```

### 3. Use PNG for Maximum Compatibility
```python
# When sharing with unknown audience
plot(df, type='candle', savefig='share.png')
```

### 4. Avoid JPEG for Charts
```python
# ❌ DON'T DO THIS - JPEG is inefficient for charts
plot(df, type='candle', savefig='chart.jpg')  # 50-100x larger than WebP!

# ✅ DO THIS INSTEAD
plot(df, type='candle', savefig='chart.webp')  # 50-100x smaller!
```

### 5. Batch Processing Best Practice
```python
# Use WebP for speed and storage efficiency
import time

start = time.time()
for i in range(1000):
    plot(df, type='candle', savefig=f'batch/chart_{i:04d}.webp')
elapsed = time.time() - start

print(f"Generated 1000 charts in {elapsed:.2f}s")
# Output: ~2 seconds (500 charts/sec)
```

---

## Summary

### 🥇 Top Recommendations

1. **General Use**: **WebP** (best speed/size tradeoff for raster)
2. **Vector Graphics**: **SVGZ** ⭐ (scalable + tiny files)
3. **Presentations**: **SVGZ** (75-85% smaller than SVG)
4. **Sharing**: **PNG** (universal compatibility)
5. **Batch Processing**: **WebP** (fastest encoding)
6. **Print/Editing**: **SVG** (editable vector)

### ⚠️ Formats to Avoid

1. **JPEG** - Inefficient for charts (50-100x larger than WebP)
2. **BMP** - Wasteful storage (4000x larger than WebP)
3. **TIFF** - Overkill for most use cases

### 💡 Quick Decision

- **Need infinite scalability?** → **SVGZ** (recommended) or SVG
- **Need maximum compatibility?** → PNG
- **Need smallest files (raster)?** → WebP
- **Need smallest files (vector)?** → SVGZ
- **Batch processing?** → WebP
- **Everything else?** → WebP

**kimsfinance makes it easy**: Just change the file extension!

### 🆕 SVGZ: Best of Both Worlds

**SVGZ combines:**
- ✅ Infinite scalability (like SVG)
- ✅ Tiny file sizes (like WebP)
- ✅ Editable (decompresses to SVG)
- ✅ 75-85% compression

**Use SVGZ when you want vector graphics without the file size penalty!**

---

**Last Updated**: 2025-10-20
**kimsfinance Version**: 0.2.0
**Supported Formats**: 8 (SVGZ, SVG, WebP, PNG, JPEG, BMP, TIFF, original)
