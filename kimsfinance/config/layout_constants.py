"""
Layout Constants for Chart Rendering
=====================================

Centralized configuration for layout ratios and spacing constants
used throughout the rendering pipeline.

This module eliminates magic numbers scattered across the codebase,
making layout parameters easy to discover, understand, and customize.
"""

# Chart height distribution (must sum to 1.0)
CHART_HEIGHT_RATIO = 0.7  # 70% of total height for price chart
VOLUME_HEIGHT_RATIO = 0.3  # 30% of total height for volume panel

# Spacing and sizing ratios
SPACING_RATIO = 0.2  # Spacing between candles/bars as ratio of candle width
WICK_WIDTH_RATIO = 0.1  # Wick width as ratio of candle body width
TICK_LENGTH_RATIO = 0.4  # OHLC tick length as ratio of bar width

# Centering and positioning
CENTER_OFFSET = 0.5  # Offset for centering elements (50%)
QUARTER_OFFSET = 0.25  # Offset for quarter position (25%)
THREE_QUARTER_OFFSET = 0.75  # Offset for three-quarter position (75%)

# Grid and visual appearance
GRID_LINE_WIDTH = 1.0  # Width of grid lines in pixels
GRID_ALPHA = 0.25  # Opacity/alpha for grid lines (25%)
VOLUME_ALPHA = 0.5  # Opacity/alpha for volume bars (50%)
FILL_AREA_ALPHA = 0.2  # Opacity for filled areas under lines (20%)

# Box size calculations (for Renko/PnF charts)
BOX_SIZE_ATR_MULTIPLIER = 0.75  # Multiplier for ATR when auto-calculating box size
BOX_SIZE_FALLBACK_RATIO = 0.01  # Fallback: 1% of price range for small datasets

# Column/brick layout
COLUMN_BOX_WIDTH_RATIO = 0.8  # Box width as ratio of column width (80%)
BRICK_SPACING_RATIO = 0.1  # Spacing between bricks as ratio of brick width (10%)

# Minimum dimensions (in pixels)
MIN_WICK_WIDTH = 1  # Minimum wick width in pixels
MIN_BODY_HEIGHT = 1.0  # Minimum body height for doji candles (SVG)
MIN_BRICK_HEIGHT = 1  # Minimum brick height for Renko charts (pixels)
MIN_BOX_HEIGHT = 10.0  # Minimum box height for PnF charts (pixels)

# Grid divisions
HORIZONTAL_GRID_DIVISIONS = 10  # Number of horizontal price level divisions
MAX_VERTICAL_GRID_LINES = 20  # Maximum number of vertical time marker lines

# Line chart styling
DEFAULT_LINE_WIDTH = 2  # Default width for line charts

# Typography (placeholder for future use)
DEFAULT_FONT_SIZE = 12
TITLE_FONT_SIZE = 14

# Margins (placeholder for future use)
DEFAULT_MARGIN_LEFT = 60
DEFAULT_MARGIN_RIGHT = 20
DEFAULT_MARGIN_TOP = 40
DEFAULT_MARGIN_BOTTOM = 40
