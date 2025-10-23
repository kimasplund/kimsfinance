"""
Render Configuration Dataclass
===============================

Centralized configuration for chart rendering operations.
Replaces scattered kwargs with validated, type-safe configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any


@dataclass
class RenderConfig:
    """
    Configuration for chart rendering operations.

    Provides type-safe, validated configuration with sensible defaults.
    Replaces scattered kwargs throughout the rendering pipeline.

    Attributes:
        width: Image width in pixels (100-8192, default 1920)
        height: Image height in pixels (100-8192, default 1080)
        theme: Color theme ('classic', 'modern', 'light', 'tradingview')
        bg_color: Background color override (hex string, optional)
        up_color: Bullish/up color override (hex string, optional)
        down_color: Bearish/down color override (hex string, optional)
        grid: Show grid lines
        volume: Show volume panel
        line_width: Line width for lines and borders (0.1-20.0)
        antialiasing: Antialiasing mode ('fast', 'best', 'none')
        show_grid: Display price/time grid lines (alias for grid)
        enable_antialiasing: Enable antialiasing (boolean, simplified)

    Example:
        >>> config = RenderConfig(width=3840, height=2160, theme='tradingview')
        >>> config.validate()
        >>> chart_params = config.to_dict()
    """

    # Image dimensions
    width: int = 1920
    height: int = 1080

    # Visual theme
    theme: str = "classic"
    bg_color: str | None = None
    up_color: str | None = None
    down_color: str | None = None

    # Display options
    grid: bool = True
    volume: bool = True
    line_width: float = 1.0
    antialiasing: str = "fast"

    # Aliases for compatibility
    show_grid: bool = True
    enable_antialiasing: bool = True

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "RenderConfig":
        """
        Create RenderConfig from keyword arguments.

        Filters kwargs to only include valid RenderConfig fields,
        ignoring unknown parameters for forward compatibility.

        Args:
            **kwargs: Keyword arguments from plot() or render functions

        Returns:
            RenderConfig instance with filtered parameters

        Example:
            >>> config = RenderConfig.from_kwargs(
            ...     width=3840, height=2160, theme='tradingview',
            ...     some_unknown_param='ignored'
            ... )
            >>> config.width
            3840
        """
        # Get valid field names
        valid_keys = {f.name for f in dataclass_fields(cls)}

        # Filter kwargs to only valid fields
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}

        return cls(**filtered)

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid

        Example:
            >>> config = RenderConfig(width=50)  # Too small
            >>> config.validate()
            Traceback (most recent call last):
                ...
            ValueError: width must be 100-8192, got 50
        """
        # Width/height bounds
        if not (100 <= self.width <= 8192):
            raise ValueError(
                f"width must be 100-8192, got {self.width}. " f"Common values: 1920 (HD), 3840 (4K)"
            )

        if not (100 <= self.height <= 8192):
            raise ValueError(
                f"height must be 100-8192, got {self.height}. "
                f"Common values: 1080 (HD), 2160 (4K)"
            )

        # Theme validation
        valid_themes = ("classic", "modern", "light", "tradingview")
        if self.theme not in valid_themes:
            raise ValueError(
                f"Invalid theme: '{self.theme}'. " f"Valid themes: {', '.join(valid_themes)}"
            )

        # Line width validation
        if not (0.1 <= self.line_width <= 20.0):
            raise ValueError(f"line_width must be 0.1-20.0, got {self.line_width}")

        # Antialiasing mode validation
        valid_aa_modes = ("fast", "best", "none")
        if self.antialiasing not in valid_aa_modes:
            raise ValueError(
                f"Invalid antialiasing: '{self.antialiasing}'. "
                f"Valid modes: {', '.join(valid_aa_modes)}"
            )

        # Color validation (if provided)
        if self.bg_color is not None:
            self._validate_color(self.bg_color, "bg_color")
        if self.up_color is not None:
            self._validate_color(self.up_color, "up_color")
        if self.down_color is not None:
            self._validate_color(self.down_color, "down_color")

    def _validate_color(self, color: str, name: str) -> None:
        """Validate hex color string."""
        if not color.startswith("#"):
            raise ValueError(
                f"{name} must start with '#', got '{color}'. " f"Example: '#FF0000' for red"
            )

        # Remove '#' and check hex digits
        hex_part = color[1:]
        valid_lengths = (3, 6, 8)  # RGB, RRGGBB, RRGGBBAA
        if len(hex_part) not in valid_lengths:
            raise ValueError(
                f"{name} must be 3, 6, or 8 hex digits after '#', got {len(hex_part)}. "
                f"Example: '#FF0000' or '#F00'"
            )

        # Validate hex characters
        try:
            int(hex_part, 16)
        except ValueError:
            raise ValueError(
                f"{name} contains invalid hex digits: '{color}'. " f"Use only 0-9, A-F characters"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for renderer functions.

        Returns:
            Dictionary with all configuration values

        Example:
            >>> config = RenderConfig(width=1920, height=1080)
            >>> params = config.to_dict()
            >>> params['width']
            1920
        """
        return {
            "width": self.width,
            "height": self.height,
            "theme": self.theme,
            "bg_color": self.bg_color,
            "up_color": self.up_color,
            "down_color": self.down_color,
            "grid": self.grid,
            "volume": self.volume,
            "line_width": self.line_width,
            "antialiasing": self.antialiasing,
            "show_grid": self.show_grid,
            "enable_antialiasing": self.enable_antialiasing,
        }

    def merge(self, **overrides: Any) -> "RenderConfig":
        """
        Create new RenderConfig with overridden values.

        Args:
            **overrides: Values to override from current config

        Returns:
            New RenderConfig instance with merged values

        Example:
            >>> base = RenderConfig(width=1920)
            >>> hd4k = base.merge(width=3840, height=2160)
            >>> hd4k.width
            3840
        """
        current = self.to_dict()
        current.update(overrides)
        return RenderConfig.from_kwargs(**current)
