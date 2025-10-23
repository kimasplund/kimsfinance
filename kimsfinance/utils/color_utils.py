from __future__ import annotations
import re


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """
    Convert hex color string to RGBA tuple with validation.

    Args:
        hex_color: Hex color string (e.g., '#FF0000' or 'FF0000')
        alpha: Alpha channel value (0-255)

    Returns:
        RGBA tuple (r, g, b, a) with values 0-255

    Raises:
        ValueError: If hex_color format is invalid
    """
    # Remove # prefix if present
    hex_color = hex_color.lstrip("#")

    # Validate format: must be 6 or 8 hex characters
    if not re.match(r"^[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$", hex_color):
        raise ValueError(
            f"Invalid hex color format: '{hex_color}'. "
            f"Expected format: '#RRGGBB' or '#RRGGBBAA' (e.g., '#FF0000')"
        )

    # Parse RGB
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(hex_color[6:8], 16) if len(hex_color) == 8 else alpha
    except ValueError as e:
        raise ValueError(f"Failed to parse hex color '{hex_color}': {e}")

    return (r, g, b, a)
