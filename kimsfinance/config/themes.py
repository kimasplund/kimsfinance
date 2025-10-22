from ..utils.color_utils import _hex_to_rgba

THEMES = {
    "classic": {"bg": "#000000", "up": "#00FF00", "down": "#FF0000", "grid": "#333333"},
    "modern": {"bg": "#1E1E1E", "up": "#26A69A", "down": "#EF5350", "grid": "#424242"},
    "tradingview": {"bg": "#131722", "up": "#089981", "down": "#F23645", "grid": "#2A2E39"},
    "light": {"bg": "#FFFFFF", "up": "#26A69A", "down": "#EF5350", "grid": "#E0E0E0"},
}

THEMES_RGBA = {
    theme: {
        "bg": _hex_to_rgba(colors["bg"]),
        "up": _hex_to_rgba(colors["up"]),
        "down": _hex_to_rgba(colors["down"]),
        "grid": _hex_to_rgba(colors["grid"], alpha=64),
    }
    for theme, colors in THEMES.items()
}

THEMES_RGB = THEMES
