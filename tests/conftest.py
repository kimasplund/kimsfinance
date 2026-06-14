def pytest_addoption(parser):
    """Add command-line options for baseline management."""
    parser.addoption(
        "--generate-baselines",
        action="store_true",
        default=False,
        help="Generate new baseline images"
    )
    parser.addoption(
        "--tolerance",
        type=float,
        default=0.01,
        help="Acceptable difference percentage (default: 1%%)"
    )
