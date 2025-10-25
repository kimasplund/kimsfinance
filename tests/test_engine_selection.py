#!/usr/bin/env python3
"""
Comprehensive Tests for Intelligent Engine Selection
======================================================

Tests the smart engine selection logic with GPU availability mocking,
threshold-based decisions, and integration with indicator functions.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import polars as pl

from kimsfinance.core import EngineManager
from kimsfinance.core.engine import GPU_CROSSOVER_THRESHOLDS
from kimsfinance.core.exceptions import GPUNotAvailableError, ConfigurationError
from kimsfinance.ops.indicators import calculate_atr, calculate_rsi


class TestGPUCrossoverThresholds:
    """Test GPU crossover threshold configuration."""

    def test_crossover_thresholds_exist(self):
        """Verify GPU_CROSSOVER_THRESHOLDS dictionary exists and has expected values."""
        assert GPU_CROSSOVER_THRESHOLDS is not None
        assert isinstance(GPU_CROSSOVER_THRESHOLDS, dict)
        expected_operations = ["atr", "rsi", "stochastic", "default"]
        for operation in expected_operations:
            assert operation in GPU_CROSSOVER_THRESHOLDS, f"'{operation}' not in thresholds"
        for operation, threshold in GPU_CROSSOVER_THRESHOLDS.items():
            assert isinstance(threshold, int) and threshold > 0

    def test_specific_threshold_values(self):
        """Verify specific threshold values match expected empirical values."""
        if not EngineManager.check_gpu_available():
            pytest.skip("GPU not available")
        # Updated after Phase 1: GPU thresholds moved to centralized config
        assert GPU_CROSSOVER_THRESHOLDS["atr"] == 50_000  # rolling window threshold
        assert GPU_CROSSOVER_THRESHOLDS["rsi"] == 50_000  # vectorizable_simple threshold
        assert GPU_CROSSOVER_THRESHOLDS["stochastic"] == 500_000  # iterative threshold (unchanged)
        assert GPU_CROSSOVER_THRESHOLDS["default"] == 100_000  # default threshold (unchanged)


class TestExplicitEngineSelection:
    """Test explicit CPU and GPU engine selection."""

    def setup_method(self):
        """Reset GPU cache before each test."""
        EngineManager.reset_gpu_cache()

    @pytest.mark.parametrize("data_size", [100, 1_000, 100_000])
    @pytest.mark.parametrize("operation", ["atr", "rsi", None])
    def test_explicit_cpu_selection(self, data_size, operation):
        """Test that engine='cpu' always returns 'cpu'."""
        result = EngineManager.select_engine("cpu", operation=operation, data_size=data_size)
        assert result == "cpu"

    @patch.object(EngineManager, "check_gpu_available", return_value=True)
    @pytest.mark.parametrize("data_size", [100, 1_000, 100_000])
    @pytest.mark.parametrize("operation", ["atr", "rsi", None])
    def test_explicit_gpu_selection_available(self, mock_gpu, data_size, operation):
        """Test that engine='gpu' returns 'gpu' when available."""
        result = EngineManager.select_engine("gpu", operation=operation, data_size=data_size)
        assert result == "gpu"
        mock_gpu.assert_called_once()

    @patch.object(EngineManager, "check_gpu_available", return_value=False)
    def test_explicit_gpu_selection_unavailable(self, mock_gpu):
        """Test that engine='gpu' raises error when GPU is unavailable."""
        with pytest.raises(GPUNotAvailableError):
            EngineManager.select_engine("gpu")
        mock_gpu.assert_called_once()


class TestAutoEngineSelection:
    """Test automatic engine selection."""

    def setup_method(self):
        EngineManager.reset_gpu_cache()

    @patch.object(EngineManager, "check_gpu_available", return_value=True)
    def test_auto_defaults_to_cpu_without_context(self, mock_gpu):
        """Test engine='auto' defaults to 'cpu' without context."""
        result = EngineManager.select_engine("auto")
        assert result == "cpu"
        mock_gpu.assert_called_once()

    @patch.object(EngineManager, "check_gpu_available", return_value=False)
    def test_auto_returns_cpu_with_no_gpu(self, mock_gpu):
        """Test engine='auto' returns 'cpu' when GPU is unavailable."""
        result = EngineManager.select_engine("auto", operation="atr", data_size=200_000)
        assert result == "cpu"
        mock_gpu.assert_called_once()

    @pytest.mark.parametrize(
        "operation,threshold",
        [
            ("atr", 50_000),
            ("rsi", 50_000),
            ("stochastic", 500_000),
        ],
    )
    def test_auto_with_small_data_returns_cpu(self, operation, threshold):
        """Test engine='auto' returns 'cpu' for data below threshold."""
        if not EngineManager.check_gpu_available():
            pytest.skip("GPU not available")
        result = EngineManager.select_engine("auto", operation=operation, data_size=threshold - 1)
        assert result == "cpu"

    @pytest.mark.parametrize(
        "operation,threshold",
        [
            ("atr", 50_000),
            ("rsi", 50_000),
            ("stochastic", 500_000),
        ],
    )
    def test_auto_with_large_data_returns_gpu(self, operation, threshold):
        """Test engine='auto' returns 'gpu' for data at or above threshold."""
        if not EngineManager.check_gpu_available():
            pytest.skip("GPU not available")
        result = EngineManager.select_engine("auto", operation=operation, data_size=threshold)
        assert result == "gpu"

    @patch.object(EngineManager, "check_gpu_available", return_value=True)
    def test_auto_unknown_operation_uses_default_threshold(self, mock_gpu):
        """Test that unknown operations use the default threshold."""
        result_small = EngineManager.select_engine("auto", "unknown", 50_000)
        assert result_small == "cpu"
        result_large = EngineManager.select_engine("auto", "unknown", 100_000)
        assert result_large == "gpu"


class TestInvalidEngineHandling:
    """Test handling of invalid engine parameters."""

    def test_invalid_engine_raises_configuration_error(self):
        """Test that invalid engine values raise ConfigurationError."""
        for engine in ["invalid", None, 123]:
            with pytest.raises(ConfigurationError):
                EngineManager.select_engine(engine)


class TestIndicatorEngineIntegration:
    """Test that indicators call select_engine() correctly."""

    def setup_method(self):
        EngineManager.reset_gpu_cache()
        np.random.seed(42)
        self.small_size = 1000
        self.large_size = 150_000
        self.small_closes = 100 + np.cumsum(np.random.randn(self.small_size))
        self.large_closes = 100 + np.cumsum(np.random.randn(self.large_size))

    @patch.object(EngineManager, "select_engine")
    def test_rsi_calls_select_engine_correctly(self, mock_select):
        """Test that calculate_rsi() calls select_engine() correctly."""
        mock_select.return_value = "cpu"
        calculate_rsi(self.small_closes, period=14, engine="auto")
        mock_select.assert_called_once_with("auto", operation="rsi", data_size=self.small_size)

    @patch("kimsfinance.ops.indicators.pl.LazyFrame.collect")
    def test_rsi_with_small_data_uses_cpu(self, mock_collect):
        """Test that RSI with small data uses the CPU engine."""
        mock_collect.return_value = pl.DataFrame({"rsi": []})
        calculate_rsi(self.small_closes, period=14, engine="auto")
        mock_collect.assert_called_with(engine="cpu")

    @patch("kimsfinance.ops.indicators.pl.LazyFrame.collect")
    @patch.object(EngineManager, "check_gpu_available", return_value=True)
    def test_rsi_with_large_data_uses_gpu(self, mock_gpu, mock_collect):
        """Test that RSI with large data uses the GPU engine."""
        mock_collect.return_value = pl.DataFrame({"rsi": []})
        calculate_rsi(self.large_closes, period=14, engine="auto")
        mock_collect.assert_called_with(engine="gpu")


class TestEdgeCases:
    """Test edge cases."""

    def setup_method(self):
        EngineManager.reset_gpu_cache()

    @patch.object(EngineManager, "check_gpu_available", return_value=True)
    def test_exactly_at_threshold(self, mock_gpu):
        """Test behavior at the exact threshold."""
        # Updated: ATR uses rolling window threshold (50_000)
        result = EngineManager.select_engine("auto", "atr", 50_000)
        assert result == "gpu"

    def test_one_row_below_threshold(self):
        """Test behavior one row below the threshold."""
        if not EngineManager.check_gpu_available():
            pytest.skip("GPU not available")
        result = EngineManager.select_engine("auto", "atr", 49_999)
        assert result == "cpu"

    def test_zero_data_size(self):
        """Test with zero data size."""
        result = EngineManager.select_engine("auto", "atr", 0)
        assert result == "cpu"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
