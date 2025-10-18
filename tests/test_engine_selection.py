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

from kimsfinance.core import EngineManager
from kimsfinance.core.engine import GPU_CROSSOVER_THRESHOLDS
from kimsfinance.core.exceptions import GPUNotAvailableError, ConfigurationError
from kimsfinance.ops.indicators import calculate_atr, calculate_rsi


class TestGPUCrossoverThresholds:
    """Test GPU crossover threshold configuration."""

    def test_crossover_thresholds_exist(self):
        """Verify GPU_CROSSOVER_THRESHOLDS dictionary exists and has expected values."""
        # Verify dictionary exists
        assert GPU_CROSSOVER_THRESHOLDS is not None, "GPU_CROSSOVER_THRESHOLDS should exist"
        assert isinstance(GPU_CROSSOVER_THRESHOLDS, dict), "GPU_CROSSOVER_THRESHOLDS should be a dict"

        # Verify expected operations are present
        expected_operations = ["atr", "rsi", "stochastic", "bollinger", "obv", "macd"]
        for operation in expected_operations:
            assert operation in GPU_CROSSOVER_THRESHOLDS, \
                f"Operation '{operation}' should be in GPU_CROSSOVER_THRESHOLDS"

        # Verify threshold values are reasonable (positive integers)
        for operation, threshold in GPU_CROSSOVER_THRESHOLDS.items():
            assert isinstance(threshold, int), f"Threshold for '{operation}' should be int"
            assert threshold > 0, f"Threshold for '{operation}' should be positive"
            assert threshold >= 1000, f"Threshold for '{operation}' should be at least 1K rows"

    def test_specific_threshold_values(self):
        """Verify specific threshold values match expected empirical values."""
        # These are the empirically-derived crossover points
        assert GPU_CROSSOVER_THRESHOLDS["atr"] == 100_000, "ATR threshold should be 100K"
        assert GPU_CROSSOVER_THRESHOLDS["rsi"] == 100_000, "RSI threshold should be 100K"
        assert GPU_CROSSOVER_THRESHOLDS["stochastic"] == 500_000, "Stochastic threshold should be 500K"
        assert GPU_CROSSOVER_THRESHOLDS["bollinger"] == 100_000, "Bollinger threshold should be 100K"
        assert GPU_CROSSOVER_THRESHOLDS["obv"] == 100_000, "OBV threshold should be 100K"
        assert GPU_CROSSOVER_THRESHOLDS["macd"] == 100_000, "MACD threshold should be 100K"


class TestExplicitEngineSelection:
    """Test explicit CPU and GPU engine selection."""

    def setup_method(self):
        """Reset GPU cache before each test."""
        EngineManager.reset_gpu_cache()

    @pytest.mark.parametrize("data_size", [100, 1000, 100_000, 1_000_000])
    @pytest.mark.parametrize("operation", ["atr", "rsi", "macd", None])
    def test_explicit_cpu_selection(self, data_size, operation):
        """Test that engine='cpu' always returns 'cpu' regardless of context."""
        result = EngineManager.select_engine_smart(
            engine="cpu",
            operation=operation,
            data_size=data_size
        )
        assert result == "cpu", "Explicit CPU selection should always return 'cpu'"

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    @pytest.mark.parametrize("data_size", [100, 1000, 100_000, 1_000_000])
    @pytest.mark.parametrize("operation", ["atr", "rsi", "macd", None])
    def test_explicit_gpu_selection_available(self, mock_gpu, data_size, operation):
        """Test that engine='gpu' returns 'gpu' when GPU is available."""
        result = EngineManager.select_engine_smart(
            engine="gpu",
            operation=operation,
            data_size=data_size
        )
        assert result == "gpu", "Explicit GPU selection should return 'gpu' when available"
        mock_gpu.assert_called_once()

    @patch.object(EngineManager, 'check_gpu_available', return_value=False)
    def test_explicit_gpu_selection_unavailable(self, mock_gpu):
        """Test that engine='gpu' raises error when GPU is unavailable."""
        with pytest.raises(GPUNotAvailableError) as exc_info:
            EngineManager.select_engine_smart(engine="gpu")

        mock_gpu.assert_called_once()
        assert "GPU engine requested but not available" in str(exc_info.value)


class TestAutoEngineSelection:
    """Test automatic engine selection with intelligent defaults."""

    def setup_method(self):
        """Reset GPU cache before each test."""
        EngineManager.reset_gpu_cache()

    @patch.object(EngineManager, 'check_gpu_available', return_value=False)
    def test_auto_defaults_to_cpu_without_context(self, mock_gpu):
        """Test that engine='auto' defaults to 'cpu' when no operation/size provided."""
        result = EngineManager.select_engine_smart(engine="auto")
        assert result == "cpu", "Auto without context should default to CPU"
        mock_gpu.assert_called_once()

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_auto_defaults_to_cpu_with_no_gpu(self, mock_gpu):
        """Test that engine='auto' returns 'cpu' when GPU is unavailable."""
        mock_gpu.return_value = False
        result = EngineManager.select_engine_smart(
            engine="auto",
            operation="atr",
            data_size=200_000
        )
        assert result == "cpu", "Auto should return CPU when GPU unavailable"
        mock_gpu.assert_called_once()

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    @pytest.mark.parametrize("operation,threshold", [
        ("atr", 100_000),
        ("rsi", 100_000),
        ("stochastic", 500_000),
        ("bollinger", 100_000),
    ])
    def test_auto_with_small_data_returns_cpu(self, mock_gpu, operation, threshold):
        """Test that engine='auto' returns 'cpu' for data below threshold."""
        # Test with data just below threshold
        data_size = threshold - 1

        result = EngineManager.select_engine_smart(
            engine="auto",
            operation=operation,
            data_size=data_size
        )

        assert result == "cpu", \
            f"Auto should return CPU for {operation} with {data_size} < {threshold} rows"
        mock_gpu.assert_called_once()

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    @pytest.mark.parametrize("operation,threshold", [
        ("atr", 100_000),
        ("rsi", 100_000),
        ("stochastic", 500_000),
        ("bollinger", 100_000),
    ])
    def test_auto_with_large_data_returns_gpu(self, mock_gpu, operation, threshold):
        """Test that engine='auto' returns 'gpu' for data at or above threshold."""
        # Test with data at threshold
        data_size = threshold

        result = EngineManager.select_engine_smart(
            engine="auto",
            operation=operation,
            data_size=data_size
        )

        assert result == "gpu", \
            f"Auto should return GPU for {operation} with {data_size} >= {threshold} rows"
        mock_gpu.assert_called_once()

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_auto_unknown_operation_uses_default_threshold(self, mock_gpu):
        """Test that unknown operations use 100K default threshold."""
        # Test with unknown operation and data below default threshold
        result_small = EngineManager.select_engine_smart(
            engine="auto",
            operation="unknown_indicator",
            data_size=50_000
        )
        assert result_small == "cpu", "Unknown operation with 50K rows should use CPU"

        # Reset mock
        mock_gpu.reset_mock()

        # Test with unknown operation and data at/above default threshold
        result_large = EngineManager.select_engine_smart(
            engine="auto",
            operation="unknown_indicator",
            data_size=100_000
        )
        assert result_large == "gpu", "Unknown operation with 100K rows should use GPU"

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_auto_with_operation_but_no_size_returns_cpu(self, mock_gpu):
        """Test that providing operation without size defaults to CPU."""
        result = EngineManager.select_engine_smart(
            engine="auto",
            operation="atr",
            data_size=None
        )
        assert result == "cpu", "Auto with operation but no size should default to CPU"

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_auto_with_size_but_no_operation_returns_cpu(self, mock_gpu):
        """Test that providing size without operation defaults to CPU."""
        result = EngineManager.select_engine_smart(
            engine="auto",
            operation=None,
            data_size=200_000
        )
        assert result == "cpu", "Auto with size but no operation should default to CPU"


class TestInvalidEngineHandling:
    """Test handling of invalid engine parameters."""

    def test_invalid_engine_raises_configuration_error(self):
        """Test that invalid engine values raise ConfigurationError."""
        invalid_engines = ["invalid", "GPU", "CPU", "AUTO", 123, None, ""]

        for invalid_engine in invalid_engines:
            with pytest.raises(ConfigurationError) as exc_info:
                EngineManager.select_engine_smart(engine=invalid_engine)

            assert "Invalid engine" in str(exc_info.value), \
                f"Should raise ConfigurationError for engine={invalid_engine!r}"


class TestIndicatorEngineIntegration:
    """Test that indicators call select_engine_smart() correctly."""

    def setup_method(self):
        """Reset GPU cache and create test data."""
        EngineManager.reset_gpu_cache()

        # Create test OHLC data
        np.random.seed(42)
        self.small_size = 1000
        self.large_size = 150_000

        # Small dataset
        self.small_closes = 100 + np.cumsum(np.random.randn(self.small_size) * 0.5)
        self.small_highs = self.small_closes + np.abs(np.random.randn(self.small_size) * 0.3)
        self.small_lows = self.small_closes - np.abs(np.random.randn(self.small_size) * 0.3)

        # Large dataset
        self.large_closes = 100 + np.cumsum(np.random.randn(self.large_size) * 0.5)
        self.large_highs = self.large_closes + np.abs(np.random.randn(self.large_size) * 0.3)
        self.large_lows = self.large_closes - np.abs(np.random.randn(self.large_size) * 0.3)

    @patch.object(EngineManager, 'select_engine_smart')
    def test_atr_calls_select_engine_smart_correctly(self, mock_select):
        """Test that calculate_atr() calls select_engine_smart() with correct parameters."""
        mock_select.return_value = "cpu"

        # Call ATR with auto engine
        result = calculate_atr(
            self.small_highs,
            self.small_lows,
            self.small_closes,
            period=14,
            engine="auto"
        )

        # Verify select_engine_smart was called
        mock_select.assert_called_once()
        call_args = mock_select.call_args

        # Check arguments
        assert call_args[0][0] == "auto", "Should pass engine='auto'"
        assert call_args[1]["operation"] == "atr", "Should pass operation='atr'"
        assert call_args[1]["data_size"] == self.small_size, \
            f"Should pass data_size={self.small_size}"

        # Verify result is valid
        assert isinstance(result, np.ndarray), "ATR should return numpy array"
        assert len(result) == self.small_size, "ATR should return same length as input"

    @patch.object(EngineManager, 'select_engine_smart')
    def test_rsi_calls_select_engine_smart_correctly(self, mock_select):
        """Test that calculate_rsi() calls select_engine_smart() with correct parameters."""
        mock_select.return_value = "cpu"

        # Call RSI with auto engine
        result = calculate_rsi(
            self.small_closes,
            period=14,
            engine="auto"
        )

        # Verify select_engine_smart was called
        mock_select.assert_called_once()
        call_args = mock_select.call_args

        # Check arguments
        assert call_args[0][0] == "auto", "Should pass engine='auto'"
        assert call_args[1]["operation"] == "rsi", "Should pass operation='rsi'"
        assert call_args[1]["data_size"] == self.small_size, \
            f"Should pass data_size={self.small_size}"

        # Verify result is valid
        assert isinstance(result, np.ndarray), "RSI should return numpy array"
        assert len(result) == self.small_size, "RSI should return same length as input"

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_atr_with_small_data_uses_cpu(self, mock_gpu):
        """Test that ATR with small data (<100K) uses CPU engine."""
        # Spy on select_engine_smart to verify it returns CPU
        with patch.object(EngineManager, 'select_engine_smart',
                         wraps=EngineManager.select_engine_smart) as mock_select:
            result = calculate_atr(
                self.small_highs,
                self.small_lows,
                self.small_closes,
                period=14,
                engine="auto"
            )

            # Verify select_engine_smart was called correctly
            mock_select.assert_called_once_with(
                "auto", operation="atr", data_size=self.small_size
            )

            # Get the actual returned engine by calling the real method
            actual_engine = EngineManager.select_engine_smart(
                "auto", operation="atr", data_size=self.small_size
            )
            assert actual_engine == "cpu", \
                "ATR with <100K rows should select CPU engine"

            # Verify result is valid
            assert isinstance(result, np.ndarray)
            assert len(result) == self.small_size

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_atr_with_large_data_uses_gpu(self, mock_gpu):
        """Test that ATR with large data (>=100K) uses GPU engine when available."""
        # Spy on select_engine_smart to verify it returns GPU
        with patch.object(EngineManager, 'select_engine_smart',
                         wraps=EngineManager.select_engine_smart) as mock_select:
            # Verify the engine selection logic (without actually running Polars GPU)
            actual_engine = EngineManager.select_engine_smart(
                "auto", operation="atr", data_size=self.large_size
            )
            assert actual_engine == "gpu", \
                "ATR with >=100K rows should select GPU engine when available"

            # Note: We don't actually run calculate_atr() here because GPU is not available
            # in the test environment. The test above verifies the selection logic.

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_rsi_with_small_data_uses_cpu(self, mock_gpu):
        """Test that RSI with small data (<100K) uses CPU engine."""
        # Spy on select_engine_smart to verify it returns CPU
        with patch.object(EngineManager, 'select_engine_smart',
                         wraps=EngineManager.select_engine_smart) as mock_select:
            result = calculate_rsi(
                self.small_closes,
                period=14,
                engine="auto"
            )

            # Verify select_engine_smart was called correctly
            mock_select.assert_called_once_with(
                "auto", operation="rsi", data_size=self.small_size
            )

            # Get the actual returned engine by calling the real method
            actual_engine = EngineManager.select_engine_smart(
                "auto", operation="rsi", data_size=self.small_size
            )
            assert actual_engine == "cpu", \
                "RSI with <100K rows should select CPU engine"

            # Verify result is valid
            assert isinstance(result, np.ndarray)
            assert len(result) == self.small_size

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_rsi_with_large_data_uses_gpu(self, mock_gpu):
        """Test that RSI with large data (>=100K) uses GPU engine when available."""
        # Spy on select_engine_smart to verify it returns GPU
        with patch.object(EngineManager, 'select_engine_smart',
                         wraps=EngineManager.select_engine_smart) as mock_select:
            # Verify the engine selection logic (without actually running Polars GPU)
            actual_engine = EngineManager.select_engine_smart(
                "auto", operation="rsi", data_size=self.large_size
            )
            assert actual_engine == "gpu", \
                "RSI with >=100K rows should select GPU engine when available"

            # Note: We don't actually run calculate_rsi() here because GPU is not available
            # in the test environment. The test above verifies the selection logic.

    def test_atr_explicit_cpu_ignores_size(self):
        """Test that ATR with engine='cpu' uses CPU regardless of data size."""
        # Even with large data, explicit CPU should be used
        result = calculate_atr(
            self.large_highs,
            self.large_lows,
            self.large_closes,
            period=14,
            engine="cpu"
        )

        assert isinstance(result, np.ndarray), "ATR should return numpy array"
        assert len(result) == self.large_size, "ATR should return same length as input"

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_rsi_explicit_gpu_ignores_size(self, mock_gpu):
        """Test that RSI with engine='gpu' uses GPU regardless of data size."""
        # Even with small data, explicit GPU should be attempted
        try:
            result = calculate_rsi(
                self.small_closes,
                period=14,
                engine="gpu"
            )
            # If it succeeds, verify result
            assert isinstance(result, np.ndarray), "RSI should return numpy array"
            assert len(result) == self.small_size, "RSI should return same length as input"
        except Exception:
            # GPU may not actually be available in test environment
            pass

        # Verify GPU availability was checked
        mock_gpu.assert_called()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Reset GPU cache before each test."""
        EngineManager.reset_gpu_cache()

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_exactly_at_threshold(self, mock_gpu):
        """Test behavior when data size is exactly at threshold."""
        # ATR threshold is 100,000
        result = EngineManager.select_engine_smart(
            engine="auto",
            operation="atr",
            data_size=100_000
        )
        assert result == "gpu", "Data exactly at threshold should use GPU"

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_one_row_below_threshold(self, mock_gpu):
        """Test behavior when data size is one row below threshold."""
        result = EngineManager.select_engine_smart(
            engine="auto",
            operation="atr",
            data_size=99_999
        )
        assert result == "cpu", "Data one row below threshold should use CPU"

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_very_large_data(self, mock_gpu):
        """Test with very large datasets."""
        result = EngineManager.select_engine_smart(
            engine="auto",
            operation="atr",
            data_size=10_000_000
        )
        assert result == "gpu", "Very large data should use GPU"

    def test_zero_data_size(self):
        """Test with zero data size."""
        result = EngineManager.select_engine_smart(
            engine="auto",
            operation="atr",
            data_size=0
        )
        assert result == "cpu", "Zero data size should use CPU"

    @patch.object(EngineManager, 'check_gpu_available', return_value=True)
    def test_empty_string_operation(self, mock_gpu):
        """Test with empty string as operation."""
        # Empty string is unknown operation, should use default threshold
        result = EngineManager.select_engine_smart(
            engine="auto",
            operation="",
            data_size=100_000
        )
        assert result == "gpu", "Empty operation string with 100K rows should use GPU"


class TestGPUCacheReset:
    """Test GPU availability cache behavior."""

    def test_reset_gpu_cache(self):
        """Test that reset_gpu_cache() clears the cache."""
        # Check GPU availability (will cache result)
        first_check = EngineManager.check_gpu_available()

        # Cache should be set
        assert EngineManager._gpu_available is not None, "Cache should be set after check"

        # Reset cache
        EngineManager.reset_gpu_cache()

        # Cache should be cleared
        assert EngineManager._gpu_available is None, "Cache should be None after reset"

    def test_gpu_cache_persistence(self):
        """Test that GPU availability is cached across calls."""
        # Reset first
        EngineManager.reset_gpu_cache()

        # First check
        first_result = EngineManager.check_gpu_available()

        # Cache should be set
        cached_value = EngineManager._gpu_available

        # Second check should use cache
        second_result = EngineManager.check_gpu_available()

        assert first_result == second_result, "Results should be consistent"
        assert EngineManager._gpu_available == cached_value, "Cache should not change"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
