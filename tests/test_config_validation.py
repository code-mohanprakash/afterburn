"""Tests for configuration validation."""

from __future__ import annotations

import pytest

from afterburn.config import AfterburnConfig, _dict_to_config, _validate_config
from afterburn.exceptions import ConfigError


def test_valid_default_config():
    """Test that the default config is valid."""
    config = AfterburnConfig()
    _validate_config(config)  # Should not raise


def test_valid_custom_config():
    """Test that a valid custom config passes validation."""
    data = {
        "device": "cuda",
        "dtype": "bfloat16",
        "max_memory_gb": 16.0,
        "behaviour": {
            "batch_size": 8,
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "num_samples": 3,
        },
        "report": {
            "format": "json",
            "max_examples": 10,
        },
        "weight_diff": {
            "relative_change_threshold": 0.05,
        },
    }
    config = _dict_to_config(data)
    assert config.device == "cuda"
    assert config.dtype == "bfloat16"
    assert config.max_memory_gb == 16.0


def test_invalid_device():
    """Test that invalid device values are rejected."""
    data = {"device": "tpu"}
    with pytest.raises(ConfigError, match="Invalid device 'tpu'"):
        _dict_to_config(data)


def test_invalid_dtype():
    """Test that invalid dtype values are rejected."""
    data = {"dtype": "float64"}
    with pytest.raises(ConfigError, match="Invalid dtype 'float64'"):
        _dict_to_config(data)


def test_negative_max_memory_gb():
    """Test that negative max_memory_gb is rejected."""
    data = {"max_memory_gb": -5.0}
    with pytest.raises(ConfigError, match="max_memory_gb must be > 0"):
        _dict_to_config(data)


def test_zero_max_memory_gb():
    """Test that zero max_memory_gb is converted to None (unlimited)."""
    data = {"max_memory_gb": 0.0}
    config = _dict_to_config(data)
    # Zero is treated as falsy and converted to None
    assert config.max_memory_gb is None


def test_negative_batch_size():
    """Test that negative batch_size is rejected."""
    data = {"behaviour": {"batch_size": -1}}
    with pytest.raises(ConfigError, match="batch_size must be > 0"):
        _dict_to_config(data)


def test_zero_batch_size():
    """Test that zero batch_size is rejected."""
    data = {"behaviour": {"batch_size": 0}}
    with pytest.raises(ConfigError, match="batch_size must be > 0"):
        _dict_to_config(data)


def test_negative_max_new_tokens():
    """Test that negative max_new_tokens is rejected."""
    data = {"behaviour": {"max_new_tokens": -100}}
    with pytest.raises(ConfigError, match="max_new_tokens must be > 0"):
        _dict_to_config(data)


def test_max_new_tokens_exceeds_limit():
    """Test that max_new_tokens > 8192 is rejected."""
    data = {"behaviour": {"max_new_tokens": 10000}}
    with pytest.raises(ConfigError, match="max_new_tokens must be <= 8192"):
        _dict_to_config(data)


def test_max_new_tokens_at_limit():
    """Test that max_new_tokens = 8192 is accepted."""
    data = {"behaviour": {"max_new_tokens": 8192}}
    config = _dict_to_config(data)
    assert config.behaviour.max_new_tokens == 8192


def test_negative_temperature():
    """Test that negative temperature is rejected."""
    data = {"behaviour": {"temperature": -0.5}}
    with pytest.raises(ConfigError, match="temperature must be >= 0.0 and <= 2.0"):
        _dict_to_config(data)


def test_temperature_exceeds_limit():
    """Test that temperature > 2.0 is rejected."""
    data = {"behaviour": {"temperature": 2.5}}
    with pytest.raises(ConfigError, match="temperature must be >= 0.0 and <= 2.0"):
        _dict_to_config(data)


def test_temperature_at_limits():
    """Test that temperature at boundaries is accepted."""
    data1 = {"behaviour": {"temperature": 0.0}}
    config1 = _dict_to_config(data1)
    assert config1.behaviour.temperature == 0.0

    data2 = {"behaviour": {"temperature": 2.0}}
    config2 = _dict_to_config(data2)
    assert config2.behaviour.temperature == 2.0


def test_zero_num_samples():
    """Test that zero num_samples is rejected."""
    data = {"behaviour": {"num_samples": 0}}
    with pytest.raises(ConfigError, match="num_samples must be > 0"):
        _dict_to_config(data)


def test_negative_num_samples():
    """Test that negative num_samples is rejected."""
    data = {"behaviour": {"num_samples": -1}}
    with pytest.raises(ConfigError, match="num_samples must be > 0"):
        _dict_to_config(data)


def test_invalid_report_format():
    """Test that invalid report format is rejected."""
    data = {"report": {"format": "xml"}}
    with pytest.raises(ConfigError, match="Invalid report format 'xml'"):
        _dict_to_config(data)


def test_valid_report_formats():
    """Test that all valid report formats are accepted."""
    for fmt in ["html", "markdown", "json", "pdf"]:
        data = {"report": {"format": fmt}}
        config = _dict_to_config(data)
        assert config.report.format == fmt


def test_negative_max_examples():
    """Test that negative max_examples is rejected."""
    data = {"report": {"max_examples": -1}}
    with pytest.raises(ConfigError, match="max_examples must be > 0"):
        _dict_to_config(data)


def test_zero_max_examples():
    """Test that zero max_examples is rejected."""
    data = {"report": {"max_examples": 0}}
    with pytest.raises(ConfigError, match="max_examples must be > 0"):
        _dict_to_config(data)


def test_invalid_relative_change_threshold_too_low():
    """Test that relative_change_threshold <= 0 is rejected."""
    data = {"weight_diff": {"relative_change_threshold": 0.0}}
    with pytest.raises(ConfigError, match="relative_change_threshold must be > 0 and < 1"):
        _dict_to_config(data)

    data2 = {"weight_diff": {"relative_change_threshold": -0.01}}
    with pytest.raises(ConfigError, match="relative_change_threshold must be > 0 and < 1"):
        _dict_to_config(data2)


def test_invalid_relative_change_threshold_too_high():
    """Test that relative_change_threshold >= 1 is rejected."""
    data = {"weight_diff": {"relative_change_threshold": 1.0}}
    with pytest.raises(ConfigError, match="relative_change_threshold must be > 0 and < 1"):
        _dict_to_config(data)

    data2 = {"weight_diff": {"relative_change_threshold": 1.5}}
    with pytest.raises(ConfigError, match="relative_change_threshold must be > 0 and < 1"):
        _dict_to_config(data2)


def test_negative_reward_hack_threshold():
    """Test that negative reward_hack thresholds are rejected."""
    data = {
        "reward_hack": {
            "thresholds": {
                "length_bias_cohens_d": -0.5,
                "format_increase_ratio": 2.0,
                "strategy_entropy_drop": 0.3,
                "sycophancy_increase": 0.15,
            }
        }
    }
    with pytest.raises(ConfigError, match="threshold 'length_bias_cohens_d' must be > 0"):
        _dict_to_config(data)


def test_weights_not_summing_to_one():
    """Test that weights not summing to 1.0 are rejected."""
    data = {
        "reward_hack": {
            "weights": {
                "length_bias": 0.30,
                "format_gaming": 0.30,
                "strategy_collapse": 0.30,
                "sycophancy": 0.30,
            }
        }
    }
    with pytest.raises(ConfigError, match="weights must sum to 1.0"):
        _dict_to_config(data)


def test_weights_summing_to_one():
    """Test that weights summing to 1.0 are accepted."""
    data = {
        "reward_hack": {
            "weights": {
                "length_bias": 0.25,
                "format_gaming": 0.25,
                "strategy_collapse": 0.25,
                "sycophancy": 0.25,
            }
        }
    }
    config = _dict_to_config(data)
    assert abs(sum(config.reward_hack.weights.values()) - 1.0) < 0.01


def test_weights_within_tolerance():
    """Test that weights within tolerance (0.01) are accepted."""
    data = {
        "reward_hack": {
            "weights": {
                "length_bias": 0.251,
                "format_gaming": 0.249,
                "strategy_collapse": 0.25,
                "sycophancy": 0.25,
            }
        }
    }
    config = _dict_to_config(data)
    assert abs(sum(config.reward_hack.weights.values()) - 1.0) < 0.01


def test_empty_config_returns_defaults():
    """Test that an empty config dict returns default values."""
    data = {}
    config = _dict_to_config(data)
    assert config.device == "auto"
    assert config.dtype == "float16"
    assert config.max_memory_gb is None
    assert config.behaviour.batch_size == 4
    assert config.behaviour.max_new_tokens == 512
    assert config.report.format == "html"


def test_partial_config():
    """Test that partial configs work with defaults for missing values."""
    data = {
        "device": "cpu",
        "behaviour": {
            "batch_size": 16,
        },
    }
    config = _dict_to_config(data)
    assert config.device == "cpu"
    assert config.behaviour.batch_size == 16
    assert config.behaviour.max_new_tokens == 512  # Default value
    assert config.dtype == "float16"  # Default value
