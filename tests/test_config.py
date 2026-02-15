"""Tests for configuration loading."""

import pytest
import yaml

from afterburn.config import AfterburnConfig, load_config
from afterburn.exceptions import ConfigError


class TestLoadConfig:
    def test_defaults(self):
        config = AfterburnConfig()
        assert config.device == "auto"
        assert config.weight_diff.enabled is True
        assert config.behaviour.batch_size == 4
        assert config.reward_hack.weights["length_bias"] == 0.25

    def test_load_from_yaml(self, tmp_path):
        config_data = {
            "device": "cpu",
            "behaviour": {
                "suites": ["math"],
                "batch_size": 2,
            },
            "reward_hack": {
                "weights": {
                    "length_bias": 0.5,
                    "format_gaming": 0.2,
                    "strategy_collapse": 0.2,
                    "sycophancy": 0.1,
                }
            },
        }
        config_path = tmp_path / ".afterburn.yaml"
        config_path.write_text(yaml.dump(config_data))

        config = load_config(str(config_path))
        assert config.device == "cpu"
        assert config.behaviour.suites == ["math"]
        assert config.behaviour.batch_size == 2
        assert config.reward_hack.weights["length_bias"] == 0.5

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError):
            load_config("/nonexistent/path.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("{{invalid yaml")

        with pytest.raises(ConfigError):
            load_config(str(config_path))

    def test_no_config_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert isinstance(config, AfterburnConfig)
        assert config.device == "auto"

    def test_empty_yaml_returns_defaults(self, tmp_path):
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        config = load_config(str(config_path))
        assert isinstance(config, AfterburnConfig)
