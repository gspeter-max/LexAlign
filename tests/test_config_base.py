# tests/test_config_base.py
"""Tests for shared BaseConfigParser logic."""

import pytest
from lexalign.config.errors import ConfigError
from lexalign.config.base_parser import BaseConfigParser


class SampleParser(BaseConfigParser):
    """Minimal concrete parser for testing."""
    REQUIRED_FIELDS = ["model", "training"]
    DEFAULTS = {
        "training": {
            "lr": 1e-3,
            "epochs": 3,
        }
    }


class TestValidateRequiredFields:
    def test_passes_with_all_required_fields(self):
        parser = SampleParser()
        parser._validate_required_fields({"model": {}, "training": {}})

    def test_raises_on_missing_field(self):
        parser = SampleParser()
        with pytest.raises(ConfigError, match="Missing required field: training"):
            parser._validate_required_fields({"model": {}})

    def test_raises_if_config_is_not_dict(self):
        parser = SampleParser()
        with pytest.raises(ConfigError, match="Config must be a YAML mapping"):
            parser._validate_required_fields(["model", "training"])


class TestApplyDefaults:
    def test_fills_in_missing_defaults(self):
        parser = SampleParser()
        config = {"model": {}, "training": {}}
        parser._apply_defaults(config)
        assert config["training"]["lr"] == 1e-3
        assert config["training"]["epochs"] == 3

    def test_does_not_overwrite_existing_values(self):
        parser = SampleParser()
        config = {"model": {}, "training": {"lr": 5e-5}}
        parser._apply_defaults(config)
        assert config["training"]["lr"] == 5e-5
        assert config["training"]["epochs"] == 3

    def test_creates_missing_section(self):
        parser = SampleParser()
        config = {"model": {}}
        parser._apply_defaults(config)
        assert "training" in config
        assert config["training"]["lr"] == 1e-3
