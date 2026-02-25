# tests/test_lora_config.py
"""Tests for LoraConfigBuilder."""

import platform
import warnings
import pytest
from unittest.mock import patch, MagicMock

from lexalign.finetuner.lora_config import LoraConfigBuilder


def test_build_lora_config_defaults():
    """build() should return a LoraConfig with sensible defaults."""
    builder = LoraConfigBuilder()
    with patch("lexalign.finetuner.lora_config.LoraConfig") as MockLoraConfig:
        MockLoraConfig.return_value = MagicMock()
        builder.build({})
        MockLoraConfig.assert_called_once()
        call_kwargs = MockLoraConfig.call_args[1]
        assert call_kwargs["r"] == 16
        assert call_kwargs["lora_alpha"] == 32
        assert call_kwargs["lora_dropout"] == 0.05
        assert call_kwargs["bias"] == "none"


def test_build_lora_config_custom():
    """build() should respect custom params from training_params."""
    builder = LoraConfigBuilder()
    params = {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1}
    with patch("lexalign.finetuner.lora_config.LoraConfig") as MockLoraConfig:
        MockLoraConfig.return_value = MagicMock()
        builder.build(params)
        call_kwargs = MockLoraConfig.call_args[1]
        assert call_kwargs["r"] == 8
        assert call_kwargs["lora_alpha"] == 16
        assert call_kwargs["lora_dropout"] == 0.1


def test_get_quantization_config_4bit():
    """get_quantization_config(4) should return load_in_4bit=True."""
    builder = LoraConfigBuilder()
    with patch("platform.system", return_value="Linux"):
        result = builder.get_quantization_config(4)
    assert result == {"load_in_4bit": True}


def test_get_quantization_config_8bit():
    """get_quantization_config(8) should return load_in_8bit=True."""
    builder = LoraConfigBuilder()
    with patch("platform.system", return_value="Linux"):
        result = builder.get_quantization_config(8)
    assert result == {"load_in_8bit": True}


def test_get_quantization_config_invalid():
    """get_quantization_config with bits != 4 or 8 should raise ValueError."""
    builder = LoraConfigBuilder()
    with patch("platform.system", return_value="Linux"):
        with pytest.raises(ValueError, match="Quantization must be 4 or 8"):
            builder.get_quantization_config(16)


def test_qlora_warns_on_non_linux():
    """get_quantization_config() should emit RuntimeWarning on macOS/Windows."""
    builder = LoraConfigBuilder()
    with patch("platform.system", return_value="Darwin"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.get_quantization_config(4)

    assert result == {"load_in_4bit": True}  # still works, just warns
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1, (
        f"Expected 1 RuntimeWarning, got {len(runtime_warnings)}"
    )
    assert "bitsandbytes" in str(runtime_warnings[0].message)
    assert "Linux" in str(runtime_warnings[0].message)


def test_qlora_no_warning_on_linux():
    """get_quantization_config() should NOT warn on Linux."""
    builder = LoraConfigBuilder()
    with patch("platform.system", return_value="Linux"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.get_quantization_config(4)

    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, (
        "Unexpected RuntimeWarning on Linux"
    )
