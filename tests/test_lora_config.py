import pytest
from lexalign.finetuner.lora_config import LoraConfigBuilder

def test_build_lora_config():
    training_params = {
        "method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
    }

    builder = LoraConfigBuilder()
    config = builder.build(training_params)

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.target_modules == {"q_proj", "v_proj"}

def test_build_qlora_config():
    training_params = {
        "method": "qlora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "quantization_bits": 4
    }

    builder = LoraConfigBuilder()
    config = builder.build(training_params, quantization_bits=4)

    assert config.r == 8
    assert config.lora_alpha == 16

def test_build_qlora_load_in_4bit():
    builder = LoraConfigBuilder()
    load_in_4bit = builder.get_quantization_config(4)

    assert load_in_4bit == {"load_in_4bit": True}

def test_build_qlora_load_in_8bit():
    builder = LoraConfigBuilder()
    load_in_8bit = builder.get_quantization_config(8)

    assert load_in_8bit == {"load_in_8bit": True}

def test_invalid_quantization_bits():
    builder = LoraConfigBuilder()
    with pytest.raises(ValueError, match="Quantization must be 4 or 8"):
        builder.get_quantization_config(16)
