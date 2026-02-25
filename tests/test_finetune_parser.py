import pytest
from lexalign.config.finetune_parser import FinetuneConfigParser, ConfigError

def test_parse_valid_finetune_config():
    yaml_content = """
    model:
      path: "./models/gpt2"
    dataset:
      path: "./data/dataset"
      format: "auto"
      text_field: "text"
    training:
      method: "lora"
      learning_rate: 3e-4
      batch_size: 4
      num_epochs: 3
    device: "cuda"
    """

    parser = FinetuneConfigParser()
    config = parser.parse(yaml_content)

    assert config["model"]["path"] == "./models/gpt2"
    assert config["training"]["method"] == "lora"
    assert config["device"] == "cuda"

def test_validate_required_fields():
    yaml_content = """
    model:
      path: "./models/gpt2"
    # Missing dataset section
    """

    parser = FinetuneConfigParser()
    with pytest.raises(ConfigError, match="Missing required field"):
        parser.parse(yaml_content)

def test_default_training_parameters():
    yaml_content = """
    model:
      path: "./models/gpt2"
    dataset:
      path: "./data/dataset"
    training:
      method: "qlora"
      quantization_bits: 4
    device: "cpu"
    """

    parser = FinetuneConfigParser()
    config = parser.parse(yaml_content)

    assert config["training"]["learning_rate"] == 3e-4  # Default
    assert config["training"]["lora_r"] == 16  # Default
    assert config["training"]["quantization_bits"] == 4

def test_invalid_method():
    yaml_content = """
    model:
      path: "./models/gpt2"
    dataset:
      path: "./data/dataset"
    training:
      method: "invalid"
    device: "cpu"
    """

    parser = FinetuneConfigParser()
    with pytest.raises(ConfigError, match="Invalid training method"):
        parser.parse(yaml_content)

def test_invalid_quantization_bits():
    yaml_content = """
    model:
      path: "./models/gpt2"
    dataset:
      path: "./data/dataset"
    training:
      method: "qlora"
      quantization_bits: 16
    device: "cpu"
    """

    parser = FinetuneConfigParser()
    with pytest.raises(ConfigError, match="Quantization must be 4 or 8"):
        parser.parse(yaml_content)
