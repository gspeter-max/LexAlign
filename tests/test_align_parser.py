# tests/test_align_parser.py
import pytest
from lexalign.config.align_parser import AlignConfigParser, ConfigError

def test_parse_minimal_valid_config():
    yaml_content = """
model:
  path: "./checkpoints/gpt2-finetuned"

dataset:
  path: "./data/preferences"
  format: "auto"

alignment:
  method: "dpo"

device: "cuda"
"""
    parser = AlignConfigParser()
    config = parser.parse(yaml_content)

    assert config["model"]["path"] == "./checkpoints/gpt2-finetuned"
    assert config["dataset"]["path"] == "./data/preferences"
    assert config["alignment"]["method"] == "dpo"
    assert config["device"] == "cuda"


def test_parse_invalid_method_raises_error():
    yaml_content = """
model:
  path: "./checkpoints/gpt2-finetuned"

dataset:
  path: "./data/preferences"

alignment:
  method: "invalid_method"

device: "cuda"
"""
    parser = AlignConfigParser()
    with pytest.raises(ConfigError, match="Invalid alignment method"):
        parser.parse(yaml_content)


def test_parse_invalid_group_delay_size_raises_error():
    yaml_content = """
model:
  path: "./checkpoints/gpt2-finetuned"

dataset:
  path: "./data/preferences"

alignment:
  method: "gdpo"
  group_delay_size: 1

device: "cuda"
"""
    parser = AlignConfigParser()
    with pytest.raises(ConfigError, match="group_delay_size must be >= 2"):
        parser.parse(yaml_content)
