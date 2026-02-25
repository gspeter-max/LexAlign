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
