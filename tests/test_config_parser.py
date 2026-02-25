import pytest
from lexalign.config.parser import ConfigParser, ConfigValidationError

def test_parse_valid_config():
    yaml_content = """
huggingface:
  token: "${HF_TOKEN}"

models:
  - repo: "org/model"
    files:
      - "config.json"
    output_dir: "./models"
"""
    parser = ConfigParser()
    config = parser.parse(yaml_content, {"HF_TOKEN": "test_token"})
    assert config["huggingface"]["token"] == "test_token"
    assert len(config["models"]) == 1
    assert config["models"][0]["repo"] == "org/model"

def test_parse_missing_token_env_var():
    yaml_content = """
huggingface:
  token: "${NONEXISTENT_TOKEN}"

models:
  - repo: "org/model"
    files:
      - "config.json"
    output_dir: "./models"
"""
    parser = ConfigParser()
    with pytest.raises(ConfigValidationError, match="Environment variable"):
        parser.parse(yaml_content, {})

def test_parse_invalid_yaml_structure():
    yaml_content = """
huggingface:
  token: "test"
# Missing models or datasets section
"""
    parser = ConfigParser()
    with pytest.raises(ConfigValidationError, match="must have models or datasets"):
        parser.parse(yaml_content, {})

def test_validate_file_patterns():
    yaml_content = """
huggingface:
  token: "test"

models:
  - repo: "org/model"
    files: []
    output_dir: "./models"
"""
    parser = ConfigParser()
    with pytest.raises(ConfigValidationError, match="at least one file pattern"):
        parser.parse(yaml_content, {})
