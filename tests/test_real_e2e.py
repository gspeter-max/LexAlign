"""Real end-to-end integration tests with actual model downloads and training."""
import os
import pytest
from pathlib import Path
from click.testing import CliRunner

# Skip all tests in this file if HF_TOKEN not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="HF_TOKEN environment variable not set (use: export HF_TOKEN=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)"
    )
]


@pytest.fixture
def sample_finetune_data(tmp_path):
    """Create 4-sample fine-tuning dataset."""
    import json
    data = [
        {"text": "The sky is blue during the day."},
        {"text": "Grass is green and grows in gardens."},
        {"text": "The sun provides light and warmth."},
        {"text": "Water is essential for all life forms."}
    ]
    data_file = tmp_path / "finetune.json"
    data_file.write_text(json.dumps(data))
    return str(data_file)


@pytest.fixture
def sample_dpo_data(tmp_path):
    """Create 4-sample DPO preference dataset."""
    import json
    data = [
        {"prompt": "What color is the sky?", "chosen": "The sky is blue.", "rejected": "I don't know."},
        {"prompt": "What color is grass?", "chosen": "Grass is green.", "rejected": "Purple."},
        {"prompt": "What does the sun do?", "chosen": "The sun provides light.", "rejected": "Nothing."},
        {"prompt": "Why is water important?", "chosen": "Water sustains life.", "rejected": "It's not."}
    ]
    data_file = tmp_path / "dpo.json"
    data_file.write_text(json.dumps(data))
    return str(data_file)


@pytest.fixture
def download_config(tmp_path):
    """Create download config for distilgpt2."""
    config = f"""
huggingface:
  token: "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"

models:
  - repo: "distilgpt2"
    files:
      - "config.json"
      - "pytorch_model.bin"
      - "tokenizer.json"
      - "vocab.json"
      - "merges.txt"
    output_dir: "{tmp_path}/models/distilgpt2"
"""
    config_file = tmp_path / "download.yaml"
    config_file.write_text(config)
    return str(config_file)
