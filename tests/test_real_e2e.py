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


def test_download_distilgpt2(download_config, tmp_path):
    """Test downloading distilgpt2 model from Hugging Face."""
    from download import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", download_config])

    # Verify CLI succeeded
    assert result.exit_code == 0, f"Download failed: {result.output}"

    # Verify model files exist
    model_dir = tmp_path / "models" / "distilgpt2"
    assert model_dir.exists(), "Model directory not created"
    assert (model_dir / "config.json").exists(), "config.json missing"
    assert (model_dir / "pytorch_model.bin").exists(), "pytorch_model.bin missing"

    # Verify tokenizer files exist
    assert (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists()


def test_finetune_distilgpt2(download_config, sample_finetune_data, tmp_path):
    """Test fine-tuning distilgpt2 with LoRA (1 step)."""
    from download import cli as download_cli
    from finetune import cli as finetune_cli

    # First download the model
    download_runner = CliRunner()
    download_result = download_runner.invoke(download_cli, ["--config", download_config])
    assert download_result.exit_code == 0, f"Download failed: {download_result.output}"

    model_dir = tmp_path / "models" / "distilgpt2"

    # Create finetune config
    config = f"""
model:
  path: "{model_dir}"

dataset:
  path: "{sample_finetune_data}"
  format: "json"
  text_field: "text"

training:
  method: "lora"
  max_steps: 1
  max_seq_length: 64
  batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 3e-4
  num_epochs: 1
  save_steps: 1000
  output_dir: "{tmp_path}/checkpoints/finetuned"

device: "cpu"
"""
    config_file = tmp_path / "finetune.yaml"
    config_file.write_text(config)

    # Run fine-tuning
    runner = CliRunner()
    result = runner.invoke(finetune_cli, ["--config", str(config_file)])

    # Verify training succeeded
    assert result.exit_code == 0, f"Fine-tuning failed: {result.output}"

    # Verify checkpoint created
    output_dir = tmp_path / "checkpoints" / "finetuned"
    assert output_dir.exists(), "Output directory not created"
    assert (output_dir / "adapter_model.bin").exists() or (output_dir / "adapter_config.json").exists()

    # Verify model can be loaded
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(str(output_dir), device_map="cpu")
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to load fine-tuned model: {e}")
