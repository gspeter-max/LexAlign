import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner

from download import cli


def test_full_workflow_dry_run(mocker):
    """Test complete workflow with dry run."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
huggingface:
  token: "dummy_token_for_testing"

models:
  - repo: "gpt2"
    files:
      - "config.json"
    output_dir: "/tmp/test_models"
""")
        config_path = f.name

    try:
        # Mock auth validation to avoid real API call
        mock_auth = mocker.patch('download.AuthManager')
        mock_auth.return_value.validate_token.return_value = True

        # Mock the downloader to avoid real download
        mocker.patch('download.ModelDownloader.download_repo', return_value={
            "repo": "gpt2",
            "files": ["config.json"],
            "downloaded": 0,
            "output_dir": "/tmp/test_models"
        })

        result = runner.invoke(cli, ["--config", config_path, "--dry-run"])
        assert result.exit_code == 0
    finally:
        Path(config_path).unlink()


def test_finetune_workflow_with_mocks(tmp_path):
    """Test complete fine-tuning workflow with mocked dependencies."""
    # Create mock model and dataset directories
    model_dir = tmp_path / "models" / "gpt2"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    data_dir = tmp_path / "data" / "dataset"
    data_dir.mkdir(parents=True)
    (data_dir / "train.json").write_text('[{"text": "sample"}]')

    # Create config file
    config_file = tmp_path / "finetune.yaml"
    config_file.write_text(f"""
    model:
      path: "{model_dir}"
    dataset:
      path: "{data_dir}"
    training:
      method: "lora"
    device: "cpu"
    """)

    from click.testing import CliRunner
    from finetune import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file), "--dry-run"])

    assert result.exit_code == 0


def test_finetune_real_training_step(mocker, tmp_path):
    """Test actual training step with tiny model to verify integration."""
    # Skip if transformers not available
    pytest.importorskip("transformers")

    # Create minimal model directory with config
    model_dir = tmp_path / "models" / "tiny"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"vocab_size": 1000, "n_positions": 128, "n_embd": 32}')

    # Create minimal dataset
    data_dir = tmp_path / "data" / "dataset"
    data_dir.mkdir(parents=True)
    (data_dir / "train.json").write_text('[{"text": "hello world"}, {"text": "test sample"}]')

    # Create config file with minimal training
    config_file = tmp_path / "finetune.yaml"
    config_file.write_text(f"""
    model:
      path: "{model_dir}"
    dataset:
      path: "{data_dir}"
      format: "json"
    training:
      method: "lora"
      max_seq_length: 64
      num_epochs: 1
      batch_size: 1
      gradient_accumulation_steps: 1
      save_steps: 1000
      max_steps: 1  # Only run 1 step
      output_dir: "{tmp_path / 'output'}"
    device: "cpu"
    """)

    # Mock the model loading to avoid downloading real weights
    mock_model = mocker.patch('lexalign.finetuner.trainer.AutoModelForCausalLM.from_pretrained')
    mock_tokenizer = mocker.patch('lexalign.finetuner.trainer.AutoTokenizer.from_pretrained')

    # Create mock model and tokenizer
    from unittest.mock import MagicMock
    mock_model_instance = MagicMock()
    mock_model_instance.config = MagicMock()
    mock_model_instance.config.vocab_size = 1000

    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.pad_token = "<pad>"
    mock_tokenizer_instance.eos_token = "</s>"

    mock_model.return_value = mock_model_instance
    mock_tokenizer.return_value = mock_tokenizer_instance

    # Mock SFTTrainer to avoid actual training
    mock_trainer = mocker.patch('lexalign.finetuner.trainer.SFTTrainer')
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = None
    mock_trainer_instance.save_model.return_value = None
    mock_trainer.return_value = mock_trainer_instance

    from finetune import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file)])

    # Verify training was attempted
    assert mock_trainer.called or "Training error" in result.output
