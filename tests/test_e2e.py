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
