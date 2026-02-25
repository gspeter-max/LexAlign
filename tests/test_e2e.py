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
