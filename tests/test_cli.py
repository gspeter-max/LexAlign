import pytest
from click.testing import CliRunner
from download import cli

def test_cli_requires_config():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "--config" in result.output

def test_cli_with_invalid_config(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: [[")

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file)])
    assert result.exit_code != 0

def test_cli_dry_run(mocker, tmp_path):
    config_content = """
huggingface:
  token: "test_token"

models:
  - repo: "org/model"
    files:
      - "*.json"
    output_dir: "./models"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Mock auth validation
    mock_auth = mocker.patch('download.AuthManager')
    mock_auth.return_value.validate_token.return_value = True

    # Mock the downloader
    mocker.patch('download.ModelDownloader.download_repo', return_value={
        "repo": "org/model",
        "files": ["config.json"],
        "downloaded": 0,
        "output_dir": "./models"
    })

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file), "--dry-run"])
    assert result.exit_code == 0
