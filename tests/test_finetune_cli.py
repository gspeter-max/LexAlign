import pytest
from click.testing import CliRunner
from finetune import cli

def test_cli_requires_config():
    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code != 0
    assert "Missing option" in result.output or "--config" in result.output

def test_cli_dry_run(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    model:
      path: "./models/test"
    dataset:
      path: "./data/test"
    training:
      method: "lora"
    device: "cpu"
    """)

    mocker.patch('finetune.Path.exists', return_value=True)

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file), "--dry-run"])

    # Should show config without training
    assert "Fine-tuning configuration" in result.output or result.exit_code == 0

def test_cli_model_not_found(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    model:
      path: "./nonexistent/model"
    dataset:
      path: "./data/test"
    training:
      method: "lora"
    device: "cpu"
    """)

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file)])

    assert "Model path not found" in result.output
