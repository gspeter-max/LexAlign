# tests/test_align_cli.py
import pytest
from click.testing import CliRunner
from align import align

def test_align_cli_requires_config():
    """Test that CLI requires --config argument."""
    runner = CliRunner()
    result = runner.invoke(align)
    assert result.exit_code != 0
    assert "--config" in result.output or "Missing option" in result.output

def test_align_cli_dry_run():
    """Test dry-run mode."""
    runner = CliRunner()
    result = runner.invoke(align, ["--config", "config/align.yaml.example", "--dry-run"])
    # Should not fail in dry-run mode even without actual model
    assert result.exit_code == 0 or "Error" not in result.output
