# tests/test_align_e2e.py
import pytest
from pathlib import Path
from click.testing import CliRunner
from align import align

@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock alignment config file."""

    # Create mock directories first
    model_dir = tmp_path / "checkpoints" / "test-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    # Create mock preference dataset
    import json
    data_dir = tmp_path / "data"
    pref_file = data_dir / "test-preferences.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    pref_data = [
        {"prompt": "Test prompt 1", "chosen": "Good response 1", "rejected": "Bad response 1"},
        {"prompt": "Test prompt 2", "chosen": "Good response 2", "rejected": "Bad response 2"},
    ]
    pref_file.write_text(json.dumps(pref_data))

    # Create config with absolute paths
    config_content = f"""
model:
  path: "{model_dir}"

dataset:
  path: "{pref_file}"
  format: "json"
  prompt_field: "prompt"
  chosen_field: "chosen"
  rejected_field: "rejected"

alignment:
  method: "dpo"
  beta: 0.1
  learning_rate: 1e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 1
  warmup_steps: 10
  weight_decay: 0.01
  save_steps: 100
  output_dir: "./checkpoints/test-aligned"

device: "cpu"
"""
    config_file = tmp_path / "align.yaml"
    config_file.write_text(config_content)

    return str(config_file)

def test_e2e_alignment_workflow(mock_config_file):
    """Test end-to-end alignment workflow config validation via --dry-run."""
    runner = CliRunner()
    result = runner.invoke(align, ["--config", mock_config_file, "--dry-run"])

    # Should succeed — dry-run validates config and exits cleanly without loading models
    assert result.exit_code == 0, f"Unexpected exit: {result.output}"
    # The dry-run output should mention the config contents
    assert "dpo" in result.output.lower() or "dry run" in result.output.lower()
