import pytest
from pathlib import Path
from lexalign.finetuner.checkpoint import CheckpointManager

def test_find_latest_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Create checkpoint directories
    (checkpoint_dir / "checkpoint-500").mkdir()
    (checkpoint_dir / "checkpoint-1000").mkdir()
    (checkpoint_dir / "checkpoint-1500").mkdir()

    manager = CheckpointManager()
    latest = manager.find_latest(str(checkpoint_dir))

    assert latest.endswith("checkpoint-1500")

def test_no_checkpoints_found(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    manager = CheckpointManager()
    latest = manager.find_latest(str(checkpoint_dir))

    assert latest is None

def test_get_checkpoint_step_from_name():
    manager = CheckpointManager()
    step = manager.get_step("/path/to/checkpoint-1000")

    assert step == 1000
