# tests/test_gdpo_trainer.py
import pytest
from unittest.mock import Mock, patch
from lexalign.aligner.gdpo_trainer import GDPOTrainerWrapper

def test_gdpo_trainer_initialization():
    """Test GDPO trainer wrapper initializes correctly."""
    mock_model = Mock()
    mock_ref_model = Mock()
    mock_tokenizer = Mock()

    config = {
        "beta": 0.1,
        "loss_type": "sigmoid",
        "group_delay_size": 4,
        "group_delay_weight": 0.5,
        "learning_rate": 1e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "output_dir": "./checkpoints/test-gdpo",
    }

    wrapper = GDPOTrainerWrapper(mock_model, mock_ref_model, mock_tokenizer, config)

    assert wrapper.group_delay_size == 4
    assert wrapper.group_delay_weight == 0.5
    assert wrapper.model == mock_model
    assert wrapper.ref_model == mock_ref_model
