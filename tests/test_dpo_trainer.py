# tests/test_dpo_trainer.py
import pytest
from unittest.mock import Mock, MagicMock, patch
from lexalign.aligner.dpo_trainer import DPOTrainerWrapper

def test_dpo_trainer_initialization():
    """Test DPO trainer wrapper initializes correctly."""
    mock_model = Mock()
    mock_ref_model = Mock()
    mock_tokenizer = Mock()

    config = {
        "beta": 0.1,
        "loss_type": "sigmoid",
        "learning_rate": 1e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "save_steps": 500,
        "output_dir": "./checkpoints/test-aligned",
    }

    with patch('lexalign.aligner.dpo_trainer.DPOTrainer') as mock_dpo:
        mock_trainer_instance = MagicMock()
        mock_dpo.return_value = mock_trainer_instance

        wrapper = DPOTrainerWrapper(mock_model, mock_ref_model, mock_tokenizer, config)

        # Verify DPOTrainer was called with correct params
        mock_dpo.assert_called_once()
        call_kwargs = mock_dpo.call_args[1]
        assert call_kwargs["model"] == mock_model
        assert call_kwargs["ref_model"] == mock_ref_model
        assert call_kwargs["beta"] == 0.1
        assert call_kwargs["loss_type"] == "sigmoid"
