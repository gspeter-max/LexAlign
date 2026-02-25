# tests/test_dpo_trainer.py
"""Tests for DPOTrainerWrapper."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from lexalign.aligner.dpo_trainer import DPOTrainerWrapper


def _make_config() -> dict:
    return {
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


def test_dpo_trainer_initialization():
    """DPOTrainerWrapper should store model/tokenizer/config without building the trainer yet."""
    config = _make_config()
    model = Mock()
    ref_model = Mock()
    tokenizer = Mock()

    wrapper = DPOTrainerWrapper(model, ref_model, tokenizer, config)

    assert wrapper.model is model
    assert wrapper.ref_model is ref_model
    assert wrapper.tokenizer is tokenizer
    assert wrapper.config is config
    assert wrapper._trainer is None  # not built yet


def test_train_passes_dataset_to_dpo_trainer():
    """train() must pass train_dataset to the underlying DPOTrainer — not ignore it."""
    config = _make_config()
    model = Mock()
    ref_model = Mock()
    tokenizer = Mock()
    dataset = [{"prompt": "hi", "chosen": "hello", "rejected": "bye"}]

    wrapper = DPOTrainerWrapper(model, ref_model, tokenizer, config)

    with patch("lexalign.aligner.dpo_trainer.DPOTrainer") as MockDPO:
        mock_trainer = MagicMock()
        MockDPO.return_value = mock_trainer

        wrapper.train(dataset)

        MockDPO.assert_called_once()
        call_kwargs = MockDPO.call_args[1]
        assert call_kwargs["train_dataset"] is dataset, (
            "train_dataset was NOT passed to DPOTrainer — the dataset would be silently ignored"
        )
        assert call_kwargs["model"] is model
        assert call_kwargs["ref_model"] is ref_model
        assert call_kwargs["beta"] == 0.1
        mock_trainer.train.assert_called_once()


def test_save_model_reuses_trainer():
    """save_model() must reuse self._trainer from train() — not create a second DPOTrainer."""
    config = _make_config()
    wrapper = DPOTrainerWrapper(Mock(), Mock(), Mock(), config)

    with patch("lexalign.aligner.dpo_trainer.DPOTrainer") as MockDPO:
        mock_trainer = MagicMock()
        MockDPO.return_value = mock_trainer

        dataset = [{"prompt": "x", "chosen": "a", "rejected": "b"}]
        wrapper.train(dataset)
        wrapper.save_model("/tmp/out")

        # DPOTrainer must be constructed exactly once (in train), not again in save_model
        assert MockDPO.call_count == 1, (
            f"DPOTrainer was constructed {MockDPO.call_count} times; expected 1"
        )
        mock_trainer.save_model.assert_called_once_with("/tmp/out")
        wrapper.tokenizer.save_pretrained.assert_called_once_with("/tmp/out")


def test_save_model_before_train_raises():
    """save_model() must raise RuntimeError if called before train()."""
    config = _make_config()
    wrapper = DPOTrainerWrapper(Mock(), Mock(), Mock(), config)

    with pytest.raises(RuntimeError, match="train()"):
        wrapper.save_model("/tmp/out")
