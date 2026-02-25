# tests/test_gdpo_trainer.py
"""Tests for GDPOTrainerWrapper and GDPOTrainer."""

import pytest
import torch
import warnings
from unittest.mock import Mock, MagicMock, patch
from transformers import TrainingArguments

from lexalign.aligner.gdpo_trainer import GDPOTrainerWrapper, GDPOTrainer


def _make_training_args(output_dir="/tmp/gdpo-test") -> TrainingArguments:
    """Minimal real TrainingArguments that satisfies the HuggingFace Trainer."""
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",          # required: no eval dataset provided
        no_cuda=True,                # CPU only for tests
        use_cpu=True,
    )


def _make_config() -> dict:
    return {
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


def test_gdpo_trainer_initialization():
    """Test GDPO trainer wrapper initializes correctly."""
    mock_model = Mock()
    mock_ref_model = Mock()
    mock_tokenizer = Mock()

    wrapper = GDPOTrainerWrapper(mock_model, mock_ref_model, mock_tokenizer, _make_config())

    assert wrapper.group_delay_size == 4
    assert wrapper.group_delay_weight == 0.5
    assert wrapper.model is mock_model
    assert wrapper.ref_model is mock_ref_model


def test_compute_gdpo_loss_math():
    """
    Unit-test compute_gdpo_loss() with known tensors.

    With beta=0.1 and the tensors below:
      pi_logratios  = [1.0, -1.0]   →  mean = 0.0
      ref_logratios = [0.0,  0.0]
      logits_dpo    = beta * ([1.0, -1.0] - [0,0]) = [0.1, -0.1]
      dpo_loss      = -mean(log_sigmoid([0.1, -0.1]))
      delay_penalty = Var([1.0, -1.0]) = 2.0
    """
    wrapper = GDPOTrainerWrapper(Mock(), Mock(), Mock(), _make_config())

    policy_chosen    = torch.tensor([1.0, -1.0])
    policy_rejected  = torch.tensor([0.0,  0.0])
    ref_chosen       = torch.tensor([0.0,  0.0])
    ref_rejected     = torch.tensor([0.0,  0.0])

    loss = wrapper.compute_gdpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    # Loss must be a scalar
    assert loss.shape == torch.Size([])
    # Loss must be finite and positive (DPO loss is always ≥ 0)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_compute_gdpo_loss_scalar_output():
    """compute_gdpo_loss returns a scalar even for batch size 1."""
    wrapper = GDPOTrainerWrapper(Mock(), Mock(), Mock(), _make_config())

    for batch_size in (1, 4, 8):
        loss = wrapper.compute_gdpo_loss(
            torch.randn(batch_size),
            torch.randn(batch_size),
            torch.randn(batch_size),
            torch.randn(batch_size),
        )
        assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss)


def test_get_logps_shape_and_mask():
    """_get_logps should return (B,) tensor and correctly ignore -100 positions."""
    B, T, V = 2, 6, 50
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[0, -1] = -100  # one ignored position

    logps = GDPOTrainer._get_logps(logits, labels)

    assert logps.shape == (B,), f"Expected ({B},), got {logps.shape}"
    assert torch.isfinite(logps).all()


def test_compute_loss_with_preference_data():
    """
    GDPOTrainer.compute_loss() should return a scalar loss given chosen/rejected.

    We use a real nn.Module (TinyLM) so HuggingFace Trainer.__init__ can
    identify it as a PyTorch model (MagicMock fails that check).
    compute_loss() is called directly — no GPU needed, no training loop runs.
    """
    import types
    B, T, V = 2, 8, 32

    # A real tiny PyTorch model whose forward() returns an object with .logits
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(V, V)

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            # embed returns (B, T, V) — use directly as logits
            logits = self.embed(input_ids)   # shape: (B, T, V)
            return types.SimpleNamespace(logits=logits, loss=None)

    real_model = TinyLM()

    trainer = GDPOTrainer(
        model=real_model,
        ref_model=None,    # no ref model → KL term is zero
        args=_make_training_args(),
        group_delay_weight=0.5,
        beta=0.1,
    )

    inputs = {
        "chosen_input_ids":        torch.randint(0, V, (B, T)),
        "chosen_attention_mask":   torch.ones(B, T, dtype=torch.long),
        "rejected_input_ids":      torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
    }

    # Call compute_loss directly — no training loop, no GPU
    loss = trainer.compute_loss(real_model, inputs, return_outputs=False)

    assert loss.shape == torch.Size([]), f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss)


def test_compute_loss_fallback_without_preference_data():
    """
    GDPOTrainer falls back to LM loss when batch lacks chosen/rejected keys.

    We use a real nn.Module (TinyFallbackLM) that produces a .loss attribute,
    matching what a real LM returns when labels are supplied.
    """
    import types
    B, T, V = 2, 8, 32
    expected_loss = torch.tensor(1.23)

    class TinyFallbackLM(torch.nn.Module):
        """Always returns a fixed .loss regardless of inputs."""
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)  # any param so Trainer accepts it

        def forward(self, **kwargs):
            return types.SimpleNamespace(loss=expected_loss)

    real_model = TinyFallbackLM()

    trainer = GDPOTrainer(
        model=real_model,
        ref_model=None,
        args=_make_training_args(),
    )

    # A plain batch — no chosen_input_ids / rejected_input_ids
    inputs = {
        "input_ids":      torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loss = trainer.compute_loss(real_model, inputs)

    assert loss == expected_loss
    assert any("chosen_input_ids" in str(warning.message) for warning in w), (
        "Expected UserWarning about missing preference data"
    )
