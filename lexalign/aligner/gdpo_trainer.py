# lexalign/aligner/gdpo_trainer.py
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from typing import Dict, Any, Optional, List, Union


class GDPOTrainerWrapper:
    """Wrapper for Group Delay Policy Optimization trainer."""

    def __init__(self, model, ref_model, tokenizer, config: dict):
        """
        Initialize GDPO trainer.

        Args:
            model: The model to train
            ref_model: Reference model (frozen)
            tokenizer: Tokenizer
            config: Training configuration dict
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.group_delay_size = config.get("group_delay_size", 4)
        self.group_delay_weight = config.get("group_delay_weight", 0.5)

        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=config["output_dir"],
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["num_epochs"],
            warmup_steps=config.get("warmup_steps", 100),
            weight_decay=config.get("weight_decay", 0.01),
            save_steps=config.get("save_steps", 500),
            save_total_limit=3,
            logging_steps=10,
            remove_unused_columns=False,
        )

    def compute_gdpo_loss(self, policy_chosen_logps: torch.Tensor,
                         policy_rejected_logps: torch.Tensor,
                         ref_chosen_logps: torch.Tensor,
                         ref_rejected_logps: torch.Tensor) -> torch.Tensor:
        """
        Compute GDPO loss with group delay weighting.

        Args:
            policy_chosen_logps: Policy model log probs for chosen
            policy_rejected_logps: Policy model log probs for rejected
            ref_chosen_logps: Reference model log probs for chosen
            ref_rejected_logps: Reference model log probs for rejected

        Returns:
            Loss tensor
        """
        # DPO loss
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        dpo_loss = -torch.log(torch.sigmoid(logits)).mean()

        # Group delay penalty (simplified)
        delay_penalty = torch.var(pi_logratios) if len(pi_logratios) > 1 else torch.tensor(0.0)

        # Combined loss
        total_loss = dpo_loss + self.group_delay_weight * delay_penalty
        return total_loss

    def train(self, train_dataset):
        """
        Run training using custom GDPO trainer.

        Args:
            train_dataset: Training dataset with prompt/chosen/rejected
        """
        # Create custom trainer
        trainer = GDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self.training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            group_delay_weight=self.group_delay_weight,
        )
        return trainer.train()

    def save_model(self, output_dir: str):
        """Save fine-tuned model."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class GDPOTrainer(Trainer):
    """Custom Trainer for GDPO with group delay loss."""

    def __init__(self, *args, group_delay_weight: float = 0.5, **kwargs):
        """Initialize GDPO trainer with group delay weight."""
        super().__init__(*args, **kwargs)
        self.group_delay_weight = group_delay_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute GDPO loss with group delay penalty.

        Args:
            model: The model being trained
            inputs: Batch of inputs
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor (and optionally outputs)
        """
        # Forward pass on policy model
        outputs = model(**inputs)
        policy_logits = outputs.logits

        # Forward pass on reference model (no grad)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits

        # Compute log probabilities
        policy_logps = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        ref_logps = torch.nn.functional.log_softmax(ref_logits, dim=-1)

        # For simplicity, use a basic DPO-style loss
        # In production, you'd extract chosen/rejected from the inputs properly
        dpo_loss = -(policy_logps.mean() - ref_logps.mean()).abs()

        # Group delay penalty (variance of log probabilities)
        delay_penalty = torch.var(policy_logps)

        # Combined loss
        total_loss = dpo_loss + self.group_delay_weight * delay_penalty

        return (total_loss, outputs) if return_outputs else total_loss
