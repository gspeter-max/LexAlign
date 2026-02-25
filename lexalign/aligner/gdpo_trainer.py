# lexalign/aligner/gdpo_trainer.py
"""GDPO (Group Delay Policy Optimization) trainer for LexAlign."""

import logging
import warnings

import torch
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


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

    def compute_gdpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GDPO loss with group delay weighting.

        DPO base loss:
            L_dpo = -log σ( β * (log π(y_w|x) - log π(y_l|x))
                              - β * (log π_ref(y_w|x) - log π_ref(y_l|x)) )

        Group delay penalty:
            Var(log π(y_w|x) - log π(y_l|x))  — penalises high variance between
            batch elements, encouraging a stable, consistent preference signal.

        Args:
            policy_chosen_logps:   Per-sample log-prob sums for chosen sequences  (B,)
            policy_rejected_logps: Per-sample log-prob sums for rejected sequences (B,)
            ref_chosen_logps:      Reference log-prob sums for chosen sequences    (B,)
            ref_rejected_logps:    Reference log-prob sums for rejected sequences  (B,)

        Returns:
            Scalar loss tensor
        """
        beta = self.config.get("beta", 0.1)

        # Policy and reference log-ratios per sample
        pi_logratios = policy_chosen_logps - policy_rejected_logps   # (B,)
        ref_logratios = ref_chosen_logps - ref_rejected_logps         # (B,)

        # DPO loss: -log σ(β * (pi_ratio - ref_ratio))
        logits = beta * (pi_logratios - ref_logratios)
        dpo_loss = -torch.nn.functional.logsigmoid(logits).mean()

        # Group delay penalty: variance of per-sample policy log-ratios
        delay_penalty = (
            torch.var(pi_logratios) if pi_logratios.shape[0] > 1
            else torch.tensor(0.0, device=pi_logratios.device)
        )

        total_loss = dpo_loss + self.group_delay_weight * delay_penalty
        return total_loss

    def train(self, train_dataset):
        """
        Run training using custom GDPO trainer.

        Args:
            train_dataset: Training dataset with prompt/chosen/rejected columns
        """
        trainer = GDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self.training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            group_delay_weight=self.group_delay_weight,
            beta=self.config.get("beta", 0.1),
        )
        return trainer.train()

    def save_model(self, output_dir: str) -> None:
        """Save fine-tuned model and tokenizer.

        Args:
            output_dir: Directory to save model artifacts
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class GDPOTrainer(Trainer):
    """Custom Trainer implementing GDPO with proper chosen/rejected extraction."""

    def __init__(
        self,
        *args,
        ref_model=None,
        group_delay_weight: float = 0.5,
        beta: float = 0.1,
        **kwargs,
    ):
        """
        Initialize GDPO trainer.

        Args:
            ref_model: Frozen reference model for KL constraint
            group_delay_weight: Weight of the group delay variance penalty
            beta: DPO temperature coefficient
        """
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.group_delay_weight = group_delay_weight
        self.beta = beta

    @staticmethod
    def _get_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample sum of log probabilities at label positions.

        Args:
            logits: Model logits of shape (B, T, V)
            labels: Token ids of shape (B, T); -100 positions are ignored

        Returns:
            Per-sample log-prob sums of shape (B,)
        """
        # Shift so that token t predicts token t+1
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Mask padding tokens (-100)
        mask = (shift_labels != -100).float()            # (B, T-1)
        shift_labels_clamped = shift_labels.clamp(min=0)

        # Gather log probs at actual token positions
        per_token_logps = log_probs.gather(
            dim=2,
            index=shift_labels_clamped.unsqueeze(2),
        ).squeeze(2)                                      # (B, T-1)

        # Sum over non-padding tokens → (B,)
        return (per_token_logps * mask).sum(dim=1)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ):
        """
        Compute GDPO loss.

        Expects the input batch to contain:
            chosen_input_ids        (B, T)
            chosen_attention_mask   (B, T)
            rejected_input_ids      (B, T)
            rejected_attention_mask (B, T)
            chosen_labels           (B, T)  — token ids, -100 for ignored positions
            rejected_labels         (B, T)

        Falls back to standard LM cross-entropy if chosen/rejected keys are
        absent (e.g., plain language-modelling datasets or unit tests).

        Args:
            model: Policy model being trained
            inputs: Batch dictionary
            return_outputs: If True, return (loss, chosen_outputs)

        Returns:
            loss tensor, or (loss, outputs) tuple when return_outputs=True
        """
        has_preference_data = (
            "chosen_input_ids" in inputs and "rejected_input_ids" in inputs
        )

        if not has_preference_data:
            # Graceful fallback for non-preference batches
            warnings.warn(
                "GDPOTrainer received a batch without 'chosen_input_ids'/"
                "'rejected_input_ids'. Falling back to standard LM loss.",
                UserWarning,
                stacklevel=2,
            )
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # ── chosen forward pass ──────────────────────────────────────────────
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        chosen_labels = inputs.get(
            "chosen_labels", inputs["chosen_input_ids"]
        )
        policy_chosen_logps = self._get_logps(
            chosen_outputs.logits, chosen_labels
        )

        # ── rejected forward pass ────────────────────────────────────────────
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        rejected_labels = inputs.get(
            "rejected_labels", inputs["rejected_input_ids"]
        )
        policy_rejected_logps = self._get_logps(
            rejected_outputs.logits, rejected_labels
        )

        # ── reference model forward passes (no grad) ─────────────────────────
        with torch.no_grad():
            if self.ref_model is not None:
                ref_chosen_out = self.ref_model(
                    input_ids=inputs["chosen_input_ids"],
                    attention_mask=inputs["chosen_attention_mask"],
                )
                ref_rejected_out = self.ref_model(
                    input_ids=inputs["rejected_input_ids"],
                    attention_mask=inputs["rejected_attention_mask"],
                )
                ref_chosen_logps = self._get_logps(
                    ref_chosen_out.logits, chosen_labels
                )
                ref_rejected_logps = self._get_logps(
                    ref_rejected_out.logits, rejected_labels
                )
            else:
                # No reference model: treat as equal (zero KL)
                ref_chosen_logps = policy_chosen_logps.detach()
                ref_rejected_logps = policy_rejected_logps.detach()

        # ── GDPO loss ────────────────────────────────────────────────────────
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits_dpo = self.beta * (pi_logratios - ref_logratios)
        dpo_loss = -torch.nn.functional.logsigmoid(logits_dpo).mean()

        delay_penalty = (
            torch.var(pi_logratios) if pi_logratios.shape[0] > 1
            else torch.tensor(0.0, device=pi_logratios.device)
        )

        total_loss = dpo_loss + self.group_delay_weight * delay_penalty
        logger.debug(
            "GDPO loss=%.4f  dpo=%.4f  delay_penalty=%.4f",
            total_loss.item(),
            dpo_loss.item(),
            delay_penalty.item(),
        )

        return (total_loss, chosen_outputs) if return_outputs else total_loss
