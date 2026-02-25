# lexalign/aligner/dpo_trainer.py
"""DPO trainer wrapper for LexAlign."""

from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer


class DPOTrainerWrapper:
    """Wrapper around TRL's DPOTrainer."""

    def __init__(self, model, ref_model, tokenizer, config: dict) -> None:
        """
        Initialize DPO trainer.

        Args:
            model: The policy model to train
            ref_model: Reference model (frozen)
            tokenizer: Tokenizer for both models
            config: Training configuration dictionary
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config

        self._training_args = TrainingArguments(
            output_dir=config["output_dir"],
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["num_epochs"],
            warmup_steps=config["warmup_steps"],
            weight_decay=config["weight_decay"],
            save_steps=config["save_steps"],
            save_total_limit=3,
            logging_steps=10,
            remove_unused_columns=False,
        )

    def train(self, train_dataset: Dataset):
        """
        Run DPO training on the provided dataset.

        Args:
            train_dataset: Training dataset with prompt/chosen/rejected columns
        """
        # NOTE: DPOTrainer is constructed here (not in __init__) so that the
        # dataset is passed in and used — previously the dataset argument was
        # silently ignored.
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self._training_args,
            beta=self.config["beta"],
            loss_type=self.config["loss_type"],
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
        )
        return trainer.train()

    def save_model(self, output_dir: str) -> None:
        """Save fine-tuned model and tokenizer.

        Args:
            output_dir: Directory to save model artifacts
        """
        # Re-use the same training_args output_dir as a convenience
        # but allow override via explicit argument.
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self._training_args,
            tokenizer=self.tokenizer,
        )
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
