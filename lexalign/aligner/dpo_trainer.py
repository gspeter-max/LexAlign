# lexalign/aligner/dpo_trainer.py
from transformers import Trainer, TrainingArguments
from trl import DPOTrainer


class DPOTrainerWrapper:
    """Wrapper around TRL's DPOTrainer."""

    def __init__(self, model, ref_model, tokenizer, config: dict):
        """
        Initialize DPO trainer.

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

        # Training arguments
        training_args = TrainingArguments(
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

        # DPO trainer
        self.trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            beta=config["beta"],
            loss_type=config["loss_type"],
            tokenizer=tokenizer,
        )

    def train(self, train_dataset):
        """
        Run training.

        Args:
            train_dataset: Training dataset with prompt/chosen/rejected
        """
        return self.trainer.train()

    def save_model(self, output_dir: str):
        """Save fine-tuned model."""
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
