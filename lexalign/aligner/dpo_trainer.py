# lexalign/aligner/dpo_trainer.py
"""DPO trainer wrapper for LexAlign."""

from datasets import Dataset
from trl import DPOConfig, DPOTrainer


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
        self._trainer: DPOTrainer | None = None  # set during train()

        self._training_args = DPOConfig(
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
            beta=config["beta"],
            loss_type=config["loss_type"],
        )

    def train(self, train_dataset: Dataset):
        """
        Run DPO training on the provided dataset.

        The underlying DPOTrainer is stored as ``self._trainer`` so it can be
        reused by :meth:`save_model` without creating a second instance.

        Args:
            train_dataset: Training dataset with prompt/chosen/rejected columns
        """
        # NOTE: DPOTrainer is constructed here (not in __init__) so that the
        # dataset is passed in and used — previously the dataset argument was
        # silently ignored.
        self._trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self._training_args,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
        )
        return self._trainer.train()

    def save_model(self, output_dir: str) -> None:
        """Save fine-tuned model and tokenizer.

        Must be called *after* :meth:`train` — reuses the trainer that was
        created there rather than constructing a fresh (dataset-less) one.

        Args:
            output_dir: Directory to save model artifacts

        Raises:
            RuntimeError: If called before :meth:`train`.
        """
        if self._trainer is None:
            raise RuntimeError(
                "save_model() called before train(). "
                "You must call train() first so the DPOTrainer is initialised."
            )
        self._trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
