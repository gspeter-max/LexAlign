# lexalign/finetuner/trainer.py
"""Fine-tuning trainer wrapping TRL SFTTrainer with LoRA/QLoRA support."""

import logging
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from lexalign.finetuner.dataset_prep import DatasetPreparer, DatasetError
from lexalign.finetuner.lora_config import LoraConfigBuilder

logger = logging.getLogger(__name__)


class TrainerError(Exception):
    """Training-related errors."""


class FinetuneTrainer:
    """
    Wrapper for TRL SFTTrainer with LoRA/QLoRA support.

    SECURITY NOTE: This tool uses `trust_remote_code=True` when loading
    tokenizers, which allows models to execute custom code from the Hugging
    Face Hub. Only use this with models from trusted sources.
    """

    def __init__(self, config: dict, verbose: bool = False) -> None:
        """
        Initialize trainer with configuration.

        Args:
            config: Validated configuration dictionary
            verbose: Enable detailed training output
        """
        self.config = config
        self.verbose = verbose
        self.device = config.get("device", "cpu")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        """
        Validate that model and dataset paths exist.

        Raises:
            TrainerError: If paths don't exist
        """
        model_path = Path(self.config["model"]["path"])
        if not model_path.exists():
            raise TrainerError(
                f"Model path not found: {model_path}\n"
                "Please run download.py first to download the model."
            )

        dataset_path = Path(self.config["dataset"]["path"])
        if not dataset_path.exists():
            raise TrainerError(
                f"Dataset path not found: {dataset_path}\n"
                "Please run download.py first to download the dataset."
            )

    def _get_output_dir(self) -> str:
        """
        Get output directory for checkpoints.

        Returns:
            Output directory path string
        """
        training = self.config["training"]
        if training.get("output_dir"):
            return training["output_dir"]

        model_name = Path(self.config["model"]["path"]).name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"./checkpoints/{model_name}-{timestamp}"

    def _load_model_and_tokenizer(self):
        """
        Load model and tokenizer with quantization if needed.

        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = self.config["model"]["path"]
        training = self.config["training"]
        method = training.get("method", "lora")

        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        if method == "qlora":
            quant_bits = training.get("quantization_bits", 4)
            lora_builder = LoraConfigBuilder()
            quantization_config = lora_builder.get_quantization_config(quant_bits)
            model_kwargs.update(quantization_config)
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        tokenizer_path = self.config["model"].get("base_model", model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _prepare_training_arguments(self, output_dir: str) -> TrainingArguments:
        """
        Prepare TrainingArguments for SFTTrainer.

        Args:
            output_dir: Directory for checkpoints

        Returns:
            TrainingArguments instance
        """
        training = self.config["training"]

        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=training.get("learning_rate", 3e-4),
            per_device_train_batch_size=training.get("batch_size", 4),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
            num_train_epochs=training.get("num_epochs", 3),
            warmup_steps=training.get("warmup_steps", 100),
            weight_decay=training.get("weight_decay", 0.01),
            logging_steps=10 if self.verbose else 50,
            save_steps=training.get("save_steps", 500),
            max_steps=training.get("max_steps", -1),
            save_total_limit=3,
            fp16=self.device == "cuda",
            report_to="none",
            save_strategy="steps",
        )

    def _build_trainer(self, output_dir: str) -> SFTTrainer:
        """
        Build and configure a ready-to-use SFTTrainer.

        This shared factory is called by both ``train()`` and ``resume()``.

        Args:
            output_dir: Directory for checkpoints

        Returns:
            Configured SFTTrainer instance
        """
        model, tokenizer = self._load_model_and_tokenizer()

        dataset_config = self.config["dataset"]
        preparer = DatasetPreparer()
        dataset = preparer.load_dataset(
            dataset_config["path"],
            dataset_config.get("format", "auto"),
            dataset_config.get("text_field", "text"),
            dataset_config.get("train_split", "train"),
        )

        training = self.config["training"]
        lora_builder = LoraConfigBuilder()
        lora_config = lora_builder.build(training)

        training_args = self._prepare_training_arguments(output_dir)

        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field=dataset_config.get("text_field", "text"),
            peft_config=lora_config,
            tokenizer=tokenizer,
            max_seq_length=training.get("max_seq_length", 512),
            packing=training.get("packing", False),
        )

    def _log_header(self, title: str, output_dir: str) -> None:
        """Log a training header via the logger (replaces raw print())."""
        sep = "=" * 60
        logger.info(sep)
        logger.info(title)
        logger.info("Method: %s", self.config["training"].get("method", "lora").upper())
        logger.info("Device: %s", self.device)
        logger.info("Output: %s", output_dir)
        logger.info(sep)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> str:
        """
        Execute fine-tuning using TRL SFTTrainer.

        Returns:
            Output directory where checkpoints were saved

        Raises:
            TrainerError: If training fails
        """
        self._validate_paths()
        output_dir = self._get_output_dir()

        try:
            if self.verbose:
                self._log_header("Starting fine-tuning...", output_dir)

            trainer = self._build_trainer(output_dir)
            trainer.train()
            trainer.save_model(output_dir)
            trainer.tokenizer.save_pretrained(output_dir)

            if self.verbose:
                logger.info("Training complete! Model saved to: %s", output_dir)

            return output_dir

        except DatasetError as e:
            raise TrainerError(f"Dataset error: {e}")
        except Exception as e:
            raise TrainerError(f"Training failed: {e}")

    def resume(self, checkpoint_path: str) -> str:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Output directory where checkpoints were saved

        Raises:
            TrainerError: If training fails
        """
        self._validate_paths()
        output_dir = self._get_output_dir()

        try:
            if self.verbose:
                self._log_header(
                    f"Resuming training from: {checkpoint_path}", output_dir
                )

            trainer = self._build_trainer(output_dir)
            trainer.train(resume_from_checkpoint=checkpoint_path)
            trainer.save_model(output_dir)
            trainer.tokenizer.save_pretrained(output_dir)

            if self.verbose:
                logger.info("Training complete! Model saved to: %s", output_dir)

            return output_dir

        except Exception as e:
            raise TrainerError(f"Resume training failed: {e}")
