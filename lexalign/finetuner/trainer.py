from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from lexalign.finetuner.dataset_prep import DatasetPreparer, DatasetError
from lexalign.finetuner.lora_config import LoraConfigBuilder


class TrainerError(Exception):
    """Training-related errors."""
    pass


class FinetuneTrainer:
    """Wrapper for TRL SFTTrainer with LoRA/QLoRA support."""

    def __init__(self, config: dict, verbose: bool = False):
        """
        Initialize trainer with configuration.

        Args:
            config: Validated configuration dictionary
            verbose: Enable detailed training output
        """
        self.config = config
        self.verbose = verbose
        self.device = config.get("device", "cpu")

    def _validate_paths(self):
        """
        Validate that model and dataset paths exist.

        Raises:
            TrainerError: If paths don't exist
        """
        model_path = Path(self.config["model"]["path"])
        if not model_path.exists():
            raise TrainerError(
                f"Model path not found: {model_path}\n"
                f"Please run download.py first to download the model."
            )

        dataset_path = Path(self.config["dataset"]["path"])
        if not dataset_path.exists():
            raise TrainerError(
                f"Dataset path not found: {dataset_path}\n"
                f"Please run download.py first to download the dataset."
            )

    def _get_output_dir(self) -> str:
        """
        Get output directory for checkpoints.

        Returns:
            Output directory path
        """
        training = self.config["training"]
        if "output_dir" in training and training["output_dir"]:
            return training["output_dir"]

        # Default: timestamp-based
        model_name = Path(self.config["model"]["path"]).name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"./checkpoints/{model_name}-{timestamp}"

    def _load_model_and_tokenizer(self):
        """
        Load model and tokenizer with quantization if needed.

        Returns:
            tuple: (model, tokenizer)
        """
        model_path = self.config["model"]["path"]
        training = self.config["training"]
        method = training.get("method", "lora")

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        # Add quantization for QLoRA
        if method == "qlora":
            quant_bits = training.get("quantization_bits", 4)
            lora_builder = LoraConfigBuilder()
            quantization_config = lora_builder.get_quantization_config(quant_bits)
            model_kwargs.update(quantization_config)
            model_kwargs["device_map"] = "auto"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Set pad token if not present
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
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer()

            # Load dataset
            dataset_config = self.config["dataset"]
            preparer = DatasetPreparer()
            dataset = preparer.load_dataset(
                dataset_config["path"],
                dataset_config.get("format", "auto"),
                dataset_config.get("text_field", "text"),
                dataset_config.get("train_split", "train")
            )

            # Build LoRA configuration
            training = self.config["training"]
            lora_builder = LoraConfigBuilder()
            lora_config = lora_builder.build(training)

            # Prepare training arguments
            training_args = self._prepare_training_arguments(output_dir)

            # Create trainer
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                dataset_text_field=dataset_config.get("text_field", "text"),
                peft_config=lora_config,
                tokenizer=tokenizer,
                max_seq_length=training.get("max_seq_length", 512),
                packing=training.get("packing", False),
            )

            # Train
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Starting fine-tuning...")
                print(f"Method: {training.get('method', 'lora').upper()}")
                print(f"Device: {self.device}")
                print(f"Output: {output_dir}")
                print(f"{'='*60}\n")

            trainer.train()

            # Save final model
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Training complete!")
                print(f"Model saved to: {output_dir}")
                print(f"{'='*60}\n")

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
        """
        self._validate_paths()
        output_dir = self._get_output_dir()

        try:
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer()

            # Load dataset
            dataset_config = self.config["dataset"]
            preparer = DatasetPreparer()
            dataset = preparer.load_dataset(
                dataset_config["path"],
                dataset_config.get("format", "auto"),
                dataset_config.get("text_field", "text"),
                dataset_config.get("train_split", "train")
            )

            # Build LoRA configuration
            training = self.config["training"]
            lora_builder = LoraConfigBuilder()
            lora_config = lora_builder.build(training)

            # Prepare training arguments
            training_args = self._prepare_training_arguments(output_dir)

            # Create trainer
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                dataset_text_field=dataset_config.get("text_field", "text"),
                peft_config=lora_config,
                tokenizer=tokenizer,
                max_seq_length=training.get("max_seq_length", 512),
                packing=training.get("packing", False),
            )

            # Resume from checkpoint
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Resuming training from: {checkpoint_path}")
                print(f"{'='*60}\n")

            trainer.train(resume_from_checkpoint=checkpoint_path)

            # Save final model
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Training complete!")
                print(f"Model saved to: {output_dir}")
                print(f"{'='*60}\n")

            return output_dir

        except Exception as e:
            raise TrainerError(f"Resume training failed: {e}")
