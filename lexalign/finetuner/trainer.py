from pathlib import Path
from datetime import datetime
from typing import Optional


class TrainerError(Exception):
    """Training-related errors."""
    pass


class FinetuneTrainer:
    """Wrapper for TRL SFTTrainer."""

    def __init__(self, config: dict):
        """
        Initialize trainer with configuration.

        Args:
            config: Validated configuration dictionary
        """
        self.config = config

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

    def train(self):
        """
        Execute fine-tuning.

        This is a placeholder - full implementation in later tasks.
        """
        self._validate_paths()
        output_dir = self._get_output_dir()
        # Training logic to be implemented
        return output_dir
