from pathlib import Path
from typing import Optional
from datasets import load_dataset


class DatasetError(Exception):
    """Dataset-related errors."""
    pass


class DatasetPreparer:
    """Auto-detect and load datasets for fine-tuning."""

    SUPPORTED_FORMATS = ["json", "jsonl", "csv"]

    def detect_format(self, path: str) -> str:
        """
        Detect dataset format from file extension.

        Args:
            path: Path to dataset file or directory

        Returns:
            Detected format: "json", "jsonl", or "csv"

        Raises:
            DatasetError: If format cannot be detected
        """
        path_obj = Path(path)

        if path_obj.is_file():
            suffix = path_obj.suffix.lower()
            if suffix == ".json":
                return "json"
            elif suffix == ".jsonl":
                return "jsonl"
            elif suffix == ".csv":
                return "csv"

        # Check directory contents
        if path_obj.is_dir():
            files = list(path_obj.glob("*"))
            if not files:
                raise DatasetError(f"No files found in dataset directory: {path}")

            return self.detect_format(str(files[0]))

        raise DatasetError(f"Unsupported format for: {path}")

    def load_dataset(
        self,
        path: str,
        format_type: str,
        text_field: str,
        split: str = "train"
    ):
        """
        Load and validate dataset.

        Args:
            path: Path to dataset
            format_type: Format type ("auto", "json", "jsonl", "csv")
            text_field: Field name containing text data
            split: Dataset split to load

        Returns:
            Loaded dataset

        Raises:
            DatasetError: If loading fails
        """
        if not Path(path).exists():
            raise DatasetError(f"Dataset path not found: {path}")

        if format_type == "auto":
            format_type = self.detect_format(path)

        if format_type not in self.SUPPORTED_FORMATS:
            raise DatasetError(
                f"Unsupported format: {format_type}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            dataset = load_dataset(
                format_type,
                data_files=path,
                split=split
            )
            return dataset
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}")
