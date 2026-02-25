# lexalign/aligner/dataset_prep.py
from pathlib import Path
from datasets import load_dataset
import json


class DatasetError(Exception):
    """Dataset-related errors."""
    pass


class PreferenceDataset:
    """Load and validate preference datasets."""

    def load_and_validate(self, config: dict) -> object:
        """
        Load and validate preference dataset.

        Args:
            config: Dataset configuration dict

        Returns:
            datasets.Dataset object

        Raises:
            DatasetError: If validation fails
        """
        path = config["path"]
        fmt = config.get("format", "auto")

        # Auto-detect format
        if fmt == "auto":
            fmt = self._detect_format(path)

        # Load dataset
        try:
            if fmt == "json":
                dataset = load_dataset("json", data_files=path, split="train")
            elif fmt == "jsonl":
                dataset = load_dataset("json", data_files=path, split="train")
            elif fmt == "csv":
                dataset = load_dataset("csv", data_files=path, split="train")
            else:
                raise DatasetError(f"Unsupported format: {fmt}")
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}")

        # Validate required fields
        self._validate_fields(dataset, config)

        return dataset

    def _detect_format(self, path: str) -> str:
        """Auto-detect file format from extension."""
        ext = Path(path).suffix.lower()
        format_map = {".json": "json", ".jsonl": "jsonl", ".csv": "csv"}
        return format_map.get(ext, "json")

    def _validate_fields(self, dataset, config: dict):
        """Validate required fields exist in dataset."""
        prompt_field = config["prompt_field"]
        chosen_field = config["chosen_field"]
        rejected_field = config["rejected_field"]

        columns = dataset.column_names

        missing = []
        if prompt_field not in columns:
            missing.append(prompt_field)
        if chosen_field not in columns:
            missing.append(chosen_field)
        if rejected_field not in columns:
            missing.append(rejected_field)

        if missing:
            raise DatasetError(
                f"Dataset missing required fields: {', '.join(missing)}. "
                f"Available columns: {', '.join(columns)}"
            )
