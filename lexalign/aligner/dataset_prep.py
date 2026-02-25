# lexalign/aligner/dataset_prep.py
"""Preference dataset loading and validation for alignment training."""

from pathlib import Path
from typing import Dict, Any

from datasets import Dataset, load_dataset


class DatasetError(Exception):
    """Dataset-related errors."""


class PreferenceDataset:
    """Load and validate preference datasets for DPO/GDPO alignment."""

    _FORMAT_MAP: Dict[str, str] = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".csv": "csv",
    }

    def load_and_validate(self, config: Dict[str, Any]) -> Dataset:
        """
        Load and validate a preference dataset.

        Args:
            config: Dataset configuration dict with keys:
                path, format, prompt_field, chosen_field, rejected_field

        Returns:
            Loaded and validated ``datasets.Dataset`` object

        Raises:
            DatasetError: If loading or validation fails
        """
        path = config["path"]
        fmt = config.get("format", "auto")

        if fmt == "auto":
            fmt = self._detect_format(path)

        try:
            if fmt in ("json", "jsonl"):
                dataset = load_dataset("json", data_files=path, split="train")
            elif fmt == "csv":
                dataset = load_dataset("csv", data_files=path, split="train")
            else:
                raise DatasetError(f"Unsupported format: {fmt!r}")
        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}")

        self._validate_fields(dataset, config)
        return dataset

    def _detect_format(self, path: str) -> str:
        """Auto-detect file format from extension.

        Args:
            path: Path to the dataset file

        Returns:
            Format string: "json", "jsonl", or "csv". Defaults to "json".
        """
        ext = Path(path).suffix.lower()
        return self._FORMAT_MAP.get(ext, "json")

    def _validate_fields(
        self, dataset: Dataset, config: Dict[str, Any]
    ) -> None:
        """Validate required fields exist in dataset columns.

        Args:
            dataset: Loaded dataset
            config: Dataset configuration dict

        Raises:
            DatasetError: If required columns are missing
        """
        required = [
            config["prompt_field"],
            config["chosen_field"],
            config["rejected_field"],
        ]
        columns = dataset.column_names
        missing = [f for f in required if f not in columns]

        if missing:
            raise DatasetError(
                f"Dataset missing required fields: {', '.join(missing)}. "
                f"Available columns: {', '.join(columns)}"
            )
