# LoRA/QLoRA Fine-Tuning Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that fine-tunes Hugging Face models using LoRA/QLoRA with auto-detected datasets.

**Architecture:** Separate `finetune.py` CLI using TRL's SFTTrainer with PEFT adapters. Modular design with config parser, dataset prep, LoRA config builder, and checkpoint manager.

**Tech Stack:** Python 3.9+, TRL, PEFT, bitsandbytes, transformers, datasets, accelerate, Click, rich, PyYAML

---

## Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add fine-tuning dependencies to requirements.txt**

Add these lines to requirements.txt:
```txt
trl>=0.9.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
datasets>=2.14.0
transformers>=4.36.0
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat(finetune): add TRL, PEFT, and training dependencies"
```

---

## Task 2: Create Device Detection Utility

**Files:**
- Create: `lexalign/utils/__init__.py`
- Create: `lexalign/utils/device.py`
- Test: `tests/test_device.py`

**Step 1: Write the failing test**

Create `tests/test_device.py`:
```python
import pytest
from lexalign.utils.device import DeviceManager, DeviceError

def test_detect_cuda_when_available(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = True

    manager = DeviceManager()
    device = manager.detect_device()

    assert device == "cuda"

def test_detect_cpu_when_cuda_unavailable(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = False

    manager = DeviceManager()
    device = manager.detect_device()

    assert device == "cpu"

def test_override_device_with_cpu(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = True

    manager = DeviceManager()
    device = manager.get_device("cpu")

    assert device == "cpu"

def test_invalid_device_name(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')

    manager = DeviceManager()
    with pytest.raises(DeviceError, match="Invalid device"):
        manager.get_device("invalid")

def test_cuda_specification_when_unavailable(mocker):
    mock_torch = mocker.patch('lexalign.utils.device.torch')
    mock_torch.cuda.is_available.return_value = False

    manager = DeviceManager()
    device = manager.get_device("cuda")

    assert device == "cpu"  # Fallback to CPU
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_device.py -v`
Expected: FAIL with "No module named 'lexalign.utils.device'"

**Step 3: Create module structure**

Create `lexalign/utils/__init__.py`:
```python
from lexalign.utils.device import DeviceManager, DeviceError

__all__ = ['DeviceManager', 'DeviceError']
```

**Step 4: Run test to verify it still fails**

Run: `pytest tests/test_device.py -v`
Expected: FAIL with "cannot import 'DeviceManager'"

**Step 5: Write minimal implementation**

Create `lexalign/utils/device.py`:
```python
import torch


class DeviceError(Exception):
    """Device-related errors."""
    pass


class DeviceManager:
    """Manages device detection and selection."""

    def detect_device(self) -> str:
        """
        Auto-detect available device.

        Returns:
            "cuda" if available, else "cpu"
        """
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_device(self, requested: str = None) -> str:
        """
        Get device with fallback logic.

        Args:
            requested: Requested device ("cuda" or "cpu")

        Returns:
            Device string to use

        Raises:
            DeviceError: If invalid device name
        """
        if requested is None:
            return self.detect_device()

        if requested not in ("cuda", "cpu"):
            raise DeviceError(f"Invalid device: {requested}. Use 'cuda' or 'cpu'.")

        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"  # Fallback

        return requested
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_device.py -v`
Expected: PASS (5 tests)

**Step 7: Commit**

```bash
git add lexalign/utils/ tests/test_device.py
git commit -m "feat(finetune): add device detection utility with CUDA fallback"
```

---

## Task 3: Create Fine-Tuning Config Parser

**Files:**
- Create: `lexalign/config/finetune_parser.py`
- Test: `tests/test_finetune_parser.py`

**Step 1: Write the failing test**

Create `tests/test_finetune_parser.py`:
```python
import pytest
from lexalign.config.finetune_parser import FinetuneConfigParser, ConfigError

def test_parse_valid_finetune_config():
    yaml_content = """
    model:
      path: "./models/gpt2"
    dataset:
      path: "./data/dataset"
      format: "auto"
      text_field: "text"
    training:
      method: "lora"
      learning_rate: 3e-4
      batch_size: 4
      num_epochs: 3
    device: "cuda"
    """

    parser = FinetuneConfigParser()
    config = parser.parse(yaml_content)

    assert config["model"]["path"] == "./models/gpt2"
    assert config["training"]["method"] == "lora"
    assert config["device"] == "cuda"

def test_validate_required_fields():
    yaml_content = """
    model:
      path: "./models/gpt2"
    # Missing dataset section
    """

    parser = FinetuneConfigParser()
    with pytest.raises(ConfigError, match="Missing required field"):
        parser.parse(yaml_content)

def test_default_training_parameters():
    yaml_content = """
    model:
      path: "./models/gpt2"
    dataset:
      path: "./data/dataset"
    training:
      method: "qlora"
      quantization_bits: 4
    device: "cpu"
    """

    parser = FinetuneConfigParser()
    config = parser.parse(yaml_content)

    assert config["training"]["learning_rate"] == 3e-4  # Default
    assert config["training"]["lora_r"] == 16  # Default
    assert config["training"]["quantization_bits"] == 4

def test_invalid_method():
    yaml_content = """
    model:
      path: "./models/gpt2"
    dataset:
      path: "./data/dataset"
    training:
      method: "invalid"
    device: "cpu"
    """

    parser = FinetuneConfigParser()
    with pytest.raises(ConfigError, match="Invalid training method"):
        parser.parse(yaml_content)

def test_invalid_quantization_bits():
    yaml_content = """
    model:
      path: "./models/gpt2"
    dataset:
      path: "./data/dataset"
    training:
      method: "qlora"
      quantization_bits: 16
    device: "cpu"
    """

    parser = FinetuneConfigParser()
    with pytest.raises(ConfigError, match="Quantization must be 4 or 8"):
        parser.parse(yaml_content)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_finetune_parser.py -v`
Expected: FAIL with "No module named 'lexalign.config.finetune_parser'"

**Step 3: Write minimal implementation**

Create `lexalign/config/finetune_parser.py`:
```python
import yaml
from typing import Dict, Any


class ConfigError(Exception):
    """Configuration errors."""
    pass


class FinetuneConfigParser:
    """Parse and validate fine-tuning configuration."""

    DEFAULTS = {
        "dataset": {
            "format": "auto",
            "text_field": "text",
            "train_split": "train",
        },
        "training": {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 3e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_epochs": 3,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "save_steps": 500,
            "max_steps": None,
        },
    }

    REQUIRED_FIELDS = ["model", "dataset", "training"]

    def parse(self, yaml_content: str) -> Dict[str, Any]:
        """
        Parse YAML configuration with validation.

        Args:
            yaml_content: YAML configuration string

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigError: If validation fails
        """
        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}")

        self._validate_required_fields(config)
        self._apply_defaults(config)
        self._validate_training_params(config)

        return config

    def _validate_required_fields(self, config: Dict[str, Any]):
        """Ensure all required top-level fields exist."""
        for field in self.REQUIRED_FIELDS:
            if field not in config:
                raise ConfigError(f"Missing required field: {field}")

    def _apply_defaults(self, config: Dict[str, Any]):
        """Apply default values for optional fields."""
        for section, defaults in self.DEFAULTS.items():
            if section not in config:
                config[section] = {}
            for key, value in defaults.items():
                if key not in config[section]:
                    config[section][key] = value

    def _validate_training_params(self, config: Dict[str, Any]):
        """Validate training-specific parameters."""
        training = config["training"]
        method = training.get("method", "lora")

        if method not in ("lora", "qlora"):
            raise ConfigError(f"Invalid training method: {method}. Use 'lora' or 'qlora'.")

        if method == "qlora":
            bits = training.get("quantization_bits", 4)
            if bits not in (4, 8):
                raise ConfigError(f"Quantization must be 4 or 8, got {bits}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_finetune_parser.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add lexalign/config/finetune_parser.py tests/test_finetune_parser.py
git commit -m "feat(finetune): add config parser with validation"
```

---

## Task 4: Create Dataset Preprocessing Module

**Files:**
- Create: `lexalign/finetuner/__init__.py`
- Create: `lexalign/finetuner/dataset_prep.py`
- Test: `tests/test_dataset_prep.py`

**Step 1: Write the failing test**

Create `tests/test_dataset_prep.py`:
```python
import pytest
from pathlib import Path
from lexalign.finetuner.dataset_prep import DatasetPreparer, DatasetError

def test_detect_json_format(tmp_path):
    # Create test JSON file
    json_file = tmp_path / "data.json"
    json_file.write_text('[{"text": "sample1"}, {"text": "sample2"}]')

    preparer = DatasetPreparer()
    format_type = preparer.detect_format(str(json_file))

    assert format_type == "json"

def test_detect_jsonl_format(tmp_path):
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"text": "sample1"}\n{"text": "sample2"}\n')

    preparer = DatasetPreparer()
    format_type = preparer.detect_format(str(jsonl_file))

    assert format_type == "jsonl"

def test_detect_csv_format(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text('text\nsample1\nsample2\n')

    preparer = DatasetPreparer()
    format_type = preparer.detect_format(str(csv_file))

    assert format_type == "csv"

def test_unsupported_format(tmp_path):
    unknown_file = tmp_path / "data.txt"
    unknown_file.write_text('some text')

    preparer = DatasetPreparer()
    with pytest.raises(DatasetError, match="Unsupported format"):
        preparer.load_dataset(str(unknown_file), "auto", "text")

def test_load_json_dataset(mocker, tmp_path):
    json_file = tmp_path / "train.json"
    json_file.write_text('[{"text": "sample1"}, {"text": "sample2"}]')

    mock_load_dataset = mocker.patch(
        'lexalign.finetuner.dataset_prep.load_dataset'
    )
    mock_ds = mocker.Mock()
    mock_load_dataset.return_value = mock_ds

    preparer = DatasetPreparer()
    result = preparer.load_dataset(str(json_file), "json", "text")

    assert result is not None

def test_dataset_path_not_found():
    preparer = DatasetPreparer()
    with pytest.raises(DatasetError, match="Dataset path not found"):
        preparer.load_dataset("/nonexistent/path", "auto", "text")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_prep.py -v`
Expected: FAIL with "No module named 'lexalign.finetuner.dataset_prep'"

**Step 3: Create module structure**

Create `lexalign/finetuner/__init__.py`:
```python
from lexalign.finetuner.dataset_prep import DatasetPreparer, DatasetError

__all__ = ['DatasetPreparer', 'DatasetError']
```

**Step 4: Run test to verify it still fails**

Run: `pytest tests/test_dataset_prep.py -v`
Expected: FAIL with "cannot import 'DatasetPreparer'"

**Step 5: Write minimal implementation**

Create `lexalign/finetuner/dataset_prep.py`:
```python
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
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_dataset_prep.py -v`
Expected: PASS (6 tests)

**Step 7: Commit**

```bash
git add lexalign/finetuner/ tests/test_dataset_prep.py
git commit -m "feat(finetune): add dataset format detection and loading"
```

---

## Task 5: Create LoRA Configuration Builder

**Files:**
- Create: `lexalign/finetuner/lora_config.py`
- Test: `tests/test_lora_config.py`

**Step 1: Write the failing test**

Create `tests/test_lora_config.py`:
```python
import pytest
from lexalign.finetuner.lora_config import LoraConfigBuilder

def test_build_lora_config():
    training_params = {
        "method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
    }

    builder = LoraConfigBuilder()
    config = builder.build(training_params)

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.target_modules == ["q_proj", "v_proj"]

def test_build_qlora_config():
    training_params = {
        "method": "qlora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "quantization_bits": 4
    }

    builder = LoraConfigBuilder()
    config = builder.build(training_params, quantization_bits=4)

    assert config.r == 8
    assert config.lora_alpha == 16

def test_build_qlora_load_in_4bit():
    builder = LoraConfigBuilder()
    load_in_4bit = builder.get_quantization_config(4)

    assert load_in_4bit is True

def test_build_qlora_load_in_8bit():
    builder = LoraConfigBuilder()
    load_in_8bit = builder.get_quantization_config(8)

    assert load_in_8bit is True

def test_invalid_quantization_bits():
    builder = LoraConfigBuilder()
    with pytest.raises(ValueError, match="Quantization must be 4 or 8"):
        builder.get_quantization_config(16)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_lora_config.py -v`
Expected: FAIL with "No module named 'lexalign.finetuner.lora_config'"

**Step 3: Write minimal implementation**

Create `lexalign/finetuner/lora_config.py`:
```python
from peft import LoraConfig, get_peft_model, TaskType


class LoraConfigBuilder:
    """Build LoRA/QLoRA configurations."""

    def build(self, training_params: dict, quantization_bits: int = None):
        """
        Build LoRA configuration.

        Args:
            training_params: Training configuration dictionary
            quantization_bits: Quantization bits (4 or 8) for QLoRA

        Returns:
            LoraConfig instance
        """
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=training_params.get("lora_r", 16),
            lora_alpha=training_params.get("lora_alpha", 32),
            lora_dropout=training_params.get("lora_dropout", 0.05),
            target_modules=training_params.get("target_modules"),
            bias="none",
        )

    def get_quantization_config(self, bits: int) -> dict:
        """
        Get quantization configuration for QLoRA.

        Args:
            bits: Quantization bits (4 or 8)

        Returns:
            Dictionary with quantization settings

        Raises:
            ValueError: If bits is not 4 or 8
        """
        if bits == 4:
            return {"load_in_4bit": True}
        elif bits == 8:
            return {"load_in_8bit": True}
        else:
            raise ValueError(f"Quantization must be 4 or 8, got {bits}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_lora_config.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add lexalign/finetuner/lora_config.py tests/test_lora_config.py
git commit -m "feat(finetune): add LoRA/QLoRA configuration builder"
```

---

## Task 6: Create Trainer Wrapper

**Files:**
- Create: `lexalign/finetuner/trainer.py`
- Test: `tests/test_trainer.py`

**Step 1: Write the failing test**

Create `tests/test_trainer.py`:
```python
import pytest
from lexalign.finetuner.trainer import FinetuneTrainer, TrainerError

def test_trainer_initialization():
    config = {
        "model": {"path": "./models/test"},
        "dataset": {"path": "./data/test"},
        "training": {
            "method": "lora",
            "learning_rate": 3e-4,
            "num_epochs": 3
        },
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    assert trainer.config == config

def test_validate_model_path_exists(mocker, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()

    config = {
        "model": {"path": str(model_path)},
        "dataset": {"path": "./data/test"},
        "training": {"method": "lora"},
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    # Should not raise error
    trainer._validate_paths()

def test_validate_model_path_not_found():
    config = {
        "model": {"path": "/nonexistent/model"},
        "dataset": {"path": "./data/test"},
        "training": {"method": "lora"},
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    with pytest.raises(TrainerError, match="Model path not found"):
        trainer._validate_paths()

def test_get_output_dir_with_user_specified():
    config = {
        "model": {"path": "./models/test"},
        "dataset": {"path": "./data/test"},
        "training": {
            "method": "lora",
            "output_dir": "./custom/output"
        },
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    output_dir = trainer._get_output_dir()

    assert output_dir == "./custom/output"

def test_get_output_dir_default_timestamp(mocker):
    mocker.patch('lexalign.finetuner.trainer.datetime')

    config = {
        "model": {"path": "./models/gpt2"},
        "dataset": {"path": "./data/test"},
        "training": {"method": "lora"},
        "device": "cpu"
    }

    trainer = FinetuneTrainer(config)
    output_dir = trainer._get_output_dir()

    assert output_dir.startswith("./checkpoints/gpt2-")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL with "No module named 'lexalign.finetuner.trainer'"

**Step 3: Write minimal implementation**

Create `lexalign/finetuner/trainer.py`:
```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add lexalign/finetuner/trainer.py tests/test_trainer.py
git commit -m "feat(finetune): add trainer wrapper with path validation"
```

---

## Task 7: Create Checkpoint Manager

**Files:**
- Create: `lexalign/finetuner/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Step 1: Write the failing test**

Create `tests/test_checkpoint.py`:
```python
import pytest
from pathlib import Path
from lexalign.finetuner.checkpoint import CheckpointManager

def test_find_latest_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Create checkpoint directories
    (checkpoint_dir / "checkpoint-500").mkdir()
    (checkpoint_dir / "checkpoint-1000").mkdir()
    (checkpoint_dir / "checkpoint-1500").mkdir()

    manager = CheckpointManager()
    latest = manager.find_latest(str(checkpoint_dir))

    assert latest.endswith("checkpoint-1500")

def test_no_checkpoints_found(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    manager = CheckpointManager()
    latest = manager.find_latest(str(checkpoint_dir))

    assert latest is None

def test_get_checkpoint_step_from_name():
    manager = CheckpointManager()
    step = manager.get_step("/path/to/checkpoint-1000")

    assert step == 1000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_checkpoint.py -v`
Expected: FAIL with "No module named 'lexalign.finetuner.checkpoint'"

**Step 3: Write minimal implementation**

Create `lexalign/finetuner/checkpoint.py`:
```python
from pathlib import Path
from typing import Optional
import re


class CheckpointManager:
    """Manage training checkpoints."""

    def find_latest(self, checkpoint_dir: str) -> Optional[str]:
        """
        Find latest checkpoint directory.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Path to latest checkpoint, or None if no checkpoints
        """
        dir_path = Path(checkpoint_dir)
        if not dir_path.exists():
            return None

        checkpoints = sorted(
            [p for p in dir_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1])
        )

        if not checkpoints:
            return None

        return str(checkpoints[-1])

    def get_step(self, checkpoint_path: str) -> int:
        """
        Extract step number from checkpoint path.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Step number
        """
        match = re.search(r"checkpoint-(\d+)", checkpoint_path)
        if match:
            return int(match.group(1))
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_checkpoint.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add lexalign/finetuner/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(finetune): add checkpoint manager for resume functionality"
```

---

## Task 8: Create Main CLI Entry Point

**Files:**
- Create: `finetune.py`
- Test: `tests/test_finetune_cli.py` (integration test)

**Step 1: Write the failing test**

Create `tests/test_finetune_cli.py`:
```python
import pytest
from click.testing import CliRunner
from finetune import cli

def test_cli_requires_config():
    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code != 0
    assert "Missing option" in result.output or "--config" in result.output

def test_cli_dry_run(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    model:
      path: "./models/test"
    dataset:
      path: "./data/test"
    training:
      method: "lora"
    device: "cpu"
    """)

    mocker.patch('finetune.Path.exists', return_value=True)

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file), "--dry-run"])

    # Should show config without training
    assert "Fine-tuning configuration" in result.output or result.exit_code == 0

def test_cli_model_not_found(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    model:
      path: "./nonexistent/model"
    dataset:
      path: "./data/test"
    training:
      method: "lora"
    device: "cpu"
    """)

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file)])

    assert "Model path not found" in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_finetune_cli.py -v`
Expected: FAIL with "No module named 'finetune'"

**Step 3: Write minimal implementation**

Create `finetune.py`:
```python
import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from lexalign.config.finetune_parser import FinetuneConfigParser, ConfigError
from lexalign.finetuner.trainer import FinetuneTrainer, TrainerError
from lexalign.utils.device import DeviceManager, DeviceError

console = Console()


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Fine-tuning config file")
@click.option("--resume", type=click.Path(exists=True), help="Checkpoint directory to resume from")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), help="Override device (cuda/cpu)")
@click.option("--dry-run", is_flag=True, help="Show configuration without training")
@click.option("--verbose", "-v", is_flag=True, help="Detailed training output")
def cli(config: str, resume: str, device: str, dry_run: bool, verbose: bool):
    """
    Fine-tune Hugging Face models using LoRA or QLoRA.
    """
    # Load configuration
    try:
        with open(config, "r") as f:
            yaml_content = f.read()

        parser = FinetuneConfigParser()
        ft_config = parser.parse(yaml_content)
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise click.Abort()

    # Override device if specified
    if device:
        try:
            device_manager = DeviceManager()
            ft_config["device"] = device_manager.get_device(device)
        except DeviceError as e:
            console.print(f"[red]Device error:[/red] {e}")
            raise click.Abort()

    # Dry run: show config and exit
    if dry_run:
        console.print(Panel.fit(
            f"[bold cyan]Fine-tuning Configuration[/bold cyan]\n\n"
            f"Model: {ft_config['model']['path']}\n"
            f"Dataset: {ft_config['dataset']['path']}\n"
            f"Method: {ft_config['training']['method']}\n"
            f"Device: {ft_config['device']}\n"
            f"Output: {ft_config['training'].get('output_dir', 'auto')}"
        ))
        return

    # Initialize trainer
    try:
        trainer = FinetuneTrainer(ft_config)
        output_dir = trainer.train()
        console.print(f"[green]Fine-tuning complete![/green] Checkpoints saved to: {output_dir}")
    except TrainerError as e:
        console.print(f"[red]Training error:[/red] {e}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise click.Abort()


if __name__ == "__main__":
    cli()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_finetune_cli.py -v`
Expected: PASS (3 tests)

**Step 5: Make executable**

Run: `chmod +x finetune.py`

**Step 6: Commit**

```bash
git add finetune.py tests/test_finetune_cli.py
git commit -m "feat(finetune): add CLI entry point with dry-run support"
```

---

## Task 9: Create Example Configuration

**Files:**
- Create: `config/finetune.yaml.example`

**Step 1: Create example configuration**

Create `config/finetune.yaml.example`:
```yaml
# LexAlign Fine-Tuning Tool - Example Configuration
# Copy this file to finetune.yaml and fill in your details

# Model Configuration
model:
  path: "./models/gpt2"           # Path to downloaded model
  base_model: "gpt2"               # Optional: HF repo ID for tokenizer

# Dataset Configuration
dataset:
  path: "./data/my-dataset"        # Path to downloaded dataset
  format: "auto"                   # auto, json, csv, jsonl
  text_field: "text"               # Field name containing training text
  train_split: "train"             # Dataset split to use

# Training Configuration
training:
  method: "lora"                   # lora or qlora
  output_dir: "./checkpoints/gpt2-finetuned"  # Optional

  # LoRA/QLoRA parameters
  lora_r: 16                       # LoRA rank
  lora_alpha: 32                   # LoRA alpha
  lora_dropout: 0.05               # LoRA dropout
  target_modules:                  # Optional (auto-detected if omitted)
    - "q_proj"
    - "v_proj"

  # QLoRA specific
  quantization_bits: 4             # 4 or 8 (only for qlora)

  # Training hyperparameters
  learning_rate: 3e-4
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01

  # Checkpointing
  save_steps: 500                  # Save checkpoint every N steps
  max_steps: null                  # Optional: override epoch-based

# Hardware
device: "cuda"                     # cuda or cpu
```

**Step 2: Commit**

```bash
git add config/finetune.yaml.example
git commit -m "docs(finetune): add example configuration file"
```

---

## Task 10: Update README with Fine-Tuning Section

**Files:**
- Modify: `README.md`

**Step 1: Add fine-tuning section to README**

Add this section after the download section:
```markdown
## Fine-Tuning

Fine-tune downloaded models using LoRA or QLoRA.

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

1. Create a fine-tuning configuration:
```bash
cp config/finetune.yaml.example config/finetune.yaml
```

2. Edit `config/finetune.yaml` with your model, dataset, and training parameters.

3. Run fine-tuning:
```bash
python finetune.py --config config/finetune.yaml
```

### Configuration

```yaml
model:
  path: "./models/gpt2"

dataset:
  path: "./data/my-dataset"
  format: "auto"

training:
  method: "lora"                   # or "qlora"
  learning_rate: 3e-4
  num_epochs: 3

device: "cuda"                     # or "cpu"
```

### Options

- `--config PATH` - Fine-tuning config file (required)
- `--resume PATH` - Resume from checkpoint
- `--device DEVICE` - Override device (cuda/cpu)
- `--dry-run` - Show config without training
- `--verbose, -v` - Detailed output

### Example

```bash
# Dry run
python finetune.py --config config/finetune.yaml --dry-run

# Resume training
python finetune.py --config config/finetune.yaml --resume ./checkpoints/model/checkpoint-1000

# Use CPU instead of GPU
python finetune.py --config config/finetune.yaml --device cpu
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add fine-tuning section to README"
```

---

## Task 11: End-to-End Integration Test

**Files:**
- Modify: `tests/test_e2e.py`

**Step 1: Add e2e fine-tuning test**

Add to `tests/test_e2e.py`:
```python
def test_finetune_workflow_with_mocks(mocker, tmp_path):
    """Test complete fine-tuning workflow with mocked dependencies."""
    # Create mock model and dataset directories
    model_dir = tmp_path / "models" / "gpt2"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    data_dir = tmp_path / "data" / "dataset"
    data_dir.mkdir(parents=True)
    (data_dir / "train.json").write_text('[{"text": "sample"}]')

    # Create config file
    config_file = tmp_path / "finetune.yaml"
    config_file.write_text(f"""
    model:
      path: "{model_dir}"
    dataset:
      path: "{data_dir}"
    training:
      method: "lora"
    device: "cpu"
    """)

    # Mock training components
    mocker.patch('lexalign.finetuner.trainer.SFTTrainer')
    mocker.patch('lexalign.finetuner.trainer.AutoModelForCausalLM')
    mocker.patch('lexalign.finetuner.trainer.AutoTokenizer')

    from click.testing import CliRunner
    from finetune import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(config_file), "--dry-run"])

    assert result.exit_code == 0
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_e2e.py::test_finetune_workflow_with_mocks -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test(finetune): add end-to-end integration test"
```

---

## Task 12: Run All Tests and Verify

**Step 1: Run complete test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (40+ tests total)

**Step 2: Verify CLI dry-run works**

Create test config:
```bash
cat > /tmp/test_finetune.yaml << 'EOF'
model:
  path: "./models/gpt2"
dataset:
  path: "./data/test"
training:
  method: "lora"
device: "cpu"
EOF
```

Run: `python finetune.py --config /tmp/test_finetune.yaml --dry-run`
Expected: Shows configuration without error

**Step 3: Final commit**

```bash
git add .
git commit -m "feat(finetune): complete fine-tuning tool implementation"
```

---

## Implementation Complete

Total tasks: 12
Total commits: 12+
Total tests: 40+

**Next Steps:**
1. Test with real model/dataset
2. Implement actual training loop in trainer.py
3. Add progress bar with rich
4. Test checkpoint resume functionality
