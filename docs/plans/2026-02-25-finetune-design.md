# LexAlign Fine-Tuning Tool - Design Document

**Date:** 2026-02-25
**Status:** Approved

## Overview

A Python-based CLI tool that fine-tunes already-downloaded Hugging Face models using user-provided datasets. Supports LoRA and QLoRA training methods with configurable hyperparameters, auto-detection of dataset formats, and checkpoint-based resumption. Targeted at personal experimentation with different model/dataset combinations.

## Architecture

### Components

- `finetune.py` - Main CLI entry point with Click commands
- `lexalign/finetuner/` - Core fine-tuning logic module
  - `trainer.py` - Wrapper around TRL's SFTTrainer
  - `lora_config.py` - LoRA/QLoRA configuration builder
  - `dataset_prep.py` - Dataset format auto-detection and preprocessing
  - `checkpoint.py` - Checkpoint management and resumption
- `lexalign/config/` - Extended configuration handling
  - `finetune_parser.py` - Parse fine-tuning specific YAML config
- `lexalign/utils/` - Shared utilities
  - `device.py` - Device detection and management

### Data Flow

```
CLI args → Finetune config parser → Dataset auto-detection → Format conversion → LoRA/QLoRA config → SFTTrainer → Checkpointing → Model output
```

### Error Handling

- **Model/dataset not found** → Clear error with path, suggest running download
- **Unsupported dataset format** → List supported formats, show conversion example
- **GPU not available when specified** → Graceful fallback to CPU with warning
- **Training interrupted** → Save checkpoint, provide resume command
- **Insufficient memory** → Suggest reducing batch size or using gradient accumulation

## Configuration Schema

### YAML Structure

```yaml
# config/finetune.yaml
model:
  path: "./models/gpt2"           # Path to downloaded model
  base_model: "gpt2"               # Optional: HF repo ID for tokenizer loading

dataset:
  path: "./data/my-dataset"        # Path to downloaded dataset
  format: "auto"                   # auto, json, csv, jsonl
  text_field: "text"               # Field name containing training text
  train_split: "train"             # Dataset split to use

training:
  method: "lora"                   # lora or qlora
  output_dir: "./checkpoints/gpt2-finetuned"  # Optional, defaults to ./checkpoints/<model>-<timestamp>

  # LoRA/QLoRA parameters
  lora_r: 16                       # LoRA rank
  lora_alpha: 32                   # LoRA alpha
  lora_dropout: 0.05               # LoRA dropout
  target_modules:                  # Modules to apply LoRA (optional, auto-detected if omitted)
    - "q_proj"
    - "v_proj"

  # QLoRA specific (only used if method: qlora)
  quantization_bits: 4             # 4 or 8

  # Training hyperparameters
  learning_rate: 3e-4
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01

  # Checkpointing
  save_steps: 500                  # Save checkpoint every N steps
  max_steps: null                  # Optional: override epoch-based training

# Hardware
device: "cuda"                     # cuda or cpu
```

### Key Features

- Auto-detection of dataset format with override option
- Sensible defaults for LoRA parameters
- QLoRA quantization bit selection (4 or 8)
- Per-checkpoint saving control
- Output directory defaults to timestamp-based if not specified

## CLI Interface

### Command Structure

```bash
# Basic fine-tuning
python finetune.py --config config/finetune.yaml

# Override device
python finetune.py --config config/finetune.yaml --device cpu

# Resume from checkpoint
python finetune.py --config config/finetune.yaml --resume ./checkpoints/gpt2-finetuned/checkpoint-1000

# Verbose output
python finetune.py --config config/finetune.yaml --verbose

# Dry run (show config without training)
python finetune.py --config config/finetune.yaml --dry-run
```

### Options

- `--config PATH` - Fine-tuning config file (required)
- `--resume PATH` - Checkpoint directory to resume from
- `--device DEVICE` - Override device (cuda/cpu)
- `--dry-run` - Show configuration without training
- `--verbose, -v` - Detailed training output

### Output During Training

```
Epoch 1/3: [████████░░░░░░░░░] 60% | Loss: 2.341 | LR: 3e-4 | ETA: 12m 30s
```

## Dependencies

### Required Packages

```python
# requirements.txt additions
trl>=0.9.0              # SFTTrainer for LoRA fine-tuning
peft>=0.7.0             # LoRA/QLoRA implementation
bitsandbytes>=0.41.0    # 4-bit/8-bit quantization for QLoRA
accelerate>=0.25.0      # Distributed training and device management
datasets>=2.14.0        # Dataset handling and format conversion
transformers>=4.36.0    # Model loading and tokenizer
```

**Python Version:** 3.9+

### Rationale

- `trl` - Provides `SFTTrainer` specifically designed for LLM fine-tuning
- `peft` - Parameter-efficient fine-tuning (LoRA/QLoRA) implementation
- `bitsandbytes` - Efficient quantization for QLoRA (4-bit/8-bit)
- `accelerate` - Handles device placement and mixed precision training
- `datasets` - Auto-detects and converts JSON/CSV/JSONL formats
- `transformers` - Core library for model loading and tokenization

### Existing Dependencies (from download tool)
- `pyyaml`, `click`, `rich` (already present)

## File Structure

```
LexAlign/
├── README.md                          # Updated with fine-tuning section
├── requirements.txt                   # Updated with new dependencies
├── download.py                        # Existing download CLI
├── finetune.py                        # NEW: Fine-tuning CLI
├── lexalign/
│   ├── __init__.py
│   ├── downloader/                    # Existing download modules
│   ├── config/
│   │   ├── __init__.py
│   │   ├── parser.py                  # Existing config parser
│   │   └── finetune_parser.py         # NEW: Fine-tuning config parser
│   ├── finetuner/                     # NEW: Fine-tuning module
│   │   ├── __init__.py
│   │   ├── trainer.py                 # SFTTrainer wrapper
│   │   ├── lora_config.py             # LoRA/QLoRA configuration
│   │   ├── dataset_prep.py            # Dataset preprocessing
│   │   └── checkpoint.py              # Checkpoint management
│   └── utils/                         # NEW: Shared utilities
│       ├── __init__.py
│       └── device.py                  # Device detection and management
├── config/
│   ├── downloads.yaml.example         # Existing
│   └── finetune.yaml.example          # NEW: Fine-tuning config template
├── tests/
│   ├── test_*.py                      # Existing tests
│   ├── test_finetune_parser.py        # NEW
│   ├── test_dataset_prep.py           # NEW
│   ├── test_lora_config.py            # NEW
│   └── test_trainer.py                # NEW (with mocked training)
└── docs/
    └── plans/
        └── 2026-02-25-finetune-design.md  # This design doc
```

## Implementation Approach

**Selected:** TRL + PEFT with separate `finetune.py` CLI

This approach provides:
- Clean separation from download tool (both can evolve independently)
- TRL's `SFTTrainer` handles complex LoRA/QLoRA setup automatically
- Auto-detection of common dataset formats reduces user friction
- Sensible defaults with override options for experimentation
- Checkpoint/resume support for long-running training jobs

## Success Criteria

- [x] Fine-tune downloaded models using LoRA or QLoRA
- [x] Support configurable hyperparameters (learning rate, batch size, epochs, gradient accumulation, warmup, weight decay)
- [x] Auto-detect dataset formats (JSON, CSV, JSONL)
- [x] Save to user-specified or default timestamp-based checkpoint directory
- [x] Display training progress (epoch, loss, learning rate, ETA)
- [x] Configurable checkpoint saving (every N steps)
- [x] Resume training from checkpoint via `--resume` flag
- [x] Allow device selection (`--device cuda/cpu`)
- [x] Clear error if model/dataset not downloaded
- [x] YAML-based configuration separate from downloads config
