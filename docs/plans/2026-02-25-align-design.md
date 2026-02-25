# LexAlign DPO/GDPO Alignment Tool - Design Document

**Date:** 2026-02-25
**Status:** Approved

## Overview

A Python-based CLI tool that aligns already-fine-tuned models using preference datasets with DPO (Direct Preference Optimization) and GDPO (Group Delay Policy Optimization) methods. Supports configurable hyperparameters, auto-detection of dataset formats, and LoRA for efficient training. Targeted at personal experimentation with different alignment methods.

## Architecture

### Components

- `align.py` - Main CLI entry point with Click commands
- `lexalign/aligner/` - Core alignment logic module
  - `dpo_trainer.py` - Wrapper around TRL's DPOTrainer
  - `gdpo_trainer.py` - Custom GDPO implementation (extends DPO with group delay handling)
  - `dataset_prep.py` - Preference dataset preprocessing and validation
  - `checkpoint.py` - Checkpoint management (reused from finetuner)
- `lexalign/config/` - Extended configuration handling
  - `align_parser.py` - Parse alignment-specific YAML config
- `lexalign/utils/` - Shared utilities
  - `device.py` - Device detection and management (existing)

### Data Flow

```
CLI args → Align config parser → Preference dataset validation → DPO/GDPO config → Trainer → Checkpointing → Aligned model output
```

### Workflow

```
download.py → finetune.py → align.py
   ↓            ↓            ↓
 get data    SFT train    Preference align
```

### Error Handling

- **Model not found** → Clear error with path, suggest running finetune first
- **Dataset missing required fields** → List missing fields (prompt/chosen/rejected), show expected format
- **Unsupported dataset format** → List supported formats, show conversion example
- **GPU not available when specified** → Graceful fallback to CPU with warning
- **Training interrupted** → Save checkpoint, provide resume command
- **Insufficient memory** → Suggest reducing batch size, enabling LoRA, or using gradient accumulation
- **Invalid group_delay_size** → Error if value < 2 (need at least chosen + rejected)

## Configuration Schema

### YAML Structure

```yaml
# config/align.yaml
model:
  path: "./checkpoints/gpt2-finetuned"  # Path to fine-tuned model
  base_model: "gpt2"                     # Optional: HF repo ID for tokenizer

dataset:
  path: "./data/preference-dataset"      # Path to preference dataset
  format: "auto"                         # auto, json, csv, jsonl
  prompt_field: "prompt"                 # Field containing prompt
  chosen_field: "chosen"                 # Field containing chosen response
  rejected_field: "rejected"             # Field containing rejected response
  train_split: "train"                   # Dataset split to use

alignment:
  method: "dpo"                          # dpo or gdpo
  output_dir: "./checkpoints/gpt2-aligned"  # Optional, defaults to ./checkpoints/<model>-aligned-<timestamp>

  # DPO/GDPO parameters
  beta: 0.1                              # DPO beta parameter (temperature)
  loss_type: "sigmoid"                   # sigmoid, hinge, or ipo (for DPO)

  # GDPO specific (only used if method: gdpo)
  group_delay_size: 4                    # Number of responses to rank per prompt
  group_delay_weight: 0.5                # Weight for group delay loss

  # LoRA/QLoRA for alignment (optional, can also train full model)
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"

  # Training hyperparameters
  learning_rate: 1e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01

  # Checkpointing
  save_steps: 500
  max_steps: null                        # Optional: override epoch-based training

# Hardware
device: "cuda"                           # cuda or cpu
```

### Key Features

- Input model = fine-tuned model (output from finetune.py)
- Standard preference format (prompt, chosen, rejected)
- Both LoRA and full fine-tuning supported during alignment
- GDPO extends DPO with group-specific parameters
- Auto-detection of dataset format like finetune tool

## CLI Interface

### Command Structure

```bash
# Basic alignment with DPO
python align.py --config config/align.yaml

# Use GDPO instead
python align.py --config config/align-gdpo.yaml

# Override device
python align.py --config config/align.yaml --device cpu

# Resume from checkpoint
python align.py --config config/align.yaml --resume ./checkpoints/gpt2-aligned/checkpoint-1000

# Verbose output
python align.py --config config/align.yaml --verbose

# Dry run (show config without training)
python align.py --config config/align.yaml --dry-run
```

### Options

- `--config PATH` - Alignment config file (required)
- `--resume PATH` - Checkpoint directory to resume from
- `--device DEVICE` - Override device (cuda/cpu)
- `--dry-run` - Show configuration without training
- `--verbose, -v` - Detailed training output

### Output During Training

```
Epoch 1/3: [████████░░░░░░░░░] 60% | Loss: 0.823 | Policy Loss: 0.651 | Reward: 0.172 | ETA: 8m 15s
```

## Component Implementation Details

### Dataset Preparation (`dataset_prep.py`)

```python
class PreferenceDataset:
    def load_and_validate(self, path: str, config: dict) -> Dataset:
        # Auto-detect format like finetune tool
        # Validate required fields: prompt, chosen, rejected
        # Return formatted Dataset for DPOTrainer
```

### DPO Trainer (`dpo_trainer.py`)

```python
class DPOTrainerWrapper:
    def __init__(self, model, ref_model, tokenizer, config):
        self.trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            beta=config["beta"],
            loss_type=config["loss_type"],
            # ... standard DPO params
        )
```

### GDPO Trainer (`gdpo_trainer.py`)

```python
class GDPOTrainerWrapper:
    def __init__(self, model, ref_model, tokenizer, config):
        # Extend DPO with group delay logic
        self.group_delay_size = config["group_delay_size"]
        self.group_delay_weight = config["group_delay_weight"]
        # Custom loss combining DPO + group delay ranking
```

### Key Implementation Notes

- Reference model = frozen copy of base model for DPO/GDPO
- GDPO implementation = modify DPO loss to handle group ranking
- LoRA during alignment = optional but recommended for efficiency
- Reuse checkpoint.py from finetuner for consistency

## Dependencies

### Required Packages

```python
# No NEW dependencies needed - DPOTrainer is in TRL
trl>=0.9.0              # Already installed - has DPOTrainer
peft>=0.7.0             # Already installed - for LoRA
transformers>=4.36.0    # Already installed
datasets>=2.14.0        # Already installed
accelerate>=0.25.0      # Already installed
```

**Python Version:** 3.9+

### Rationale

- TRL already includes `DPOTrainer` - no extra dependencies
- GDPO will be custom implementation using same TRL base
- All other packages already installed for fine-tuning

## File Structure

```
LexAlign/
├── README.md                          # Updated with alignment section
├── requirements.txt                   # No changes (TRL already has DPO)
├── download.py                        # Existing download CLI
├── finetune.py                        # Existing fine-tuning CLI
├── align.py                           # NEW: Alignment CLI
├── lexalign/
│   ├── __init__.py
│   ├── downloader/                    # Existing download modules
│   ├── config/
│   │   ├── __init__.py
│   │   ├── parser.py                  # Existing config parser
│   │   ├── finetune_parser.py         # Existing fine-tuning config
│   │   └── align_parser.py            # NEW: Alignment config parser
│   ├── finetuner/                     # Existing fine-tuning module
│   │   ├── trainer.py
│   │   ├── lora_config.py
│   │   ├── dataset_prep.py
│   │   └── checkpoint.py
│   ├── aligner/                       # NEW: Alignment module
│   │   ├── __init__.py
│   │   ├── dpo_trainer.py             # DPOTrainer wrapper
│   │   ├── gdpo_trainer.py            # Custom GDPO implementation
│   │   ├── dataset_prep.py            # Preference dataset preprocessing
│   │   └── checkpoint.py              # Symlink or reuse from finetuner
│   └── utils/                         # Existing shared utilities
│       ├── __init__.py
│       └── device.py
├── config/
│   ├── downloads.yaml.example         # Existing
│   ├── finetune.yaml.example          # Existing
│   └── align.yaml.example             # NEW: Alignment config template
├── tests/
│   ├── test_*.py                      # Existing tests
│   ├── test_align_parser.py           # NEW
│   ├── test_align_dataset_prep.py     # NEW
│   ├── test_dpo_trainer.py            # NEW (with mocked training)
│   ├── test_gdpo_trainer.py           # NEW
│   └── test_align_cli.py              # NEW
└── docs/
    └── plans/
        ├── 2026-02-25-finetune-design.md      # Existing
        └── 2026-02-25-align-design.md         # This design doc
```

## Testing Strategy

### Unit Tests

- `test_align_parser.py` - Config parsing and validation
- `test_align_dataset_prep.py` - Dataset format detection and validation
- `test_dpo_trainer.py` - DPOTrainer wrapper with mocked training
- `test_gdpo_trainer.py` - GDPO loss calculation with mocked model

### Integration Tests

- `test_align_e2e.py` - Full workflow with mocked training step
- `test_align_cli.py` - CLI argument parsing and dry-run mode

### Test Coverage Goals

- Config parsing: all fields, defaults, error cases
- Dataset validation: missing fields, wrong formats, auto-detection
- Trainer initialization: DPO vs GDPO, LoRA vs full model
- Error handling: all error scenarios from Section 7

### Mocking Strategy

- Mock `DPOTrainer` from TRL (don't actually train)
- Mock Hugging Face model loading
- Mock dataset loading (use small synthetic datasets)
- Test GDPO loss calculation with dummy tensors

## Error Scenarios

1. **Model path doesn't exist**
   ```
   Error: Model not found at "./checkpoints/gpt2-finetuned"
   → Run: python finetune.py --config config/finetune.yaml
   ```

2. **Dataset missing required fields**
   ```
   Error: Dataset missing required field "rejected"
   Available fields: prompt, chosen, meta
   → Expected format: {prompt: str, chosen: str, rejected: str}
   ```

3. **Dataset has wrong format**
   ```
   Error: Unsupported dataset format "txt"
   Supported formats: auto, json, csv, jsonl
   ```

4. **CUDA requested but unavailable**
   ```
   Warning: CUDA requested but unavailable. Using CPU.
   Training will be significantly slower.
   ```

5. **Invalid group_delay_size**
   ```
   Error: group_delay_size must be >= 2 (got: 1)
   → Need at least chosen + rejected responses
   ```

6. **Training interrupted (Ctrl+C)**
   ```
   Training interrupted. Checkpoint saved to: ./checkpoints/gpt2-aligned/checkpoint-500
   Resume with: python align.py --config config/align.yaml --resume ./checkpoints/gpt2-aligned/checkpoint-500
   ```

7. **Insufficient memory**
   ```
   Error: CUDA out of memory
   → Try: reducing batch_size, enabling use_lora, or using gradient_accumulation_steps
   ```

## Success Criteria

### Must Have

- [ ] Align fine-tuned models using both DPO and GDPO
- [ ] Support preference datasets (prompt, chosen, rejected fields)
- [ ] Auto-detect dataset formats (JSON, CSV, JSONL)
- [ ] Configurable hyperparameters (beta, learning_rate, batch_size, epochs, etc.)
- [ ] LoRA support during alignment (optional)
- [ ] GDPO-specific parameters (group_delay_size, group_delay_weight)
- [ ] Save checkpoints and resume training via `--resume` flag
- [ ] Device selection (`--device cuda/cpu`)
- [ ] Clear error if model/dataset not found
- [ ] YAML-based configuration separate from downloads/finetune configs
- [ ] Display training progress (loss, policy loss, reward metrics)

### Nice to Have

- [ ] Support multiple loss types (sigmoid, hinge, IPO)
- [ ] Validation split for evaluation during training
- [ ] Generate sample alignment dataset for testing
- [ ] Compare DPO vs GDPO results in same run

### Completed when

All "Must Have" criteria satisfied, tests passing, documented in README.
