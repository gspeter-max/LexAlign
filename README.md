# LexAlign

A CLI tool for downloading models and datasets from Hugging Face using declarative YAML configuration.

## Installation

```bash
pip install -e ".[dev]"
```

## GPU Installation (CUDA)

By default the install above uses CPU-only PyTorch.
To enable GPU training, reinstall `torch` with the matching CUDA index URL **after** running the above:

```bash
# CUDA 12.1 (RTX 30xx/40xx, A100, H100)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older GPUs)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu118
```

> **macOS note:** QLoRA (`method: qlora`) uses `bitsandbytes` which has limited macOS support.
> For reliable QLoRA, use a Linux machine with a CUDA GPU.

## Quick Start

1. Set your Hugging Face token:
```bash
export HF_TOKEN="your_token_here"
```

2. Create a configuration file:
```bash
cp config/downloads.yaml.example config/downloads.yaml
```

3. Edit `config/downloads.yaml` with your desired models/datasets.

4. Run the downloader:
```bash
python download.py --config config/downloads.yaml
```

## Configuration

```yaml
huggingface:
  token: "${HF_TOKEN}"

models:
  - repo: "gpt2"
    files:
      - "config.json"
      - "pytorch_model.bin"
    output_dir: "./models/gpt2"

datasets:
  - repo: "username/dataset"
    files:
      - "data/*.json"
    output_dir: "./data/dataset"
```

## Options

- `--config PATH` - YAML config file (required)
- `--token TOKEN` - Override HF token
- `--dry-run` - Show what would download
- `--models-only` - Skip datasets
- `--datasets-only` - Skip models
- `--verbose` - Detailed output

## Example

```bash
# Dry run to see what will be downloaded
python download.py --config config/downloads.yaml --dry-run

# Download only models
python download.py --config config/downloads.yaml --models-only

# Verbose output
python download.py --config config/downloads.yaml --verbose
```

## Fine-Tuning

Fine-tune downloaded models using LoRA or QLoRA.

> **Security Note:** This tool uses `trust_remote_code=True` when loading tokenizers, which allows models to execute custom code from the Hugging Face Hub. Only fine-tune models from trusted sources.

### Installation

```bash
pip install -e ".[dev]"
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

## Alignment

Align fine-tuned models using DPO (Direct Preference Optimization) or GDPO (Group Delay Policy Optimization) with preference datasets.

### Quick Start

1. Create an alignment configuration:
```bash
cp config/align.yaml.example config/align.yaml
```

2. Edit `config/align.yaml` with your model path and preference dataset.

3. Run alignment:
```bash
python align.py --config config/align.yaml
```

### Configuration

```yaml
model:
  path: "./checkpoints/gpt2-finetuned"  # Your fine-tuned model

dataset:
  path: "./data/preferences"             # Preference dataset
  prompt_field: "prompt"
  chosen_field: "chosen"
  rejected_field: "rejected"

alignment:
  method: "dpo"                          # dpo or gdpo
  beta: 0.1
  learning_rate: 1e-5
  batch_size: 4

device: "cuda"
```

### Options

- `--config PATH` - Alignment config file (required)
- `--resume PATH` - Resume from checkpoint
- `--device DEVICE` - Override device (cuda/cpu)
- `--dry-run` - Show config without training
- `--verbose, -v` - Detailed output
```

## Testing

### Unit Tests (Fast)

Run mocked unit tests:
```bash
pytest
```

### Integration Tests (Real E2E)

Run real end-to-end tests with actual model downloads and training:
```bash
export HF_TOKEN="your_token_here"
pytest -m integration -v
```

**Note:** Integration tests:
- Require network connection (~60MB download)
- Use distilgpt2 model (fast download)
- Take ~2-3 minutes to complete
- Test real training (1 step each)
- Verify complete workflow: download → fine-tune → DPO align

Skip integration tests and only run unit tests:
```bash
pytest  # Integration tests skipped by default
```
