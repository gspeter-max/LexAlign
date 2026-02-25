# LexAlign Download Tool - Design Document

**Date:** 2026-02-25
**Status:** Approved

## Overview

A Python-based CLI tool that downloads models and datasets from Hugging Face using a declarative YAML configuration. The tool uses `huggingface_hub` for direct file downloads with authentication support, targeting local development and experimentation workflows.

## Architecture

### Components

- `download.py` - Main CLI entry point with argument parsing
- `lexalign/downloader/` - Core download logic module
  - `model_downloader.py` - Model repo file downloads
  - `dataset_downloader.py` - Dataset repo file downloads
  - `auth.py` - Token validation and management
- `lexalign/config/` - Configuration handling
  - `parser.py` - YAML config parser and validation

### Data Flow

```
CLI args → Config parser → Token validation → Parallel downloads → Local storage
```

### Error Handling

- **Invalid config** → Clear validation error before any downloads
- **Authentication failure** → Immediate halt with credential error
- **Download failure** → Fail fast, clean partial files, report specific error
- **Network issues** → Timeout with clear error message

## Configuration Schema

### YAML Structure

```yaml
# config/downloads.yaml
huggingface:
  token: "${HF_TOKEN}"  # Environment variable expansion

models:
  - repo: "meta-llama/Llama-2-7b-hf"
    files:
      - "pytorch_model*.bin"
      - "config.json"
      - "tokenizer.json"
    output_dir: "./models/llama2-7b"

datasets:
  - repo: "username/my-dataset"
    files:
      - "data/train/*.json"
      - "data/valid/*.json"
    output_dir: "./data/my-dataset"
```

### Key Features

- Environment variable expansion for tokens
- Glob patterns for file matching
- Per-repo output directories
- Fail-fast validation on startup

## CLI Interface

### Command Structure

```bash
# Using config file
python download.py --config config/downloads.yaml

# Override token
python download.py --config config/downloads.yaml --token $HF_TOKEN

# Dry run (show what would download)
python download.py --config config/downloads.yaml --dry-run

# Download only models or only datasets
python download.py --config config/downloads.yaml --models-only
python download.py --config config/downloads.yaml --datasets-only

# Verbose output
python download.py --config config/downloads.yaml --verbose
```

### Options

- `--config PATH` - YAML config file (required)
- `--token TOKEN` - Override HF_TOKEN from config
- `--dry-run` - Show files without downloading
- `--models-only` - Skip datasets
- `--datasets-only` - Skip models
- `--verbose` - Detailed progress output
- `--workers N` - Parallel download workers (default: 4)

## Dependencies

### Required Packages

```python
# requirements.txt
huggingface-hub>=0.20.0  # Hugging Face API
pyyaml>=6.0              # YAML parsing
rich>=13.0.0             # Pretty CLI output
requests>=2.31.0         # HTTP (transitive dep)
```

**Python Version:** 3.9+

### Rationale

- `huggingface_hub` - Official HF library, handles auth and downloads
- `pyyaml` - Standard YAML parsing
- `rich` - Professional CLI with progress bars and error formatting

## File Structure

```
LexAlign/
├── README.md
├── requirements.txt
├── download.py                 # CLI entry point
├── lexalign/
│   ├── __init__.py
│   ├── downloader/
│   │   ├── __init__.py
│   │   ├── model_downloader.py
│   │   ├── dataset_downloader.py
│   │   └── auth.py
│   └── config/
│       ├── __init__.py
│       └── parser.py
├── config/
│   └── downloads.yaml.example  # Example config
└── docs/
    └── plans/
        └── 2026-02-25-download-tool-design.md
```

## Implementation Approach

**Selected:** Python script + YAML config

This approach provides:
- Declarative, reusable configurations
- Clean CLI interface
- Separation of concerns (config vs execution)
- Easy to extend with new features

## Success Criteria

- [x] Download models from Hugging Face using auth token
- [x] Download datasets from Hugging Face repos
- [x] Support glob patterns for selective file downloads
- [x] YAML-based configuration
- [x] Environment variable expansion for tokens
- [x] Fail-fast error handling
- [x] Clear CLI output with progress indication
