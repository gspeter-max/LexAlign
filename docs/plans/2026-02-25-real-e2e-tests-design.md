# Real End-to-End Integration Tests - Design Document

**Date:** 2026-02-25
**Status:** Approved

## Overview

Add real integration tests that download a tiny model from Hugging Face, fine-tune it with LoRA, and apply DPO alignment using minimal datasets (4 samples each). These tests verify actual functionality rather than mocked code paths, providing 100% confidence that the complete workflow works correctly.

## Architecture

### Test File

`tests/test_real_e2e.py` - Integration tests marked with `@pytest.mark.integration`

### Workflow

```
1. Download distilgpt2 (60MB) to temp directory
2. Create 4-sample fine-tuning dataset
3. Run 1 training step with LoRA
4. Create 4-sample preference dataset
5. Run 1 DPO training step
6. Verify model outputs changed
7. Cleanup
```

### Components

- Reuses existing `download.py`, `finetune.py`, `align.py` CLIs
- Uses `pytest.mark.integration` marker for selective running
- Temp directories with auto-cleanup via `tmp_path` fixture
- Real Hugging Face model (distilgpt2 - 60MB, fast download)

## Test Cases

### Test 1: Complete Workflow

```python
@pytest.mark.integration
def test_real_download_finetune_dpo_workflow(tmp_path):
    """
    Complete real workflow:
    1. Download distilgpt2 model
    2. Fine-tune with 4 samples (1 step)
    3. DPO align with 4 preference pairs (1 step)
    4. Verify outputs differ from base model
    """
```

**Test Data (4 samples):**

Fine-tuning dataset:
```json
[
  {"text": "The sky is blue during the day."},
  {"text": "Grass is green and grows in gardens."},
  {"text": "The sun provides light and warmth."},
  {"text": "Water is essential for all life forms."}
]
```

DPO preference dataset:
```json
[
  {"prompt": "What color is the sky?", "chosen": "The sky is blue.", "rejected": "I don't know."},
  {"prompt": "What color is grass?", "chosen": "Grass is green.", "rejected": "Purple."},
  {"prompt": "What does the sun do?", "chosen": "The sun provides light.", "rejected": "Nothing."},
  {"prompt": "Why is water important?", "chosen": "Water sustains life.", "rejected": "It's not."}
]
```

### Test 2: Checkpoint Resume

```python
@pytest.mark.integration
def test_real_checkpoint_resume(tmp_path):
    """
    Test checkpoint saving and loading:
    1. Fine-tune for 1 step, save checkpoint
    2. Resume from checkpoint
    3. Verify training continues without error
    """
```

## Configuration

YAML configs generated programmatically at runtime:

### Download Config

```yaml
huggingface:
  token: "${HF_TOKEN}"

models:
  - repo: "distilgpt2"
    files:
      - "config.json"
      - "pytorch_model.bin"
      - "tokenizer.json"
      - "vocab.json"
      - "merges.txt"
    output_dir: "{tmp_path}/models/distilgpt2"
```

### Finetune Config

```yaml
model:
  path: "{tmp_path}/models/distilgpt2"

dataset:
  path: "{tmp_path}/data/finetune.json"
  format: "json"
  text_field: "text"

training:
  method: "lora"
  max_steps: 1
  max_seq_length: 64
  batch_size: 1
  gradient_accumulation_steps: 1
  output_dir: "{tmp_path}/checkpoints/finetuned"

device: "cpu"
```

### Align Config

```yaml
model:
  path: "{tmp_path}/checkpoints/finetuned"

dataset:
  path: "{tmp_path}/data/dpo.json"
  format: "json"
  prompt_field: "prompt"
  chosen_field: "chosen"
  rejected_field: "rejected"

alignment:
  method: "dpo"
  max_steps: 1
  batch_size: 1
  output_dir: "{tmp_path}/checkpoints/aligned"

device: "cpu"
```

## Verification Strategy

### 1. File Existence

```python
assert (model_dir / "config.json").exists()
assert (model_dir / "pytorch_model.bin").exists()
```

### 2. Checkpoint Created

```python
checkpoint_dir = output_dir / "checkpoint-1"
assert checkpoint_dir.exists()
assert (checkpoint_dir / "adapter_model.bin").exists()
```

### 3. Model Outputs Changed

```python
# Generate sample from base model
base_output = base_model.generate("The color of the sky is")

# Generate from fine-tuned model
ft_output = ft_model.generate("The color of the sky is")

# Generate from aligned model
aligned_output = aligned_model.generate("The color of the sky is")

# Verify outputs differ (training actually happened)
assert base_output != ft_output
assert ft_output != aligned_output
```

### 4. Resume Works

```python
# Resume from checkpoint
result = runner.invoke(cli, ["--config", config, "--resume", str(checkpoint_dir)])
assert result.exit_code == 0
```

## Error Handling

### Network Issues

```python
# Skip test if HF_TOKEN not set
if not os.environ.get("HF_TOKEN"):
    pytest.skip("HF_TOKEN not set")

# Skip if huggingface_hub unreachable
try:
    hf_login(token)
except Exception:
    pytest.skip("Cannot reach Hugging Face Hub")

# Timeout after 5 minutes
@pytest.mark.timeout(300)
```

### Model Loading

```python
# Clear error if distilgpt2 removed from HF
try:
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
except OSError as e:
    pytest.skip(f"distilgpt2 not available: {e}")
```

### Memory/Resource

```python
# Force CPU device (no CUDA required)
config["device"] = "cpu"

# Minimal batch sizes
config["training"]["batch_size"] = 1

# Max 1 training step
config["training"]["max_steps"] = 1
```

### Test Failures

```python
# Preserve temp directories for inspection
def test_real_workflow(tmp_path):
    try:
        # ... test code ...
    except Exception as e:
        print(f"Test artifacts preserved at: {tmp_path}")
        raise
```

## Execution & CI/CD

### Running Tests

```bash
# Run only fast mocked tests (default)
pytest

# Run integration tests too
pytest -m integration

# Run ONLY integration tests
pytest -m integration -m "not not_integration"

# Skip in CI with environment variable
SKIP_INTEGRATION=true pytest
```

### CI/CD Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - pytest  # All mocked tests (~10 seconds)

  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'
    steps:
      - pytest -m integration  # Real tests (~2-3 minutes)
```

### pytest.ini

```ini
[pytest]
markers =
  integration: marks tests as real integration tests (may require network, slow)
addopts =
  -m "not integration"  # Skip integration by default
```

### Duration Estimates

- Mocked tests: ~10 seconds
- Integration tests: ~2-3 minutes (download 60MB + training)

## Implementation Notes

### Dependencies

- Uses existing test fixtures from `conftest.py`
- No new packages required
- Requires `HF_TOKEN` environment variable

### File Structure

```
tests/
  test_real_e2e.py          # NEW: Real integration tests
  conftest.py               # EXISTING: Shared fixtures
  pytest.ini                # NEW: Markers config (or add to existing)
```

### Test Data Generation

- Created inline in tests (no external files)
- 4 samples minimal but sufficient for verification
- JSON format for compatibility

### Cleanup

```python
# tmp_path fixture auto-removes after test
def test_real_workflow(tmp_path):
    # tmp_path automatically cleaned up

# Explicit cleanup if needed
def test_real_workflow(tmp_path):
    try:
        # ... test code ...
    finally:
        if keep_artifacts:
            shutil.copytree(tmp_path, "/tmp/debug_e2e")
```

## Success Criteria

- [ ] Download distilgpt2 successfully
- [ ] Fine-tune with 4 samples (1 step)
- [ ] DPO align with 4 preference pairs (1 step)
- [ ] Verify model outputs changed after each stage
- [ ] Test checkpoint save and resume
- [ ] Tests pass in ~2-3 minutes
- [ ] Tests can run independently with `pytest -m integration`
- [ ] CI/CD skips integration tests by default

## Rationale

### Why Real E2E Tests?

| What | Mocks | Real E2E |
|------|-------|----------|
| Code paths work | ✅ | ✅ |
| TRL integrates correctly | ❌ | ✅ |
| Model trains and updates | ❌ | ✅ |
| Checkpoint save/load | ❌ | ✅ |
| Real tokenizer interaction | ❌ | ✅ |

**Mocks alone cannot give 100% confidence.** Real E2E tests verify actual functionality.

### Why distilgpt2?

- 82M parameters (tiny)
- ~60MB download (fast)
- Real causal LM from Hugging Face
- Well-maintained and stable

### Why 4 Samples?

- Minimal but sufficient for verification
- Fast training (1 step completes in seconds)
- Proves the pipeline works
- Keeps test duration short
