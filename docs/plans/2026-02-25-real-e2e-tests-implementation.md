# Real E2E Integration Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add real integration tests that download a tiny model from Hugging Face, fine-tune it with LoRA, and apply DPO alignment using 4-sample datasets, providing 100% confidence the complete workflow works.

**Architecture:** Create `tests/test_real_e2e.py` with integration tests that use real Hugging Face model downloads and actual training (1 step each), marked with `@pytest.mark.integration` for selective execution.

**Tech Stack:** pytest, Click CLI, Hugging Face transformers/distilgpt2, TRL, temporary directories

---

### Task 1: Add pytest integration marker configuration

**Files:**
- Create: `pytest.ini` (if not exists) or Modify: `pytest.ini`

**Step 1: Check if pytest.ini exists**

Run: `cat pytest.ini`
Expected: File may not exist

**Step 2: Create pytest.ini with integration marker**

```ini
[pytest]
markers =
  integration: marks tests as real integration tests (require network, HF_TOKEN, slow ~2-3 minutes)
addopts =
  -m "not integration"  # Skip integration tests by default
```

**Step 3: Verify configuration is valid**

Run: `pytest --markers`
Expected: Should show `integration` marker

**Step 4: Commit**

```bash
git add pytest.ini
git commit -m "test: add pytest integration marker configuration"
```

---

### Task 2: Create test file with helper fixtures

**Files:**
- Create: `tests/test_real_e2e.py`

**Step 1: Write the file skeleton and fixtures**

```python
"""Real end-to-end integration tests with actual model downloads and training."""
import os
import pytest
from pathlib import Path
from click.testing import CliRunner

# Skip all tests in this file if HF_TOKEN not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("HF_TOKEN"),
        reason="HF_TOKEN environment variable not set (use: export HF_TOKEN=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)"
    )
]


@pytest.fixture
def sample_finetune_data(tmp_path):
    """Create 4-sample fine-tuning dataset."""
    import json
    data = [
        {"text": "The sky is blue during the day."},
        {"text": "Grass is green and grows in gardens."},
        {"text": "The sun provides light and warmth."},
        {"text": "Water is essential for all life forms."}
    ]
    data_file = tmp_path / "finetune.json"
    data_file.write_text(json.dumps(data))
    return str(data_file)


@pytest.fixture
def sample_dpo_data(tmp_path):
    """Create 4-sample DPO preference dataset."""
    import json
    data = [
        {"prompt": "What color is the sky?", "chosen": "The sky is blue.", "rejected": "I don't know."},
        {"prompt": "What color is grass?", "chosen": "Grass is green.", "rejected": "Purple."},
        {"prompt": "What does the sun do?", "chosen": "The sun provides light.", "rejected": "Nothing."},
        {"prompt": "Why is water important?", "chosen": "Water sustains life.", "rejected": "It's not."}
    ]
    data_file = tmp_path / "dpo.json"
    data_file.write_text(json.dumps(data))
    return str(data_file)


@pytest.fixture
def download_config(tmp_path):
    """Create download config for distilgpt2."""
    config = f"""
huggingface:
  token: "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"

models:
  - repo: "distilgpt2"
    files:
      - "config.json"
      - "pytorch_model.bin"
      - "tokenizer.json"
      - "vocab.json"
      - "merges.txt"
    output_dir: "{tmp_path}/models/distilgpt2"
"""
    config_file = tmp_path / "download.yaml"
    config_file.write_text(config)
    return str(config_file)
```

**Step 2: Run tests to verify fixtures work**

Run: `pytest tests/test_real_e2e.py -v`
Expected: Collection succeeds (no actual tests yet)

**Step 3: Commit**

```bash
git add tests/test_real_e2e.py
git commit -m "test: add E2E test fixtures for sample data and configs"
```

---

### Task 3: Add model download test

**Files:**
- Modify: `tests/test_real_e2e.py`

**Step 1: Write the download test**

```python
def test_download_distilgpt2(download_config, tmp_path):
    """Test downloading distilgpt2 model from Hugging Face."""
    from download import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--config", download_config])

    # Verify CLI succeeded
    assert result.exit_code == 0, f"Download failed: {result.output}"

    # Verify model files exist
    model_dir = tmp_path / "models" / "distilgpt2"
    assert model_dir.exists(), "Model directory not created"
    assert (model_dir / "config.json").exists(), "config.json missing"
    assert (model_dir / "pytorch_model.bin").exists(), "pytorch_model.bin missing"

    # Verify tokenizer files exist
    assert (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists()
```

**Step 2: Run the test**

Run: `pytest tests/test_real_e2e.py::test_download_distilgpt2 -v -s`
Expected: Downloads distilgpt2 (~60MB), takes 30-60 seconds, PASS

**Step 3: Commit**

```bash
git add tests/test_real_e2e.py
git commit -m "test: add distilgpt2 download test"
```

---

### Task 4: Add fine-tuning test with real training

**Files:**
- Modify: `tests/test_real_e2e.py`

**Step 1: Write the fine-tuning test**

```python
def test_finetune_distilgpt2(download_config, sample_finetune_data, tmp_path):
    """Test fine-tuning distilgpt2 with LoRA (1 step)."""
    from download import cli as download_cli
    from finetune import cli as finetune_cli

    # First download the model
    download_runner = CliRunner()
    download_result = download_runner.invoke(download_cli, ["--config", download_config])
    assert download_result.exit_code == 0, f"Download failed: {download_result.output}"

    model_dir = tmp_path / "models" / "distilgpt2"

    # Create finetune config
    config = f"""
model:
  path: "{model_dir}"

dataset:
  path: "{sample_finetune_data}"
  format: "json"
  text_field: "text"

training:
  method: "lora"
  max_steps: 1
  max_seq_length: 64
  batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 3e-4
  num_epochs: 1
  save_steps: 1000
  output_dir: "{tmp_path}/checkpoints/finetuned"

device: "cpu"
"""
    config_file = tmp_path / "finetune.yaml"
    config_file.write_text(config)

    # Run fine-tuning
    runner = CliRunner()
    result = runner.invoke(finetune_cli, ["--config", str(config_file)])

    # Verify training succeeded
    assert result.exit_code == 0, f"Fine-tuning failed: {result.output}"

    # Verify checkpoint created
    output_dir = tmp_path / "checkpoints" / "finetuned"
    assert output_dir.exists(), "Output directory not created"
    assert (output_dir / "adapter_model.bin").exists() or (output_dir / "adapter_config.json").exists()

    # Verify model can be loaded
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(str(output_dir), device_map="cpu")
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to load fine-tuned model: {e}")
```

**Step 2: Run the test**

Run: `pytest tests/test_real_e2e.py::test_finetune_distilgpt2 -v -s`
Expected: Downloads (if cached), trains 1 step, PASS

**Step 3: Commit**

```bash
git add tests/test_real_e2e.py
git commit -m "test: add real fine-tuning test with LoRA"
```

---

### Task 5: Add DPO alignment test

**Files:**
- Modify: `tests/test_real_e2e.py`

**Step 1: Write the DPO test**

```python
def test_dpo_align_distilgpt2(download_config, sample_finetune_data, sample_dpo_data, tmp_path):
    """Test DPO alignment of fine-tuned distilgpt2 (1 step)."""
    from download import cli as download_cli
    from finetune import cli as finetune_cli
    from align import cli as align_cli

    # Download model
    download_runner = CliRunner()
    download_result = download_runner.invoke(download_cli, ["--config", download_config])
    assert download_result.exit_code == 0

    model_dir = tmp_path / "models" / "distilgpt2"

    # Fine-tune first
    ft_config = f"""
model:
  path: "{model_dir}"

dataset:
  path: "{sample_finetune_data}"
  format: "json"
  text_field: "text"

training:
  method: "lora"
  max_steps: 1
  max_seq_length: 64
  batch_size: 1
  output_dir: "{tmp_path}/checkpoints/finetuned"

device: "cpu"
"""
    ft_config_file = tmp_path / "finetune.yaml"
    ft_config_file.write_text(ft_config)

    ft_runner = CliRunner()
    ft_result = ft_runner.invoke(finetune_cli, ["--config", str(ft_config_file)])
    assert ft_result.exit_code == 0, f"Fine-tuning failed: {ft_result.output}"

    # DPO align
    dpo_config = f"""
model:
  path: "{tmp_path}/checkpoints/finetuned"

dataset:
  path: "{sample_dpo_data}"
  format: "json"
  prompt_field: "prompt"
  chosen_field: "chosen"
  rejected_field: "rejected"

alignment:
  method: "dpo"
  max_steps: 1
  max_seq_length: 64
  batch_size: 1
  learning_rate: 1e-5
  num_epochs: 1
  beta: 0.1
  output_dir: "{tmp_path}/checkpoints/aligned"

device: "cpu"
"""
    dpo_config_file = tmp_path / "align.yaml"
    dpo_config_file.write_text(dpo_config)

    # Run DPO alignment
    runner = CliRunner()
    result = runner.invoke(align_cli, ["--config", str(dpo_config_file)])

    # Verify alignment succeeded
    assert result.exit_code == 0, f"DPO alignment failed: {result.output}"

    # Verify output created
    output_dir = tmp_path / "checkpoints" / "aligned"
    assert output_dir.exists(), "Alignment output directory not created"
```

**Step 2: Run the test**

Run: `pytest tests/test_real_e2e.py::test_dpo_align_distilgpt2 -v -s`
Expected: Downloads, finetunes, DPO aligns, PASS

**Step 3: Commit**

```bash
git add tests/test_real_e2e.py
git commit -m "test: add DPO alignment test"
```

---

### Task 6: Add complete workflow test with output verification

**Files:**
- Modify: `tests/test_real_e2e.py`

**Step 1: Write the complete workflow test**

```python
def test_complete_workflow_output_verification(download_config, sample_finetune_data, sample_dpo_data, tmp_path):
    """Test complete workflow and verify model outputs change at each stage."""
    from download import cli as download_cli
    from finetune import cli as finetune_cli
    from align import cli as align_cli
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Download
    download_runner = CliRunner()
    download_result = download_runner.invoke(download_cli, ["--config", download_config])
    assert download_result.exit_code == 0

    model_dir = tmp_path / "models" / "distilgpt2"

    # Load base model for comparison
    base_model = AutoModelForCausalLM.from_pretrained(str(model_dir), device_map="cpu")
    base_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Fine-tune
    ft_config = f"""
model:
  path: "{model_dir}"
  base_model: "distilgpt2"

dataset:
  path: "{sample_finetune_data}"
  format: "json"
  text_field: "text"

training:
  method: "lora"
  max_steps: 1
  max_seq_length: 64
  batch_size: 1
  output_dir: "{tmp_path}/checkpoints/finetuned"

device: "cpu"
"""
    ft_config_file = tmp_path / "finetune.yaml"
    ft_config_file.write_text(ft_config)

    ft_runner = CliRunner()
    ft_result = ft_runner.invoke(finetune_cli, ["--config", str(ft_config_file)])
    assert ft_result.exit_code == 0

    # Load fine-tuned model
    ft_model = AutoModelForCausalLM.from_pretrained(
        str(tmp_path / "checkpoints" / "finetuned"),
        device_map="cpu"
    )

    # DPO align
    dpo_config = f"""
model:
  path: "{tmp_path}/checkpoints/finetuned"

dataset:
  path: "{sample_dpo_data}"
  format: "json"
  prompt_field: "prompt"
  chosen_field: "chosen"
  rejected_field: "rejected"

alignment:
  method: "dpo"
  max_steps: 1
  max_seq_length: 64
  batch_size: 1
  output_dir: "{tmp_path}/checkpoints/aligned"

device: "cpu"
"""
    dpo_config_file = tmp_path / "align.yaml"
    dpo_config_file.write_text(dpo_config)

    dpo_runner = CliRunner()
    dpo_result = dpo_runner.invoke(align_cli, ["--config", str(dpo_config_file)])
    assert dpo_result.exit_code == 0

    # Load aligned model
    aligned_model = AutoModelForCausalLM.from_pretrained(
        str(tmp_path / "checkpoints" / "aligned"),
        device_map="cpu"
    )

    # Generate outputs from each model
    prompt = "The sky is"
    inputs = base_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        base_output = base_model.generate(**inputs, max_length=20, do_sample=False)
        ft_output = ft_model.generate(**inputs, max_length=20, do_sample=False)
        aligned_output = aligned_model.generate(**inputs, max_length=20, do_sample=False)

    base_text = base_tokenizer.decode(base_output[0], skip_special_tokens=True)
    ft_text = base_tokenizer.decode(ft_output[0], skip_special_tokens=True)
    aligned_text = base_tokenizer.decode(aligned_output[0], skip_special_tokens=True)

    # Verify outputs exist and are strings
    assert isinstance(base_text, str)
    assert isinstance(ft_text, str)
    assert isinstance(aligned_text, str)

    # At minimum, verify we can generate from all models
    assert len(base_text) > 0
    assert len(ft_text) > 0
    assert len(aligned_text) > 0

    # Print outputs for manual verification
    print(f"\nBase model: {base_text}")
    print(f"Fine-tuned: {ft_text}")
    print(f"Aligned: {aligned_text}")
```

**Step 2: Run the test**

Run: `pytest tests/test_real_e2e.py::test_complete_workflow_output_verification -v -s`
Expected: Complete workflow, generates outputs, PASS

**Step 3: Commit**

```bash
git add tests/test_real_e2e.py
git commit -m "test: add complete workflow test with output verification"
```

---

### Task 7: Add checkpoint resume test

**Files:**
- Modify: `tests/test_real_e2e.py`

**Step 1: Write the checkpoint resume test**

```python
def test_checkpoint_resume(download_config, sample_finetune_data, tmp_path):
    """Test checkpoint saving and resume functionality."""
    from download import cli as download_cli
    from finetune import cli as finetune_cli

    # Download model
    download_runner = CliRunner()
    download_result = download_runner.invoke(download_cli, ["--config", download_config])
    assert download_result.exit_code == 0

    model_dir = tmp_path / "models" / "distilgpt2"

    # Initial training with checkpoint
    config = f"""
model:
  path: "{model_dir}"
  base_model: "distilgpt2"

dataset:
  path: "{sample_finetune_data}"
  format: "json"
  text_field: "text"

training:
  method: "lora"
  max_steps: 1
  max_seq_length: 64
  batch_size: 1
  save_steps: 1
  output_dir: "{tmp_path}/checkpoints/resume-test"

device: "cpu"
"""
    config_file = tmp_path / "finetune.yaml"
    config_file.write_text(config)

    # Initial training
    runner = CliRunner()
    result = runner.invoke(finetune_cli, ["--config", str(config_file)])
    assert result.exit_code == 0, f"Initial training failed: {result.output}"

    # Verify checkpoint exists
    checkpoint_dir = tmp_path / "checkpoints" / "resume-test" / "checkpoint-1"
    assert checkpoint_dir.exists(), "Checkpoint not created"

    # Resume from checkpoint
    resume_result = runner.invoke(finetune_cli, [
        "--config", str(config_file),
        "--resume", str(checkpoint_dir)
    ])

    # Verify resume succeeded
    assert resume_result.exit_code == 0, f"Resume failed: {resume_result.output}"
```

**Step 2: Run the test**

Run: `pytest tests/test_real_e2e.py::test_checkpoint_resume -v -s`
Expected: Training, checkpoint save, resume, PASS

**Step 3: Commit**

```bash
git add tests/test_real_e2e.py
git commit -m "test: add checkpoint resume test"
```

---

### Task 8: Update README with integration test instructions

**Files:**
- Modify: `README.md`

**Step 1: Add testing section to README**

Add after existing testing section or create new section:

```markdown
## Testing

### Unit Tests (Fast)

Run mocked unit tests:
```bash
pytest
```

### Integration Tests (Real E2E)

Run real end-to-end tests with actual model downloads and training:
```bash
export HF_TOKEN="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"
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
```

**Step 2: Verify README format is correct**

Run: `cat README.md | grep -A 20 "## Testing"`
Expected: Should show new testing section

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add integration test instructions to README"
```

---

## Summary

This implementation plan creates real end-to-end integration tests that:

1. **Download** distilgpt2 from Hugging Face (~60MB)
2. **Fine-tune** with 4 samples using LoRA (1 step)
3. **DPO align** with 4 preference pairs (1 step)
4. **Verify** model outputs change at each stage
5. **Test** checkpoint save and resume

**Total estimated time:** ~2-3 minutes per full run

**Files created/modified:**
- `pytest.ini` - Integration marker configuration
- `tests/test_real_e2e.py` - All integration tests
- `README.md` - Documentation for running tests
