# tests/test_align_dataset_prep.py
import pytest
from pathlib import Path
from lexalign.aligner.dataset_prep import PreferenceDataset, DatasetError

@pytest.fixture
def sample_preference_json(tmp_path):
    """Create a sample preference dataset JSON file."""
    data = [
        {"prompt": "What is AI?", "chosen": "AI is artificial intelligence.", "rejected": "I don't know."},
        {"prompt": "Explain gravity.", "chosen": "Gravity is a force.", "rejected": "No idea."},
    ]
    file_path = tmp_path / "preferences.json"
    import json
    file_path.write_text(json.dumps(data))
    return str(file_path)

def test_load_and_validate_preference_dataset(sample_preference_json):
    config = {
        "path": sample_preference_json,
        "format": "json",
        "prompt_field": "prompt",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
        "train_split": "train",
    }

    prep = PreferenceDataset()
    dataset = prep.load_and_validate(config)

    assert len(dataset) == 2
    assert "prompt" in dataset.column_names
    assert "chosen" in dataset.column_names
    assert "rejected" in dataset.column_names


@pytest.fixture
def incomplete_preference_json(tmp_path):
    """Create dataset missing rejected field."""
    data = [
        {"prompt": "What is AI?", "chosen": "AI is artificial intelligence."},
    ]
    file_path = tmp_path / "incomplete.json"
    import json
    file_path.write_text(json.dumps(data))
    return str(file_path)

def test_missing_rejected_field_raises_error(incomplete_preference_json):
    config = {
        "path": incomplete_preference_json,
        "format": "json",
        "prompt_field": "prompt",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
        "train_split": "train",
    }

    prep = PreferenceDataset()
    with pytest.raises(DatasetError, match="missing required field.*rejected"):
        prep.load_and_validate(config)


def test_unsupported_format_raises_error(tmp_path):
    """Test that unsupported file format raises error."""
    file_path = tmp_path / "data.txt"
    file_path.write_text("some text")

    config = {
        "path": str(file_path),
        "format": "txt",  # Unsupported
        "prompt_field": "prompt",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
    }

    prep = PreferenceDataset()
    with pytest.raises(DatasetError, match="Unsupported format"):
        prep.load_and_validate(config)
