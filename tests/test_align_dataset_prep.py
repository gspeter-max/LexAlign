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
