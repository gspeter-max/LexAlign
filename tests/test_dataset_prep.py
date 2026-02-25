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
