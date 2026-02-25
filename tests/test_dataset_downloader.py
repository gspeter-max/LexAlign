import pytest
from lexalign.downloader.dataset_downloader import DatasetDownloader, DownloadError
from lexalign.downloader.auth import AuthManager

def test_dataset_downloader_is_same_as_model():
    """Dataset downloader should reuse model downloader logic."""
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    # Should have same interface as ModelDownloader
    assert hasattr(downloader, 'list_files')
    assert hasattr(downloader, 'download_repo')
    assert hasattr(downloader, '_filter_by_patterns')

def test_list_files_from_dataset_repo(mocker):
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    mock_list_files = mocker.patch(
        'lexalign.downloader.dataset_downloader.list_repo_files',
        return_value=["data/train.json", "data/test.json", "README.md"]
    )

    files = downloader.list_files("user/dataset")
    assert len(files) == 3
    mock_list_files.assert_called_once_with("user/dataset", repo_type="dataset", token="test_token")

def test_filter_dataset_files_by_patterns():
    """Test filtering dataset files with glob patterns."""
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    all_files = [
        "data/train.json",
        "data/test.json",
        "data/valid.json",
        "README.md",
        "dataset_infos.json"
    ]

    patterns = ["data/*.json"]
    filtered = downloader._filter_by_patterns(all_files, patterns)

    assert "data/train.json" in filtered
    assert "data/test.json" in filtered
    assert "data/valid.json" in filtered
    assert "README.md" not in filtered
    assert "dataset_infos.json" not in filtered

def test_filter_dataset_files_multiple_patterns():
    """Test filtering with multiple patterns."""
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    all_files = [
        "data/train.json",
        "data/test.csv",
        "metadata.txt",
        "README.md"
    ]

    patterns = ["data/*.json", "data/*.csv", "*.txt"]
    filtered = downloader._filter_by_patterns(all_files, patterns)

    assert "data/train.json" in filtered
    assert "data/test.csv" in filtered
    assert "metadata.txt" in filtered
    assert "README.md" not in filtered

def test_download_dataset_file_success(mocker, tmp_path):
    """Test downloading a single dataset file."""
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    mock_download = mocker.patch('lexalign.downloader.dataset_downloader.hf_hub_download')

    downloader.download_file(
        repo_id="user/dataset",
        filename="data/train.json",
        output_dir=str(tmp_path),
        local_dir=str(tmp_path)
    )

    mock_download.assert_called_once()

def test_download_dataset_file_failure(mocker, tmp_path):
    """Test download failure handling."""
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    mocker.patch(
        'lexalign.downloader.dataset_downloader.hf_hub_download',
        side_effect=Exception("Network error")
    )

    with pytest.raises(DownloadError, match="Error downloading"):
        downloader.download_file(
            repo_id="user/dataset",
            filename="data/train.json",
            output_dir=str(tmp_path),
            local_dir=str(tmp_path)
        )

def test_download_dataset_repo_dry_run(mocker):
    """Test dry run for dataset repository."""
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    mocker.patch(
        'lexalign.downloader.dataset_downloader.list_repo_files',
        return_value=["data/train.json", "data/test.json", "README.md"]
    )

    result = downloader.download_repo(
        repo_id="user/dataset",
        file_patterns=["data/*.json"],
        output_dir="./data/dataset",
        dry_run=True
    )

    assert len(result["files"]) == 2
    assert "data/train.json" in result["files"]
    assert "data/test.json" in result["files"]
    assert result["downloaded"] == 0
    assert result["repo"] == "user/dataset"

def test_download_dataset_repo_no_matches(mocker):
    """Test when no files match patterns."""
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    mocker.patch(
        'lexalign.downloader.dataset_downloader.list_repo_files',
        return_value=["README.md", "dataset_infos.json"]
    )

    result = downloader.download_repo(
        repo_id="user/dataset",
        file_patterns=["data/*.json"],
        output_dir="./data/dataset",
        dry_run=True
    )

    assert len(result["files"]) == 0
    assert result["downloaded"] == 0

def test_download_dataset_repo_with_download(mocker, tmp_path):
    """Test actual download with mocked HF hub."""
    auth = AuthManager("test_token")
    downloader = DatasetDownloader(auth)

    mocker.patch(
        'lexalign.downloader.dataset_downloader.list_repo_files',
        return_value=["data/train.json", "data/test.json"]
    )

    mock_download = mocker.patch('lexalign.downloader.dataset_downloader.hf_hub_download')

    result = downloader.download_repo(
        repo_id="user/dataset",
        file_patterns=["data/*.json"],
        output_dir=str(tmp_path),
        dry_run=False
    )

    assert result["downloaded"] == 2
    assert mock_download.call_count == 2
