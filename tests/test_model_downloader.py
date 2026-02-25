import pytest
from pathlib import Path
from lexalign.downloader.model_downloader import ModelDownloader, DownloadError
from lexalign.downloader.auth import AuthManager

# NOTE: mocks target `base_downloader` because that's where
# list_repo_files / hf_hub_download are imported after the BaseDownloader refactor.
_BASE = "lexalign.downloader.base_downloader"


def test_list_files_from_repo(mocker):
    auth = AuthManager("test_token")
    downloader = ModelDownloader(auth)

    mock_list_files = mocker.patch(
        f"{_BASE}.list_repo_files",
        return_value=["config.json", "pytorch_model.bin", "tokenizer.json"]
    )

    files = downloader.list_files("org/model")
    assert len(files) == 3
    assert "config.json" in files
    mock_list_files.assert_called_once_with("org/model", repo_type="model", token="test_token")


def test_filter_files_by_patterns():
    auth = AuthManager("test_token")
    downloader = ModelDownloader(auth)

    all_files = [
        "config.json",
        "pytorch_model-00001.bin",
        "pytorch_model-00002.bin",
        "tokenizer.json",
        "training_args.bin"
    ]

    patterns = ["*.json", "pytorch_model*.bin"]
    filtered = downloader._filter_by_patterns(all_files, patterns)

    assert "config.json" in filtered
    assert "pytorch_model-00001.bin" in filtered
    assert "pytorch_model-00002.bin" in filtered
    assert "tokenizer.json" in filtered
    assert "training_args.bin" not in filtered


def test_download_file_success(mocker, tmp_path):
    auth = AuthManager("test_token")
    downloader = ModelDownloader(auth)

    mock_download = mocker.patch(f"{_BASE}.hf_hub_download")

    downloader.download_file(
        repo_id="org/model",
        filename="config.json",
        output_dir=str(tmp_path),
        local_dir=str(tmp_path)
    )

    mock_download.assert_called_once()


def test_download_file_failure(mocker, tmp_path):
    auth = AuthManager("test_token")
    downloader = ModelDownloader(auth)

    mocker.patch(
        f"{_BASE}.hf_hub_download",
        side_effect=Exception("Network error")
    )

    with pytest.raises(DownloadError, match="Error downloading"):
        downloader.download_file(
            repo_id="org/model",
            filename="config.json",
            output_dir=str(tmp_path),
            local_dir=str(tmp_path)
        )


def test_download_repo_dry_run(mocker):
    auth = AuthManager("test_token")
    downloader = ModelDownloader(auth)

    mocker.patch(
        f"{_BASE}.list_repo_files",
        return_value=["config.json", "model.bin"]
    )

    result = downloader.download_repo(
        repo_id="org/model",
        file_patterns=["*.json"],
        output_dir="./models",
        dry_run=True
    )

    assert len(result["files"]) == 1
    assert result["files"][0] == "config.json"
    assert result["downloaded"] == 0
