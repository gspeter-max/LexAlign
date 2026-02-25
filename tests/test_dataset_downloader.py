import pytest
from lexalign.downloader.dataset_downloader import DatasetDownloader
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
