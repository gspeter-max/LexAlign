# lexalign/downloader/dataset_downloader.py
"""Dataset downloader — downloads from Hugging Face dataset repositories."""

from lexalign.downloader.base_downloader import BaseDownloader, DownloadError

__all__ = ["DatasetDownloader", "DownloadError"]


class DatasetDownloader(BaseDownloader):
    """Download datasets from Hugging Face dataset repositories."""

    REPO_TYPE = "dataset"
