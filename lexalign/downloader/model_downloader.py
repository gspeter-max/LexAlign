# lexalign/downloader/model_downloader.py
"""Model downloader — downloads from Hugging Face model repositories."""

from lexalign.downloader.base_downloader import BaseDownloader, DownloadError

__all__ = ["ModelDownloader", "DownloadError"]


class ModelDownloader(BaseDownloader):
    """Download models from Hugging Face model repositories."""

    REPO_TYPE = "model"
