# lexalign/downloader/base_downloader.py
"""Abstract base downloader for Hugging Face repositories."""

from pathlib import Path
from typing import Dict, Any, List
from fnmatch import fnmatch

from huggingface_hub import list_repo_files, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from lexalign.downloader.auth import AuthManager


class DownloadError(Exception):
    """Raised when a download fails."""


class BaseDownloader:
    """
    Base class for ModelDownloader and DatasetDownloader.

    Subclasses must define:
        REPO_TYPE: str  — Hugging Face repo type ("model" or "dataset")
    """

    REPO_TYPE: str = ""  # overridden by subclasses

    def __init__(self, auth: AuthManager) -> None:
        """
        Initialize downloader.

        Args:
            auth: Authentication manager instance
        """
        self.auth = auth

    def list_files(self, repo_id: str) -> List[str]:
        """
        List all files in a repository.

        Args:
            repo_id: Repository ID (e.g., "org/repo-name")

        Returns:
            List of file paths in the repository

        Raises:
            DownloadError: If listing files fails
        """
        try:
            files = list(
                list_repo_files(
                    repo_id,
                    repo_type=self.REPO_TYPE,
                    token=self.auth.token,
                )
            )
            return files
        except HfHubHTTPError as e:
            raise DownloadError(f"Failed to list files in {repo_id}: {e}")
        except Exception as e:
            raise DownloadError(f"Error listing files: {e}")

    def _filter_by_patterns(
        self, files: List[str], patterns: List[str]
    ) -> List[str]:
        """
        Filter files by glob patterns.

        Args:
            files: List of file paths
            patterns: List of glob patterns

        Returns:
            Sorted list of files matching any pattern
        """
        matched = set()
        for file in files:
            for pattern in patterns:
                if fnmatch(file, pattern):
                    matched.add(file)
                    break
        return sorted(matched)

    def download_file(
        self,
        repo_id: str,
        filename: str,
        output_dir: str,
        local_dir: str = None,
    ) -> str:
        """
        Download a single file from a repository.

        Args:
            repo_id: Repository ID
            filename: File path in repository
            output_dir: Directory to save file
            local_dir: Local directory for HF hub cache (defaults to output_dir)

        Returns:
            Path to downloaded file

        Raises:
            DownloadError: If download fails
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=self.REPO_TYPE,
                token=self.auth.token,
                local_dir=local_dir or output_dir,
                local_dir_use_symlinks=False,
            )
            return path
        except HfHubHTTPError as e:
            raise DownloadError(f"Failed to download {filename}: {e}")
        except Exception as e:
            raise DownloadError(f"Error downloading {filename}: {e}")

    def download_repo(
        self,
        repo_id: str,
        file_patterns: List[str],
        output_dir: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Download files from a repository matching glob patterns.

        Args:
            repo_id: Repository ID
            file_patterns: List of glob patterns to match files
            output_dir: Directory to save files
            dry_run: If True, list files without downloading

        Returns:
            Dictionary with keys: repo, files, downloaded, output_dir

        Raises:
            DownloadError: If download fails
        """
        all_files = self.list_files(repo_id)
        files_to_download = self._filter_by_patterns(all_files, file_patterns)

        base_result: Dict[str, Any] = {
            "repo": repo_id,
            "files": files_to_download,
            "downloaded": 0,
            "output_dir": output_dir,
        }

        if not files_to_download or dry_run:
            return base_result

        downloaded = 0
        for filename in files_to_download:
            try:
                self.download_file(repo_id, filename, output_dir)
                downloaded += 1
            except DownloadError as e:
                raise DownloadError(
                    f"Download failed for {repo_id}/{filename}: {e}"
                )

        base_result["downloaded"] = downloaded
        return base_result
