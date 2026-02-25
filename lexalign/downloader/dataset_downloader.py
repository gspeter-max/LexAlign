from typing import List, Dict, Any
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import list_repo_files, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from lexalign.downloader.auth import AuthManager
from lexalign.downloader.model_downloader import DownloadError


class DatasetDownloader:
    """Download datasets from Hugging Face."""

    def __init__(self, auth: AuthManager):
        """
        Initialize dataset downloader.

        Args:
            auth: Authentication manager instance
        """
        self.auth = auth

    def list_files(self, repo_id: str) -> List[str]:
        """
        List all files in a dataset repository.

        Args:
            repo_id: Dataset repository ID (e.g., "user/dataset-name")

        Returns:
            List of file paths in the repository

        Raises:
            DownloadError: If listing files fails
        """
        try:
            files = list_repo_files(
                repo_id,
                repo_type="dataset",
                token=self.auth.token
            )
            return files
        except HfHubHTTPError as e:
            raise DownloadError(f"Failed to list files in {repo_id}: {e}")
        except Exception as e:
            raise DownloadError(f"Error listing files: {e}")

    def _filter_by_patterns(self, files: List[str], patterns: List[str]) -> List[str]:
        """
        Filter files by glob patterns.

        Args:
            files: List of file paths
            patterns: List of glob patterns

        Returns:
            Filtered list of files matching any pattern
        """
        matched = set()
        for file in files:
            for pattern in patterns:
                if fnmatch(file, pattern):
                    matched.add(file)
                    break
        return sorted(list(matched))

    def download_file(
        self,
        repo_id: str,
        filename: str,
        output_dir: str,
        local_dir: str = None
    ) -> str:
        """
        Download a single file from a repository.

        Args:
            repo_id: Dataset repository ID
            filename: File path in repository
            output_dir: Directory to save file
            local_dir: Local directory for HF hub cache

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
                repo_type="dataset",
                token=self.auth.token,
                local_dir=local_dir or output_dir,
                local_dir_use_symlinks=False
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
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Download files from a dataset repository.

        Args:
            repo_id: Dataset repository ID
            file_patterns: List of glob patterns to match files
            output_dir: Directory to save files
            dry_run: If True, list files without downloading

        Returns:
            Dictionary with download results

        Raises:
            DownloadError: If download fails
        """
        all_files = self.list_files(repo_id)
        files_to_download = self._filter_by_patterns(all_files, file_patterns)

        if not files_to_download:
            return {
                "repo": repo_id,
                "files": [],
                "downloaded": 0,
                "output_dir": output_dir
            }

        if dry_run:
            return {
                "repo": repo_id,
                "files": files_to_download,
                "downloaded": 0,
                "output_dir": output_dir
            }

        downloaded = 0
        for filename in files_to_download:
            try:
                self.download_file(repo_id, filename, output_dir)
                downloaded += 1
            except DownloadError as e:
                raise DownloadError(
                    f"Download failed for {repo_id}/{filename}: {e}"
                )

        return {
            "repo": repo_id,
            "files": files_to_download,
            "downloaded": downloaded,
            "output_dir": output_dir
        }
