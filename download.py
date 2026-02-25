#!/usr/bin/env python3
"""
LexAlign Download Tool - CLI entry point

Download models and datasets from Hugging Face using YAML configuration.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lexalign.config.parser import ConfigParser, ConfigValidationError
from lexalign.downloader.auth import AuthManager, AuthError
from lexalign.downloader.model_downloader import ModelDownloader, DownloadError
from lexalign.downloader.dataset_downloader import DatasetDownloader

console = Console()
stderr_console = Console(stderr=True)


def print_error(message: str) -> None:
    """Print error message in red."""
    stderr_console.print(f"[red]Error:[/red] {message}")


def print_success(message: str) -> None:
    """Print success message in green."""
    console.print(f"[green]Success:[/green] {message}")


def print_info(message: str) -> None:
    """Print info message in blue."""
    console.print(f"[blue]Info:[/blue] {message}")


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML configuration file"
)
@click.option(
    "--token",
    "-t",
    default=None,
    help="Override Hugging Face token from config"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be downloaded without actually downloading"
)
@click.option(
    "--models-only",
    is_flag=True,
    help="Only download models, skip datasets"
)
@click.option(
    "--datasets-only",
    is_flag=True,
    help="Only download datasets, skip models"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output"
)
def cli(
    config: str,
    token: Optional[str],
    dry_run: bool,
    models_only: bool,
    datasets_only: bool,
    verbose: bool
) -> None:
    """
    LexAlign Download Tool - Download HF models/datasets from config.

    Example:
        python download.py --config config/downloads.yaml
    """
    try:
        # Load and parse config
        if verbose:
            print_info(f"Loading config from {config}")

        with open(config, "r") as f:
            yaml_content = f.read()

        parser = ConfigParser()
        parsed_config = parser.parse(yaml_content)

        # Get token
        hf_token = token or parsed_config.get("huggingface", {}).get("token")
        if not hf_token:
            print_error("No Hugging Face token provided")
            sys.exit(1)

        # Initialize auth
        if verbose:
            print_info("Validating authentication...")

        auth = AuthManager(hf_token)
        if not auth.validate_token():
            print_error("Invalid Hugging Face token")
            sys.exit(1)

        print_success("Authentication successful")

        # Download models
        models = parsed_config.get("models", [])
        if models and not datasets_only:
            model_downloader = ModelDownloader(auth)

            for model_config in models:
                repo_id = model_config["repo"]
                file_patterns = model_config["files"]
                output_dir = model_config["output_dir"]

                print_info(f"Processing model: {repo_id}")

                if dry_run:
                    print_info(f"[DRY RUN] Would download from {repo_id}")

                result = model_downloader.download_repo(
                    repo_id=repo_id,
                    file_patterns=file_patterns,
                    output_dir=output_dir,
                    dry_run=dry_run
                )

                if result["files"]:
                    print_info(f"  Files: {len(result['files'])} matched")
                    if dry_run:
                        for f in result["files"][:5]:
                            print_info(f"    - {f}")
                        if len(result["files"]) > 5:
                            print_info(f"    ... and {len(result['files']) - 5} more")
                    else:
                        print_success(f"  Downloaded {result['downloaded']} files to {output_dir}")
                else:
                    print_info(f"  No files matched patterns")

        # Download datasets
        datasets = parsed_config.get("datasets", [])
        if datasets and not models_only:
            dataset_downloader = DatasetDownloader(auth)

            for dataset_config in datasets:
                repo_id = dataset_config["repo"]
                file_patterns = dataset_config["files"]
                output_dir = dataset_config["output_dir"]

                print_info(f"Processing dataset: {repo_id}")

                if dry_run:
                    print_info(f"[DRY RUN] Would download from {repo_id}")

                result = dataset_downloader.download_repo(
                    repo_id=repo_id,
                    file_patterns=file_patterns,
                    output_dir=output_dir,
                    dry_run=dry_run
                )

                if result["files"]:
                    print_info(f"  Files: {len(result['files'])} matched")
                    if dry_run:
                        for f in result["files"][:5]:
                            print_info(f"    - {f}")
                        if len(result["files"]) > 5:
                            print_info(f"    ... and {len(result['files']) - 5} more")
                    else:
                        print_success(f"  Downloaded {result['downloaded']} files to {output_dir}")
                else:
                    print_info(f"  No files matched patterns")

        if dry_run:
            print_success("Dry run complete - no files downloaded")
        else:
            print_success("All downloads complete")

        sys.exit(0)

    except ConfigValidationError as e:
        print_error(f"Configuration error: {e}")
        sys.exit(1)
    except AuthError as e:
        print_error(f"Authentication error: {e}")
        sys.exit(1)
    except DownloadError as e:
        print_error(f"Download error: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    cli()
