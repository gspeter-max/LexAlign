"""Files router — list downloaded models and datasets."""

import os
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class FileEntry(BaseModel):
    name: str
    path: str
    size_mb: float


def _scan_directory(base_dir: str) -> List[dict]:
    """Scan a directory and return list of subdirectories with their sizes."""
    entries = []
    base = Path(base_dir)
    if not base.exists():
        return entries

    for item in sorted(base.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            total_size = sum(
                f.stat().st_size for f in item.rglob("*") if f.is_file()
            )
            entries.append({
                "name": item.name,
                "path": str(item),
                "size_mb": round(total_size / (1024 * 1024), 2),
            })
        elif item.is_file():
            entries.append({
                "name": item.name,
                "path": str(item),
                "size_mb": round(item.stat().st_size / (1024 * 1024), 2),
            })
    return entries


@router.get("/files/models")
async def list_models():
    """List downloaded models in ./models/ directory."""
    return _scan_directory("./models")


@router.get("/files/datasets")
async def list_datasets():
    """List downloaded datasets in ./data/ directory."""
    return _scan_directory("./data")


@router.get("/files/checkpoints")
async def list_checkpoints():
    """List training checkpoints in ./checkpoints/ directory."""
    return _scan_directory("./checkpoints")
