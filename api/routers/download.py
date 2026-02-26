"""Download router — streams HuggingFace download logs via SSE."""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.store import job_store, JobStatus

router = APIRouter()
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    repo: str
    files: List[str] = ["*"]
    output_dir: str = "./models"


class DatasetConfig(BaseModel):
    repo: str
    files: List[str] = ["*"]
    output_dir: str = "./data"


class DownloadRequest(BaseModel):
    hf_token: str
    models: List[ModelConfig] = []
    datasets: List[DatasetConfig] = []
    dry_run: bool = False
    models_only: bool = False
    datasets_only: bool = False


async def run_download(req: DownloadRequest):
    """Run download in a thread and stream logs."""
    import json
    import threading

    loop = asyncio.get_event_loop()
    log_queue: asyncio.Queue = asyncio.Queue()

    def sync_log(msg: str):
        asyncio.run_coroutine_threadsafe(log_queue.put(msg), loop)
        asyncio.run_coroutine_threadsafe(job_store.emit(msg), loop)

    def sync_progress(current: int, total: int, label: str):
        event = json.dumps({"type": "progress", "current": current, "total": total, "label": label})
        asyncio.run_coroutine_threadsafe(log_queue.put(event), loop)

    def do_download():
        try:
            from lexalign.downloader.auth import AuthManager, AuthError
            from lexalign.downloader.model_downloader import ModelDownloader
            from lexalign.downloader.dataset_downloader import DatasetDownloader

            sync_log("🔐 Validating HuggingFace token...")
            auth = AuthManager(req.hf_token)
            if not auth.validate_token():
                sync_log("❌ Invalid HuggingFace token")
                asyncio.run_coroutine_threadsafe(log_queue.put(None), loop)
                return

            sync_log("✅ Authentication successful")

            items = []
            if req.models and not req.datasets_only:
                items += [("model", m) for m in req.models]
            if req.datasets and not req.models_only:
                items += [("dataset", d) for d in req.datasets]

            total = len(items)
            model_downloader = None
            dataset_downloader = None

            for idx, (kind, item) in enumerate(items):
                sync_progress(idx + 1, total, f"{item.repo}")
                sync_log(f"⬇️  Downloading {kind}: {item.repo}")

                if req.dry_run:
                    sync_log(f"   [DRY RUN] Would download {item.repo} → {item.output_dir}")
                else:
                    if kind == "model":
                        if model_downloader is None:
                            model_downloader = ModelDownloader(auth)
                        result = model_downloader.download_repo(
                            repo_id=item.repo,
                            file_patterns=item.files,
                            output_dir=item.output_dir,
                            dry_run=False,
                        )
                    else:
                        if dataset_downloader is None:
                            dataset_downloader = DatasetDownloader(auth)
                        result = dataset_downloader.download_repo(
                            repo_id=item.repo,
                            file_patterns=item.files,
                            output_dir=item.output_dir,
                            dry_run=False,
                        )
                    sync_log(f"   ✅ Downloaded {result.get('downloaded', 0)} files → {item.output_dir}")

            sync_progress(total, total, "Complete")
            sync_log("🎉 Download complete!")
            asyncio.run_coroutine_threadsafe(_set_done(), loop)
        except Exception as e:
            sync_log(f"❌ Error: {e}")
            asyncio.run_coroutine_threadsafe(_set_error(), loop)
        finally:
            asyncio.run_coroutine_threadsafe(log_queue.put(None), loop)

    async def _set_done():
        job_store.status = JobStatus.DONE

    async def _set_error():
        job_store.status = JobStatus.ERROR

    thread = threading.Thread(target=do_download, daemon=True)
    thread.start()

    while True:
        msg = await log_queue.get()
        if msg is None:
            break
        yield msg


@router.post("/download")
async def download(req: DownloadRequest):
    if job_store.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="A job is already running")

    job_store.reset()
    job_store.job_type = "download"

    async def event_generator():
        async for line in run_download(req):
            yield {"data": line}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
