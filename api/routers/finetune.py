"""Fine-tune router — streams training logs via SSE."""

import asyncio
import logging
import threading
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.store import job_store, JobStatus

router = APIRouter()
logger = logging.getLogger(__name__)


class FinetuneRequest(BaseModel):
    model_path: str
    dataset_path: str
    method: str = "lora"          # "lora" or "qlora"
    device: str = "cpu"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 3e-4
    batch_size: int = 4
    num_epochs: int = 3
    max_seq_length: int = 512
    quantization_bits: int = 4
    output_dir: str = "./checkpoints/finetuned"
    dry_run: bool = False


def _build_config(req: FinetuneRequest) -> dict:
    return {
        "model": {"path": req.model_path},
        "dataset": {"path": req.dataset_path, "format": "auto", "text_field": "text", "train_split": "train"},
        "training": {
            "method": req.method,
            "lora_r": req.lora_r,
            "lora_alpha": req.lora_alpha,
            "lora_dropout": req.lora_dropout,
            "learning_rate": req.learning_rate,
            "batch_size": req.batch_size,
            "num_epochs": req.num_epochs,
            "max_seq_length": req.max_seq_length,
            "quantization_bits": req.quantization_bits,
            "output_dir": req.output_dir,
        },
        "device": req.device,
    }


async def run_finetune(req: FinetuneRequest):
    import json

    loop = asyncio.get_event_loop()
    log_queue: asyncio.Queue = asyncio.Queue()

    def sync_log(msg: str):
        asyncio.run_coroutine_threadsafe(log_queue.put(msg), loop)
        asyncio.run_coroutine_threadsafe(job_store.emit(msg), loop)

    def sync_progress(epoch: int, total_epochs: int, percent: int):
        event = json.dumps({"type": "progress", "epoch": epoch, "total_epochs": total_epochs, "percent": percent})
        asyncio.run_coroutine_threadsafe(log_queue.put(event), loop)

    def sync_metrics(epoch: int, loss: float, lr: float):
        event = json.dumps({"type": "metrics", "epoch": epoch, "loss": round(loss, 4), "lr": lr})
        asyncio.run_coroutine_threadsafe(log_queue.put(event), loop)

    def do_finetune():
        try:
            config = _build_config(req)

            if req.dry_run:
                sync_log("📋 Fine-tune Configuration (Dry Run):")
                sync_log(f"   Model:    {req.model_path}")
                sync_log(f"   Dataset:  {req.dataset_path}")
                sync_log(f"   Method:   {req.method.upper()}")
                sync_log(f"   Device:   {req.device}")
                sync_log(f"   LoRA r:   {req.lora_r}")
                sync_log(f"   LR:       {req.learning_rate}")
                sync_log(f"   Epochs:   {req.num_epochs}")
                sync_log(f"   Output:   {req.output_dir}")
                sync_progress(req.num_epochs, req.num_epochs, 100)
                sync_log("✅ Dry run complete — no training performed.")
                asyncio.run_coroutine_threadsafe(_set_done(), loop)
                asyncio.run_coroutine_threadsafe(log_queue.put(None), loop)
                return

            sync_log(f"🚀 Starting {req.method.upper()} fine-tuning on {req.device}...")
            sync_progress(0, req.num_epochs, 0)
            from lexalign.finetuner.trainer import FinetuneTrainer, TrainerError

            trainer = FinetuneTrainer(config, verbose=True)
            output_dir = trainer.train()
            sync_progress(req.num_epochs, req.num_epochs, 100)
            sync_log(f"🎉 Fine-tuning complete! Saved to: {output_dir}")
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

    thread = threading.Thread(target=do_finetune, daemon=True)
    thread.start()

    while True:
        msg = await log_queue.get()
        if msg is None:
            break
        yield msg


@router.post("/finetune")
async def finetune(req: FinetuneRequest):
    if job_store.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="A job is already running")

    job_store.reset()
    job_store.job_type = "finetune"

    async def event_generator():
        async for line in run_finetune(req):
            yield {"data": line}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
