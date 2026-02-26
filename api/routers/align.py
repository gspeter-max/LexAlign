"""Align router — streams DPO/GDPO alignment logs via SSE."""

import asyncio
import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.store import job_store, JobStatus

router = APIRouter()
logger = logging.getLogger(__name__)


class AlignRequest(BaseModel):
    model_path: str
    dataset_path: str
    method: str = "dpo"           # "dpo" or "gdpo"
    device: str = "cpu"
    beta: float = 0.1
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 1
    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    output_dir: str = "./checkpoints/aligned"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    dry_run: bool = False


def _build_align_config(req: AlignRequest) -> dict:
    return {
        "model": {"path": req.model_path},
        "dataset": {
            "path": req.dataset_path,
            "prompt_field": req.prompt_field,
            "chosen_field": req.chosen_field,
            "rejected_field": req.rejected_field,
        },
        "alignment": {
            "method": req.method,
            "beta": req.beta,
            "learning_rate": req.learning_rate,
            "batch_size": req.batch_size,
            "num_epochs": req.num_epochs,
            "output_dir": req.output_dir,
            "use_lora": req.use_lora,
            "lora_r": req.lora_r,
            "lora_alpha": req.lora_alpha,
        },
        "device": req.device,
    }


async def run_align(req: AlignRequest):
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

    def do_align():
        try:
            if req.dry_run:
                sync_log("📋 Alignment Configuration (Dry Run):")
                sync_log(f"   Model:    {req.model_path}")
                sync_log(f"   Dataset:  {req.dataset_path}")
                sync_log(f"   Method:   {req.method.upper()}")
                sync_log(f"   Device:   {req.device}")
                sync_log(f"   Beta:     {req.beta}")
                sync_log(f"   LR:       {req.learning_rate}")
                sync_log(f"   Output:   {req.output_dir}")
                sync_progress(req.num_epochs, req.num_epochs, 100)
                sync_log("✅ Dry run complete — no training performed.")
                asyncio.run_coroutine_threadsafe(_set_done(), loop)
                asyncio.run_coroutine_threadsafe(log_queue.put(None), loop)
                return

            config = _build_align_config(req)
            sync_log(f"🚀 Starting {req.method.upper()} alignment on {req.device}...")
            sync_progress(0, req.num_epochs, 0)

            from pathlib import Path
            from lexalign.aligner.dataset_prep import PreferenceDataset
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType

            sync_log("📂 Loading preference dataset...")
            dataset_prep = PreferenceDataset()
            train_dataset = dataset_prep.load_and_validate(config["dataset"])
            sync_log(f"   Loaded {len(train_dataset)} preference pairs")

            sync_log("🤖 Loading model and tokenizer...")
            model_path = req.model_path
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            ref_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            for param in ref_model.parameters():
                param.requires_grad = False

            if req.use_lora:
                sync_log("🔧 Applying LoRA...")
                lora_cfg = LoraConfig(
                    r=req.lora_r,
                    lora_alpha=req.lora_alpha,
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_cfg)

            model.to(req.device)
            ref_model.to(req.device)

            sync_log(f"🏋️  Training with {req.method.upper()}...")
            if req.method == "dpo":
                from lexalign.aligner.dpo_trainer import DPOTrainerWrapper
                trainer = DPOTrainerWrapper(model, ref_model, tokenizer, config["alignment"])
            else:
                from lexalign.aligner.gdpo_trainer import GDPOTrainerWrapper
                trainer = GDPOTrainerWrapper(model, ref_model, tokenizer, config["alignment"])

            trainer.train(train_dataset)
            trainer.save_model(req.output_dir)
            sync_progress(req.num_epochs, req.num_epochs, 100)
            sync_log(f"🎉 Alignment complete! Saved to: {req.output_dir}")
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

    thread = threading.Thread(target=do_align, daemon=True)
    thread.start()

    while True:
        msg = await log_queue.get()
        if msg is None:
            break
        yield msg


@router.post("/align")
async def align(req: AlignRequest):
    if job_store.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="A job is already running")

    job_store.reset()
    job_store.job_type = "align"

    async def event_generator():
        async for line in run_align(req):
            yield {"data": line}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
