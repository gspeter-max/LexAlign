"""Status router — current job state."""

from fastapi import APIRouter
from api.store import job_store

router = APIRouter()


@router.get("/status")
async def get_status():
    return {
        "status": job_store.status,
        "job_type": job_store.job_type,
        "logs": job_store.logs[-100:],
    }
